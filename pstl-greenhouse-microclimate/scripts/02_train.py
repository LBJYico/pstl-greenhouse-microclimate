#!/usr/bin/env python3
import argparse,copy,json,time,traceback,random
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from pstl_experiments.config import experiment_paths,resolve_config_path
from pstl_experiments.common import read_yaml,write_json,select_device,set_seed,environment_report,sha256
from pstl_experiments.models import build_model

def _regression_metrics(target,pred):
    e=pred-target; den=np.sum((target-target.mean())**2)
    return {'RMSE':float(np.sqrt(np.mean(e**2))),'MAE':float(np.mean(np.abs(e))),
            'R2':float(1-np.sum(e**2)/den) if den>1e-12 else float('nan')}

def atomic_torch_save(obj,path):
    tmp=Path(str(path)+'.tmp'); torch.save(obj,tmp); tmp.replace(path)

def load_checkpoint(path,map_location='cpu'):
    try: return torch.load(path,map_location=map_location,weights_only=False)
    except TypeError: return torch.load(path,map_location=map_location)

def metrics(y,p,scaler,nodes):
    # inverse transform to physical units; stored shape [sample,horizon,node,target]
    mean=scaler['y_mean'].reshape(1,1,y.shape[2],y.shape[3]); std=scaler['y_std'].reshape(1,1,y.shape[2],y.shape[3])
    yy=y*std+mean; pp=p*std+mean; result={}
    names=['temperature','humidity','light']
    for i,name in enumerate(names):
        result[name]=_regression_metrics(yy[:,:,:,i],pp[:,:,:,i])
    result['by_horizon']={}
    for h in range(y.shape[1]):
        result['by_horizon'][str(h+1)]={}
        for i,name in enumerate(names):
            result['by_horizon'][str(h+1)][name]=_regression_metrics(yy[:,h,:,i],pp[:,h,:,i])
    result['by_sensor']={}; result['by_site']={}; result['by_height']={}
    for ni,node in enumerate(nodes):
        key=f"{node['site']}_sensor_{node['sensor_id']}"; result['by_sensor'][key]={'site':node['site'],'sensor_id':node['sensor_id'],'height_m':node['height_m']}
        result['by_site'].setdefault(node['site'],{}); hk=str(node['height_m']); result['by_height'].setdefault(hk,{})
        for i,name in enumerate(names):
            vals=_regression_metrics(yy[:,:,ni,i],pp[:,:,ni,i]); result['by_sensor'][key][name]=vals
            result['by_site'][node['site']].setdefault(name,[]).append((yy[:,:,ni,i],pp[:,:,ni,i]))
            result['by_height'][hk].setdefault(name,[]).append((yy[:,:,ni,i],pp[:,:,ni,i]))
    for grouped in ('by_site','by_height'):
        for group,variables in result[grouped].items():
            result[grouped][group]={name:_regression_metrics(np.concatenate([a.ravel() for a,b in pairs]),np.concatenate([b.ravel() for a,b in pairs])) for name,pairs in variables.items()}
    return result,yy,pp

def main():
    ap=argparse.ArgumentParser(description='Train one reproducible model run')
    ap.add_argument('--config',default='configs/experiments/main_forecasting.yaml'); ap.add_argument('--model',default='pstl')
    ap.add_argument('--history',type=int,default=12); ap.add_argument('--horizon',type=int,default=6)
    ap.add_argument('--seed',type=int,default=42); ap.add_argument('--device',default='auto')
    ap.add_argument('--feature-mode'); ap.add_argument('--allow-unverified-layout',action='store_true')
    ap.add_argument('--resume',action='store_true',help='Resume from checkpoint_last.pt in this exact run directory.')
    a=ap.parse_args(); a.config=str(resolve_config_path(a.config)); ec,dc,pc,sp,mc=experiment_paths(a.config); protocol=read_yaml(pc); mcfg=read_yaml(mc)
    mode=a.feature_mode or ec.get('feature_mode','main_solar')
    outroot=Path(ec.get('output_root','../../runs')); outroot=(Path(a.config).resolve().parent/outroot).resolve() if not outroot.is_absolute() else outroot
    dsdir=(outroot.parent/'datasets'/ec['name']/f'{mode}_h{a.history}_p{a.horizon}').resolve()
    if not (dsdir/'manifest.json').exists(): raise FileNotFoundError(f'Dataset missing: run scripts/01_build_dataset.py first: {dsdir}')
    dsmeta=json.loads((dsdir/'manifest.json').read_text(encoding='utf-8')); layout=dsmeta['dataset']
    if not layout['mapping_verified'] and not a.allow_unverified_layout: raise RuntimeError('Layout is unverified; pass --allow-unverified-layout only for exploratory runs.')
    device=select_device(a.device); set_seed(a.seed)
    run=outroot/ec['name']/f'h{a.history}_p{a.horizon}'/a.model/mode/f'seed_{a.seed}'
    if (run/'manifest.json').exists() and not a.resume:
        old=json.loads((run/'manifest.json').read_text())
        if old.get('status') in ('completed','running','failed'):
            raise FileExistsError(f"Run already exists with status={old.get('status')}. Use a new seed/config directory or pass --resume for an interrupted run.")
    if a.resume and not (run/'checkpoint_last.pt').exists(): raise FileNotFoundError(f'No resumable checkpoint: {run}/checkpoint_last.pt')
    run.mkdir(parents=True,exist_ok=True); write_json(environment_report(device),run/'environment.json')
    package_root=Path(__file__).resolve().parents[1]
    code_hashes={str(p.relative_to(package_root)):sha256(p) for base in (package_root/'pstl_experiments',package_root/'scripts') for p in sorted(base.glob('*.py'))}
    config_resolved={'experiment':ec['name'],'experiment_config':str(Path(a.config).resolve()),'model':a.model,'history':a.history,'horizon':a.horizon,
                     'feature_mode':mode,'seed':a.seed,'device':str(device),'protocol':protocol,'model_config':mcfg,'dataset':str(dsdir),
                     'code_sha256':code_hashes}
    (run/'config_resolved.json').write_text(json.dumps(config_resolved,indent=2,ensure_ascii=False),encoding='utf-8')
    status={'status':'running','run_dir':str(run)}; write_json(status,run/'manifest.json')
    try:
        def load_split(s):
            d=np.load(dsdir/f'{s}.npz'); return d['X'].astype(np.float32),d['Y'].astype(np.float32),d['target_start']
        Xtr,Ytr,_=load_split('train'); Xv,Yv,_=load_split('validation'); Xte,Yte,starts=load_split('test')
        scaler={k:v for k,v in np.load(dsdir/'scalers.npz').items()}
        coords=np.asarray(layout['coords'],np.float32); model=build_model(a.model,Xtr.shape[-1],Xtr.shape[2],Ytr.shape[-1],a.horizon,coords,mcfg).to(device)
        bs=int(protocol.get('batch_size',32)); train_dl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(Ytr)),batch_size=bs,shuffle=True)
        val_dl=DataLoader(TensorDataset(torch.from_numpy(Xv),torch.from_numpy(Yv)),batch_size=bs,shuffle=False)
        test_dl=DataLoader(TensorDataset(torch.from_numpy(Xte),torch.from_numpy(Yte)),batch_size=bs,shuffle=False)
        opt=torch.optim.Adam(model.parameters(),lr=float(protocol.get('learning_rate',1e-4)),weight_decay=float(protocol.get('weight_decay',.01)))
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=int(protocol.get('epochs',100))) if protocol.get('scheduler')=='cosine_annealing' else None
        target_w=torch.tensor([protocol.get('target_weights',{}).get(n,1.) for n in ('temperature','humidity','light')],device=device).reshape(1,1,1,-1)
        best=float('inf'); best_state=None; wait=0; history=[]; start=time.time(); start_epoch=1
        if a.resume:
            state=load_checkpoint(run/'checkpoint_last.pt',map_location='cpu')
            model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
            if scheduler is not None and state.get('scheduler') is not None: scheduler.load_state_dict(state['scheduler'])
            start_epoch=int(state['epoch'])+1; best=float(state['best']); wait=int(state['wait']); history=list(state['history'])
            best_ck=load_checkpoint(run/'checkpoint_best.pt',map_location='cpu'); best_state=best_ck['model']
            if 'torch_rng' in state: torch.set_rng_state(state['torch_rng'].cpu())
            if 'numpy_rng' in state: np.random.set_state(state['numpy_rng'])
            if 'python_rng' in state: random.setstate(state['python_rng'])
            if device.type=='cuda' and state.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
        with (run/'train.log').open('a' if a.resume else 'w',encoding='utf-8') as log:
            for epoch in range(start_epoch,int(protocol.get('epochs',100))+1):
                model.train(); losses=[]
                for xb,yb in train_dl:
                    xb=xb.to(device); yb=yb.to(device); opt.zero_grad(set_to_none=True)
                    pred=model(xb); loss=(((pred-yb)**2)*target_w).mean(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),float(protocol.get('grad_clip_norm',1.)))
                    opt.step(); losses.append(float(loss.detach().cpu()))
                model.eval(); vl=[]
                with torch.no_grad():
                    for xb,yb in val_dl: vl.append(float(((((model(xb.to(device))-yb.to(device))**2)*target_w).mean()).cpu()))
                if scheduler is not None: scheduler.step()
                tr=float(np.mean(losses)); va=float(np.mean(vl)); row={'epoch':epoch,'train_loss':tr,'validation_loss':va}; history.append(row)
                line=f"epoch={epoch} train={tr:.7f} validation={va:.7f}"; print(line); log.write(line+'\n'); log.flush()
                if va<best:
                    best=va; wait=0; best_state=copy.deepcopy(model.state_dict())
                    atomic_torch_save({'model':best_state,'epoch':epoch,'config':config_resolved},run/'checkpoint_best.pt')
                else:
                    wait+=1
                atomic_torch_save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict() if scheduler else None,
                                   'epoch':epoch,'best':best,'wait':wait,'history':history,'config':config_resolved,
                                   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                                   'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},run/'checkpoint_last.pt')
                if wait>=int(protocol.get('patience',15)): break
            model.load_state_dict(best_state)
            ps=[]; ts=[]
            with torch.no_grad():
                for xb,yb in test_dl: ps.append(model(xb.to(device)).cpu().numpy()); ts.append(yb.numpy())
        pred=np.concatenate(ps); true=np.concatenate(ts); score,yphysical,pphysical=metrics(true,pred,scaler,layout['nodes'])
        np.savez_compressed(run/'predictions.npz',y_true= yphysical,y_pred=pphysical,target_start=starts)
        import csv
        with (run/'history.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=['epoch','train_loss','validation_loss']); wr.writeheader(); wr.writerows(history)
        write_json({'metrics':score,'best_validation_loss':best,'epochs_completed':len(history),'runtime_seconds':time.time()-start},run/'metrics.json')
        write_json({'status':'completed','run_dir':str(run),'checkpoint':'checkpoint_best.pt'},run/'manifest.json')
    except Exception as exc:
        write_json({'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'run_dir':str(run)},run/'manifest.json')
        raise
if __name__=='__main__': main()
