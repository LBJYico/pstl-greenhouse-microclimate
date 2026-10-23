#!/usr/bin/env python3
"""Queue the configured matrix. Dry-run unless --execute is supplied."""
import argparse,csv,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pstl_experiments.config import experiment_paths,resolve_config_path
from pstl_experiments.common import read_yaml

MODELS=['pstl','cnn_lstm','gcn','gconv','gat','transformer']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/experiments/main_forecasting.yaml')
    ap.add_argument('--execute',action='store_true',help='Launch the queued training matrix; default only prints the plan.')
    ap.add_argument('--device',default='auto'); ap.add_argument('--seeds',nargs='*',type=int)
    ap.add_argument('--allow-unverified-layout',action='store_true')
    a=ap.parse_args(); a.config=str(resolve_config_path(a.config)); ec,dc,pc,sp,mc=experiment_paths(a.config); protocol=read_yaml(pc)
    seeds=a.seeds or protocol.get('seeds',[42]); modes=ec.get('feature_modes',[ec.get('feature_mode','main_solar')])
    models=['pstl'] if 'feature_modes' in ec else MODELS
    outroot=Path(ec.get('output_root','../../runs')); outroot=(Path(a.config).resolve().parent/outroot).resolve() if not outroot.is_absolute() else outroot
    dataset_root=outroot.parent/'datasets'/ec['name']; project_root=Path(__file__).resolve().parents[1]
    jobs=[]
    for h in ec.get('history_lengths',protocol.get('history_lengths',[6,12,24])):
        for p in ec.get('horizons',protocol.get('horizons',[6,12,24])):
            for mode in modes:
                for model in models:
                    for seed in seeds:
                        jobs.append((int(h),int(p),str(mode),str(model),int(seed)))
    print(f'Planned training runs: {len(jobs)}; device={a.device}; execute={a.execute}')
    if not a.execute:
        for h,p,mode,model,seed in jobs: print(f'{ec["name"]} history={h} horizon={p} mode={mode} model={model} seed={seed}')
        print('No work launched. Add --execute to run this matrix.')
        return
    if a.device=='auto': print('Device auto-detection will select CUDA, then MPS, otherwise CPU.')
    matrix_log=outroot/ec['name']/'matrix_runs.csv'; matrix_log.parent.mkdir(parents=True,exist_ok=True)
    fields=['history','horizon','feature_mode','model','seed','status','return_code','started','finished','run_dir','detail']
    if not matrix_log.exists():
        with matrix_log.open('w',newline='',encoding='utf-8') as fp: csv.DictWriter(fp,fieldnames=fields).writeheader()
    _wait_for_active_runs(outroot/ec['name'])
    for h,p,mode,model,seed in jobs:
        dsdir=dataset_root/f'{mode}_h{h}_p{p}'
        if not (dsdir/'manifest.json').exists():
            build=[sys.executable,'scripts/01_build_dataset.py','--config',a.config,'--history',str(h),'--horizon',str(p),
                   '--feature-mode',mode,'--device',a.device]
            if a.allow_unverified_layout: build.append('--allow-unverified-layout')
            ret=subprocess.run(build,cwd=project_root)
            if ret.returncode:
                _record(matrix_log,fields,dict(history=h,horizon=p,feature_mode=mode,model='-',seed='-',status='dataset_failed',
                    return_code=ret.returncode,started='',finished=time.strftime('%Y-%m-%dT%H:%M:%S'),run_dir=str(dsdir),detail='dataset build failed'))
                continue
        run=outroot/ec['name']/f'h{h}_p{p}'/model/mode/f'seed_{seed}'
        manifest=run/'manifest.json'; existing={}
        if manifest.exists():
            try: existing=json.loads(manifest.read_text(encoding='utf-8'))
            except Exception: existing={}
        while existing.get('status')=='running':
            last=run/'checkpoint_last.pt'
            if last.exists() and time.time()-last.stat().st_mtime>900: break
            print(f'Waiting for existing run to finish: {run}',flush=True); time.sleep(30)
            try: existing=json.loads(manifest.read_text(encoding='utf-8'))
            except Exception: existing={}
        if existing.get('status')=='completed':
            _record(matrix_log,fields,dict(history=h,horizon=p,feature_mode=mode,model=model,seed=seed,status='skipped_completed',
                return_code=0,started='',finished=time.strftime('%Y-%m-%dT%H:%M:%S'),run_dir=str(run),detail='existing completed run'))
            continue
        if existing.get('status')=='failed':
            _record(matrix_log,fields,dict(history=h,horizon=p,feature_mode=mode,model=model,seed=seed,status='skipped_failed',
                return_code=1,started='',finished=time.strftime('%Y-%m-%dT%H:%M:%S'),run_dir=str(run),detail='preserved failed run; inspect manifest'))
            continue
        command=[sys.executable,'scripts/02_train.py','--config',a.config,'--history',str(h),'--horizon',str(p),
                 '--feature-mode',mode,'--model',model,'--seed',str(seed),'--device',a.device]
        if a.allow_unverified_layout: command.append('--allow-unverified-layout')
        if existing.get('status')=='running' and (run/'checkpoint_last.pt').exists(): command.append('--resume')
        started=time.strftime('%Y-%m-%dT%H:%M:%S'); print(f'RUN {model} h{h} p{p} {mode} seed={seed}',flush=True)
        ret=subprocess.run(command,cwd=project_root); finished=time.strftime('%Y-%m-%dT%H:%M:%S')
        status='completed' if ret.returncode==0 else 'failed'
        _record(matrix_log,fields,dict(history=h,horizon=p,feature_mode=mode,model=model,seed=seed,status=status,
            return_code=ret.returncode,started=started,finished=finished,run_dir=str(run),detail=''))
    print(f'Matrix queue finished. Run ledger: {matrix_log}')

def _record(path,fields,row):
    with path.open('a',newline='',encoding='utf-8') as fp:
        csv.DictWriter(fp,fieldnames=fields).writerow(row)

def _wait_for_active_runs(experiment_root):
    while True:
        active=[]
        for manifest in Path(experiment_root).rglob('manifest.json'):
            try: row=json.loads(manifest.read_text(encoding='utf-8'))
            except Exception: continue
            if row.get('status')!='running': continue
            run=manifest.parent; last=run/'checkpoint_last.pt'
            marker=last if last.exists() else manifest
            if time.time()-marker.stat().st_mtime>1200: continue
            active.append(run)
        if not active: return
        print(f'Waiting for {len(active)} existing training run(s) before starting CPU matrix.',flush=True)
        time.sleep(30)

if __name__=='__main__': main()
