#!/usr/bin/env python3
"""Extract input-conditioned state-update intensity from a completed PSTL run."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from pstl_experiments.config import experiment_paths,resolve_config_path
from pstl_experiments.common import read_yaml,select_device
from pstl_experiments.data import load_and_merge,solar_features
from pstl_experiments.models import build_model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--device',default='auto')
    ap.add_argument('--allow-unverified-layout',action='store_true'); ap.add_argument('--output')
    a=ap.parse_args(); run=Path(a.run_dir).resolve(); cfg=json.loads((run/'config_resolved.json').read_text())
    package_root=Path(__file__).resolve().parents[1]
    exp_cfg=Path(cfg['experiment_config'])
    if not exp_cfg.exists():
        exp_cfg=package_root/'configs'/'experiments'/f"{cfg.get('experiment','main_forecasting')}.yaml"
    exp_cfg=resolve_config_path(exp_cfg)
    ec,dc,pc,sp,mc=experiment_paths(exp_cfg); mcfg=read_yaml(mc)
    ds=Path(cfg['dataset'])
    if not ds.exists():
        ds=package_root/'datasets'/ec['name']/f"{cfg['feature_mode']}_h{cfg['history']}_p{cfg['horizon']}"
    dsm=json.loads((ds/'manifest.json').read_text()); layout=dsm['dataset']
    if not layout['mapping_verified'] and not a.allow_unverified_layout: raise RuntimeError('Cannot interpret spatial response with unverified layout; pass --allow-unverified-layout for exploratory output.')
    test=np.load(ds/'test.npz'); X=test['X'].astype(np.float32); starts=test['target_start']; H=cfg['history']
    device=select_device(a.device); model=build_model(cfg['model'],X.shape[-1],X.shape[2],3,cfg['horizon'],np.asarray(layout['coords']),mcfg).to(device)
    if cfg['model']!='pstl': raise ValueError('State-update analysis requires a PSTL selective-SSM checkpoint.')
    try: ck=torch.load(run/'checkpoint_best.pt',map_location='cpu',weights_only=False)
    except TypeError: ck=torch.load(run/'checkpoint_best.pt',map_location='cpu')
    model.load_state_dict(ck['model']); model.eval()
    # Reference scan is portable, but capture windows in modest batches to bound memory.
    updates=[]; batch=16
    with torch.no_grad():
        for s in range(0,len(X),batch):
            xb=torch.from_numpy(X[s:s+batch]).to(device); _=model(xb,capture=True)
            u=model.state_updates().cpu().numpy() # [layer, batch*node, history]
            u=u.reshape(u.shape[0],xb.shape[0],xb.shape[2],xb.shape[1]) # layer,batch,node,time
            updates.append(u.mean(axis=2)) # layer,batch,time; nodes macro-averaged
    upd=np.concatenate(updates,axis=1) # layer,sample,time
    # Recover timestamps and event labels from the same raw sources and recorded indices.
    raw,meta,dconf,_=load_and_merge(dc,sp,allow_unverified=True); ts=pd.DatetimeIndex(raw[meta['timestamp_column']])
    solar=solar_features(ts,dconf['latitude_deg'],dconf['longitude_deg'],dconf['timezone_offset_hours'])
    altitude=np.rad2deg(np.arcsin(np.clip(solar['solar_altitude_sin'].to_numpy(),-1,1)))
    event_rows=[]
    # sunrise: first below-to-above horizon crossing each day; noon: daily solar-altitude maximum.
    work=pd.DataFrame({'time':ts,'alt':altitude})
    work['day']=work.time.dt.date
    for day,g in work.groupby('day',sort=True):
        ids=g.index.to_numpy(); aalt=g.alt.to_numpy()
        crosses=np.where((aalt[1:]>0)&(aalt[:-1]<=0))[0]+1
        if len(crosses): event_rows.append(('sunrise',int(ids[crosses[0]]),str(day)))
        if np.max(aalt)>0: event_rows.append(('noon',int(ids[int(np.argmax(aalt))]),str(day)))
    if 'light' in raw:
        light=pd.to_numeric(raw.light,errors='coerce').to_numpy(float)
        med=pd.Series(light).rolling(6,min_periods=3,center=False).median().to_numpy()
        ratio=np.divide(light,med,out=np.ones_like(light),where=np.isfinite(med)&(med>1))
        # Objective candidate: daytime outdoor-light decline >=50% from preceding 60-min median.
        candidates=np.where((altitude>5)&(med>100)&(ratio<=.5))[0]
        last=-10**9
        for idx in candidates:
            if idx-last>=6: event_rows.append(('cloudy_transition',int(idx),str(ts[idx].date()))); last=idx
    # Match the forecast-origin timestamp to the nearest held-out context endpoint (within 30 min).
    endpoints=starts-1; records=[]
    for kind,event_idx,day in event_rows:
        dist=np.abs(endpoints-event_idx); j=int(np.argmin(dist)) if len(dist) else -1
        if j<0 or dist[j]>3: continue
        for layer in range(upd.shape[0]):
            for step,val in enumerate(upd[layer,j]):
                records.append({'event':kind,'event_date':day,'event_timestamp':str(ts[event_idx]),
                                'matched_context_end':str(ts[endpoints[j]]),'offset_steps':int(endpoints[j]-event_idx),
                                'sample_index':j,'layer':layer+1,'history_step':step+1,
                                'state_update_intensity':float(val)})
    if not records: raise RuntimeError('No event could be matched to a held-out history window. Check test period and event thresholds.')
    out=Path(a.output) if a.output else run/'state_response'; out.mkdir(parents=True,exist_ok=True)
    with (out/'state_update_events.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    pd.DataFrame(records).groupby(['event','layer','history_step']).state_update_intensity.agg(['mean','std','count']).reset_index().to_csv(out/'state_update_summary.csv',index=False)
    (out/'event_detection.json').write_text(json.dumps({'sunrise':'solar altitude crossing <=0 to >0','noon':'daily maximum solar altitude',
        'cloudy_transition':'outdoor light <=50% of trailing six-sample median while solar altitude >5 deg; candidates separated by at least six samples',
        'matching':'nearest held-out context endpoint within three samples','records':len(records)},indent=2),encoding='utf-8')
    print(f'Saved event-level state updates to {out}')
if __name__=='__main__': main()
