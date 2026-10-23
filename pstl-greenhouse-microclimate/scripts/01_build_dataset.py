#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pstl_experiments.config import experiment_paths,resolve_config_path
from pstl_experiments.data import load_and_merge,build_windows
from pstl_experiments.common import read_yaml,write_json

def main():
    ap=argparse.ArgumentParser(description='Build chronological, train-scaled forecasting windows')
    ap.add_argument('--config',default='configs/experiments/main_forecasting.yaml')
    ap.add_argument('--history',type=int,default=12); ap.add_argument('--horizon',type=int,default=6)
    ap.add_argument('--feature-mode'); ap.add_argument('--allow-unverified-layout',action='store_true')
    ap.add_argument('--device',default='auto',help='Accepted for matrix-runner compatibility; dataset construction is CPU-based.')
    a=ap.parse_args(); a.config=str(resolve_config_path(a.config)); ec,dc,pc,sp,mc=experiment_paths(a.config)
    df,meta,dconf,mapping=load_and_merge(dc,sp,a.allow_unverified_layout)
    split=ec.get('split',{'train':.7,'validation':.1,'test':.2})
    mode=a.feature_mode or ec.get('feature_mode','main_solar')
    arrays,scaler,dsmeta=build_windows(df,meta,mode,a.history,a.horizon,split,ec.get('include_outdoor_solar_history',True))
    outroot=Path(ec.get('output_root','../../runs'))
    if not outroot.is_absolute(): outroot=(Path(a.config).resolve().parent/outroot).resolve()
    dest=(outroot.parent/'datasets'/ec['name']/f'{mode}_h{a.history}_p{a.horizon}').resolve()
    dest.mkdir(parents=True,exist_ok=True)
    for split_name,vals in arrays.items():
        np.savez_compressed(dest/f'{split_name}.npz',X=vals['X'],Y=vals['Y'],target_start=vals['target_start'])
    np.savez_compressed(dest/'scalers.npz',**scaler)
    write_json({'data':meta,'dataset':dsmeta,'feature_mode':mode,'shapes':{k:{kk:list(vv.shape) for kk,vv in x.items()} for k,x in arrays.items()}},dest/'manifest.json')
    print(f'Dataset saved: {dest}')
    for k,v in arrays.items(): print(k, v['X'].shape, v['Y'].shape)
if __name__=='__main__': main()
