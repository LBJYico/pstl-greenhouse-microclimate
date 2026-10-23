#!/usr/bin/env python3
"""Recompute physical-unit metrics from an immutable saved prediction artifact."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pstl_experiments.common import write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); a=ap.parse_args(); run=Path(a.run_dir)
    cfg=json.loads((run/'config_resolved.json').read_text()); ds=Path(cfg['dataset']); z=np.load(ds/'scalers.npz'); pred=np.load(run/'predictions.npz')
    y=pred['y_true']; p=pred['y_pred']; mean=z['y_mean'].reshape(1,1,y.shape[2],3); std=z['y_std'].reshape(1,1,y.shape[2],3)
    y=y*std+mean; p=p*std+mean; result={}
    for j,name in enumerate(('temperature','humidity','light')):
        e=p[:,:,:,j]-y[:,:,:,j]; den=np.sum((y[:,:,:,j]-y[:,:,:,j].mean())**2)
        result[name]={'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(np.abs(e))),'R2':float(1-np.sum(e*e)/den)}
    write_json({'metrics_recomputed_from_predictions':result,'run_dir':str(run)},run/'metrics_recomputed.json')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
