#!/usr/bin/env python3
"""Synthetic shape/gradient smoke check; does not read data or write model results."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from pstl_experiments.models import build_model
from pstl_experiments.common import select_device

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='auto'); a=ap.parse_args(); d=select_device(a.device)
    torch.manual_seed(17); cfg={'d_model':24,'d_state':4,'n_layers':1,'n_heads':4,'dropout':0.0}
    x=torch.randn(2,5,27,8,device=d); y=torch.randn(2,3,27,3,device=d)
    coords=np.random.default_rng(17).random((27,3),dtype=np.float32)
    for name in ('pstl','cnn_lstm','transformer','gcn','gconv','gat'):
        model=build_model(name,8,27,3,3,coords,cfg).to(d); pred=model(x); loss=(pred-y).square().mean(); loss.backward()
        if pred.shape!=y.shape or not torch.isfinite(pred).all() or not torch.isfinite(loss): raise RuntimeError(f'{name}: invalid output/loss')
        if name=='pstl':
            model.zero_grad(set_to_none=True); _=model(x,capture=True)
            if model.state_updates() is None or not torch.isfinite(model.state_updates()).all(): raise RuntimeError('PSTL state trace invalid')
        print(f'{name}: output={tuple(pred.shape)}, loss={loss.item():.5f}, device={d}')
if __name__=='__main__': main()
