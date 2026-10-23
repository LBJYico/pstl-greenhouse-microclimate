from __future__ import annotations
import json, os, random, hashlib, platform, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

def read_yaml(path):
    with open(path, encoding='utf-8') as f: return yaml.safe_load(f)

def resolve_path(path, relative_to):
    p=Path(path)
    return p if p.is_absolute() else (Path(relative_to)/p).resolve()

def write_json(obj, path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False,default=_json_default)

def _json_default(x):
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,)): return float(x)
    if isinstance(x,(np.ndarray,)): return x.tolist()
    if isinstance(x,(Path,)): return str(x)
    return str(x)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def set_seed(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True,warn_only=True)
            if hasattr(torch.backends,'cudnn'): torch.backends.cudnn.benchmark=False
    except ImportError: pass

def select_device(request='auto'):
    import torch
    if request!='auto':
        d=torch.device(request)
        if d.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
        if d.type=='mps' and not (hasattr(torch.backends,'mps') and torch.backends.mps.is_available()): raise RuntimeError('MPS requested but unavailable')
        return d
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def environment_report(device):
    import torch
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),
            'processor':platform.processor(),'torch':torch.__version__,'numpy':np.__version__,'device':str(device),
            'cuda_available':torch.cuda.is_available(),'cuda_version':torch.version.cuda,
            'mps_available':bool(hasattr(torch.backends,'mps') and torch.backends.mps.is_available()),
            'cpu_count':os.cpu_count()}
    if device.type=='cuda':
        report['gpu_name']=torch.cuda.get_device_name(device)
        report['gpu_capability']=torch.cuda.get_device_capability(device)
    return report
