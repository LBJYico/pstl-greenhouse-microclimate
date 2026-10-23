from pathlib import Path
from .common import read_yaml, resolve_path
from .common import PACKAGE_ROOT

def resolve_config_path(path):
    p=Path(path)
    if not p.is_absolute() and not p.exists(): p=PACKAGE_ROOT/p
    return p.resolve()

def experiment_paths(experiment_config):
    p=resolve_config_path(experiment_config); c=read_yaml(p); base=p.parent
    def ref(k): return resolve_path(c[k],base)
    return c,ref('data'),ref('protocol'),ref('spatial'),ref('model_config')
