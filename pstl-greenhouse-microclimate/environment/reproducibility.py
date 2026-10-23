from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pstl_experiments.common import set_seed

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=42); p.add_argument('--deterministic',action='store_true'); a=p.parse_args()
    set_seed(a.seed,a.deterministic); print(f'seed={a.seed}; deterministic={a.deterministic}')
