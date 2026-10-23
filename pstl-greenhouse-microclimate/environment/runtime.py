from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pstl_experiments.common import select_device

if __name__=='__main__': print(select_device('auto'))
