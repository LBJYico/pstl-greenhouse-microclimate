from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pstl_experiments.common import select_device,environment_report

if __name__=='__main__': print(json.dumps(environment_report(select_device('auto')),indent=2))
