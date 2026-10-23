#!/usr/bin/env python3
"""Wait for the main matrix, then queue XAI ablations and state-response analysis."""
import argparse,csv,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pstl_experiments.config import resolve_config_path

def row_count(path):
    if not path.exists(): return 0
    with path.open(newline='',encoding='utf-8') as fp: return max(0,sum(1 for _ in csv.reader(fp))-1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cpu'); ap.add_argument('--main-runs',type=int,default=270)
    ap.add_argument('--include-state-response',action='store_true',default=True); a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]; py=sys.executable
    main_ledger=root/'runs/main_forecasting/matrix_runs.csv'
    print(f'Waiting until main matrix ledger has {a.main_runs} entries: {main_ledger}',flush=True)
    while row_count(main_ledger)<a.main_runs:
        print(f'Main matrix progress: {row_count(main_ledger)}/{a.main_runs} terminal run records',flush=True)
        time.sleep(60)
    # State response uses a saved main PSTL checkpoint and does not launch training.
    state_run=root/'runs/main_forecasting/h12_p6/pstl/main_solar/seed_42'
    if a.include_state_response and (state_run/'manifest.json').exists():
        import json
        try: state=json.loads((state_run/'manifest.json').read_text(encoding='utf-8'))
        except Exception: state={}
        if state.get('status')=='completed' and (state_run/'checkpoint_best.pt').exists():
            cmd=[py,'scripts/05_analyze_state_response.py','--run-dir',str(state_run),'--device',a.device]
            subprocess.run(cmd,cwd=root,check=False)
        else: print(f'State analysis deferred; checkpoint status is {state.get("status")}.',flush=True)
    for cfg in ('configs/experiments/solar_geometry_vs_calendar.yaml','configs/experiments/component_ablation.yaml'):
        cfg=resolve_config_path(root/cfg)
        cmd=[py,'scripts/04_run_experiment_matrix.py','--config',str(cfg),'--execute','--device',a.device]
        print(f'Launching supplementary matrix: {cfg}',flush=True)
        subprocess.run(cmd,cwd=root,check=False)
    print('All queued supplementary experiment matrices have returned.',flush=True)
if __name__=='__main__': main()
