#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pstl_experiments.config import experiment_paths
from pstl_experiments.data import audit_data
from pstl_experiments.common import write_json

def main():
    ap=argparse.ArgumentParser(description='Read-only data and mapping audit')
    ap.add_argument('--config',default='configs/experiments/main_forecasting.yaml')
    ap.add_argument('--output',default='reports/data_audit.json')
    a=ap.parse_args(); ec,dc,pc,sp,mc=experiment_paths(a.config)
    result=audit_data(dc,sp,a.output)
    print(f"Merged rows: {result['merged_rows']} | interval median: {result['cadence_minutes']['median']} min")
    print(f"Targets: {len(result['target_columns'])} | missing target values: {result['target_missing_total']}")
    print(f"A1-A9 layout verified: {result['mapping_verified']} (coordinates: {result['spatial_coordinate_units']})")
    print(f"Audit written: {Path(a.output).resolve()}")
if __name__=='__main__': main()
