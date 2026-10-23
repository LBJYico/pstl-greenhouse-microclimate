#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pstl_experiments.common import read_yaml
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/spatial/sensor_mapping.yaml'); ap.add_argument('--output',default='reports/a1_a9_layout_review.png'); a=ap.parse_args()
    cfg=read_yaml(a.config); fig,ax=plt.subplots(figsize=(8,6))
    for site,v in cfg['sites'].items():
        x,y=v['xy']; ax.scatter(x,y,s=90,c='#e9803d',edgecolor='black',zorder=3); ax.text(x,y,site,ha='center',va='center',fontsize=8,color='white',weight='bold')
        ax.text(x,y+.035,'/'.join(v['sensor_ids']),ha='center',fontsize=7,color='#2166ac')
    ax.set_title('A1–A9 greenhouse sensor sites (figure-relative; verify coordinates)')
    ax.set_xlabel('Plan-view relative x'); ax.set_ylabel('Plan-view relative y'); ax.set_xlim(0,1); ax.set_ylim(1,0); ax.set_aspect('equal'); ax.grid(alpha=.2)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); fig.tight_layout(); fig.savefig(a.output,dpi=180); print(Path(a.output).resolve())
if __name__=='__main__': main()
