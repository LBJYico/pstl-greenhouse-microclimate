#!/usr/bin/env python3
import argparse,csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--metrics',default='results/summaries/run_metrics.csv'); ap.add_argument('--spatial-metrics',default='results/summaries/spatial_metrics.csv'); ap.add_argument('--spatial-config',default='configs/spatial/sensor_mapping.yaml'); ap.add_argument('--runs-root',default='runs'); ap.add_argument('--spatial-history',type=int,default=12); ap.add_argument('--spatial-horizon',type=int,default=6); ap.add_argument('--spatial-feature-mode',default='main_solar'); ap.add_argument('--outdir',default='results/figures'); a=ap.parse_args()
    df=pd.read_csv(a.metrics); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    # Keep experiment, history length, and feature mode separate in every figure.
    for experiment in sorted(df.experiment.unique()):
        exp_df=df[(df.experiment==experiment)&(df.metric=='RMSE')]
        for target in ('temperature','humidity','light'):
            target_df=exp_df[exp_df.target==target]
            for history in sorted(target_df.history.unique()):
                sub=target_df[target_df.history==history]
                if experiment=='main_forecasting':
                    sub=sub[(sub.feature_mode=='main_solar')]
                    hue='model'; prefix='main'
                else:
                    sub=sub[sub.model=='pstl']
                    hue='feature_mode'; prefix=experiment
                if sub.empty: continue
                fig,ax=plt.subplots(figsize=(8,4.5))
                sns.pointplot(data=sub,x='horizon',y='value',hue=hue,errorbar='sd',dodge=.25,ax=ax)
                ax.set_xlabel('Forecast horizon (steps)'); ax.set_ylabel(f'RMSE ({target})')
                ax.set_title(f'{experiment.replace("_"," ").title()} | history={history}')
                fig.tight_layout(); base=f'{prefix}_rmse_{target}_hist{history}'
                fig.savefig(out/f'{base}.pdf'); fig.savefig(out/f'{base}.svg'); plt.close(fig)
    # Training/validation curves from actual run logs.
    for hist in Path(a.runs_root).rglob('history.csv'):
        hdf=pd.read_csv(hist)
        if hdf.empty: continue
        fig,ax=plt.subplots(figsize=(6,4)); ax.plot(hdf.epoch,hdf.train_loss,label='Train'); ax.plot(hdf.epoch,hdf.validation_loss,label='Validation')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Weighted normalized MSE'); ax.legend(); fig.tight_layout()
        dest=out/'training_curves'; dest.mkdir(parents=True,exist_ok=True)
        label='_'.join(hist.parent.relative_to(Path(a.runs_root)).parts)
        fig.savefig(dest/(label+'.pdf')); fig.savefig(dest/(label+'.svg')); plt.close(fig)
    sm=Path(a.spatial_metrics)
    if sm.exists():
        sdf=pd.read_csv(sm); sdf=sdf[(sdf.spatial_level=='sensor')&(sdf.metric=='R2')&(sdf.model=='pstl')&
            (sdf.history==a.spatial_history)&(sdf.horizon==a.spatial_horizon)&(sdf.feature_mode==a.spatial_feature_mode)]
        with open(a.spatial_config,encoding='utf-8') as fp: mapping=yaml.safe_load(fp)
        lookup={}
        for site,spec in mapping['sites'].items():
            for hi,sensor in enumerate(spec['sensor_ids']): lookup[f'{site}_sensor_{sensor}']=(spec['xy'],float(mapping['height_m'][hi]),site)
        sdf=sdf[sdf.spatial_group.isin(lookup)].copy()
        for target in ('temperature','humidity','light'):
            g=sdf[sdf.target==target]
            if g.empty: continue
            by=g.groupby('spatial_group').value.agg(['mean','std','count']).reset_index()
            fig,ax=plt.subplots(figsize=(6,5))
            for height,marker in zip(mapping['height_m'],['o','s','^']):
                rows=[r for _,r in by.iterrows() if lookup[r.spatial_group][1]==float(height)]
                if rows:
                    xs=[lookup[r.spatial_group][0][0] for r in rows]; ys=[lookup[r.spatial_group][0][1] for r in rows]
                    sc=ax.scatter(xs,ys,c=[r['mean'] for r in rows],vmin=-1,vmax=1,cmap='viridis',marker=marker,s=110,edgecolor='black',label=f'{height:g} m')
                    for r,x,y in zip(rows,xs,ys): ax.annotate(lookup[r.spatial_group][2],(x,y),xytext=(5,4),textcoords='offset points',fontsize=7)
            ax.set_xlim(0,1); ax.set_ylim(1,0); ax.set_aspect('equal'); ax.set_xlabel('Plan-view relative x'); ax.set_ylabel('Plan-view relative y')
            note='Figure-derived relative coordinates; not surveyed.' if not mapping.get('metric_coordinates_verified',False) else 'Surveyed sensor coordinates.'
            ax.set_title(f'{target.capitalize()} R² by A1–A9 sensor location\n{note}'); ax.legend(title='Sensor height'); fig.colorbar(sc,ax=ax,label='R² across test forecasts'); fig.tight_layout()
            fig.savefig(out/f'spatial_r2_{target}.pdf'); fig.savefig(out/f'spatial_r2_{target}.svg'); plt.close(fig)
    print('Figures are derived only from completed run_metrics.csv.')
if __name__=='__main__': main()
