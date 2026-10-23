#!/usr/bin/env python3
"""Paired seed-level Wilcoxon tests for baseline and feature-mode comparisons."""
import argparse,csv,sys,itertools
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from scipy.stats import wilcoxon

def add_test(rows,meta,a,b,family,feature_mode=''):
    common=sorted(set(a.index)&set(b.index))
    if len(common)<2: return
    x=a.loc[common].to_numpy(); y=b.loc[common].to_numpy()
    try: stat,p=wilcoxon(x,y,alternative='two-sided',zero_method='wilcox',method='auto')
    except ValueError: stat,p=float('nan'),1.0
    row=dict(meta); row.update(test_family=family,feature_mode=feature_mode,reference=str(a.name),comparison=str(b.name),
        n_pairs=len(common),wilcoxon_statistic=float(stat),p_raw=float(p))
    rows.append(row)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--metrics',default='results/summaries/run_metrics.csv')
    ap.add_argument('--metric',default='RMSE'); ap.add_argument('--reference',default='pstl')
    ap.add_argument('--output',default='results/summaries/paired_tests.csv'); a=ap.parse_args()
    df=pd.read_csv(a.metrics); df=df[(df.metric==a.metric)&(df.target.isin(['temperature','humidity','light']))]
    rows=[]; base_keys=['experiment','history','horizon','target','metric']
    # PSTL versus baselines, paired by seed under the same feature treatment.
    for key,g in df.groupby(['experiment','history','horizon','feature_mode','target','metric']):
        exp,h,p,mode,target,metric=key
        ref=g[g.model==a.reference].set_index('seed').value; ref.name=a.reference
        if ref.empty: continue
        for model in sorted(set(g.model)-{a.reference}):
            comp=g[g.model==model].set_index('seed').value; comp.name=model
            add_test(rows,dict(zip(base_keys,(exp,h,p,target,metric))),ref,comp,'model_vs_pstl',mode)
    # Paired ablation comparisons across feature treatments using the same PSTL seed.
    for key,g in df[df.model=='pstl'].groupby(['experiment','history','horizon','target','metric']):
        exp,h,p,target,metric=key; modes=sorted(g.feature_mode.unique())
        if len(modes)<2: continue
        meta=dict(zip(base_keys,(exp,h,p,target,metric)))
        by_mode={m:g[g.feature_mode==m].set_index('seed').value for m in modes}
        for m0,m1 in itertools.combinations(modes,2):
            a0=by_mode[m0]; a0.name=m0; a1=by_mode[m1]; a1.name=m1
            add_test(rows,meta,a0,a1,'feature_mode_pair',f'{m0}__vs__{m1}')
    if not rows: raise SystemExit('Not enough paired seed-level runs for statistical tests.')
    # Holm adjustment within experiment × test family × metric × target.
    corrections=defaultdict(list)
    for i,row in enumerate(rows): corrections[(row['experiment'],row['test_family'],row['metric'],row['target'])].append(i)
    for ids in corrections.values():
        order=sorted(ids,key=lambda i:rows[i]['p_raw']); m=len(order); running=0.
        for rank,i in enumerate(order):
            running=max(running,min(1.0,(m-rank)*rows[i]['p_raw'])); rows[i]['p_holm']=running
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0].keys())
    with out.open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {len(rows)} paired tests (seed is the replication unit) to {out}')
if __name__=='__main__': main()
