#!/usr/bin/env python3
import argparse,csv,json,sys,math
from collections import defaultdict
from scipy.stats import t as student_t
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runs-root',default='runs'); ap.add_argument('--output',default='results/summaries/run_metrics.csv'); a=ap.parse_args()
    root=Path(a.runs_root); rows=[]; horizon_rows=[]; spatial_rows=[]
    for f in root.rglob('metrics.json'):
        run=f.parent; manifest=run/'manifest.json'
        try:
            m=json.loads(f.read_text()); cfg=json.loads((run/'config_resolved.json').read_text())
            for target in ('temperature','humidity','light'):
                vals=m['metrics'].get(target,{})
                for metric,value in vals.items():
                    if isinstance(value,(int,float)): rows.append({'run':str(run),'experiment':cfg.get('experiment',''),
                        'model':cfg['model'],'history':cfg['history'],'horizon':cfg['horizon'],'feature_mode':cfg['feature_mode'],
                        'seed':cfg['seed'],'target':target,'metric':metric,'value':value})
            for step,targets in m['metrics'].get('by_horizon',{}).items():
                for target,vals in targets.items():
                    for metric,value in vals.items():
                        if isinstance(value,(int,float)): horizon_rows.append({'run':str(run),'experiment':cfg.get('experiment',''),
                            'model':cfg['model'],'history':cfg['history'],'horizon':cfg['horizon'],'feature_mode':cfg['feature_mode'],
                            'seed':cfg['seed'],'forecast_step':int(step),'target':target,'metric':metric,'value':value})
            for level,keyname in [('sensor','by_sensor'),('site','by_site'),('height','by_height')]:
                for group,values in m['metrics'].get(keyname,{}).items():
                    for target,metrics in values.items():
                        if not isinstance(metrics,dict): continue
                        for metric,value in metrics.items():
                            if isinstance(value,(int,float)): spatial_rows.append({'run':str(run),'experiment':cfg.get('experiment',''),
                                'model':cfg['model'],'history':cfg['history'],'horizon':cfg['horizon'],'feature_mode':cfg['feature_mode'],
                                'seed':cfg['seed'],'spatial_level':level,'spatial_group':group,'target':target,'metric':metric,'value':value})
        except Exception as e: print(f'Skipping unreadable {f}: {e}',file=sys.stderr)
    if not rows: raise SystemExit('No completed metrics.json files found; no result table written.')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if horizon_rows:
        hout=out.with_name('horizon_metrics.csv')
        with hout.open('w',newline='',encoding='utf-8') as fp:
            w=csv.DictWriter(fp,fieldnames=list(horizon_rows[0])); w.writeheader(); w.writerows(horizon_rows)
        hgroups=defaultdict(list)
        hkeys=('experiment','model','history','horizon','feature_mode','forecast_step','target','metric')
        for r in horizon_rows: hgroups[tuple(r[k] for k in hkeys)].append(float(r['value']))
        hsums=[]
        for key,vals in hgroups.items():
            n=len(vals); mean=sum(vals)/n; sd=math.sqrt(sum((x-mean)**2 for x in vals)/(n-1)) if n>1 else 0.
            half=float(student_t.ppf(.975,n-1))*sd/math.sqrt(n) if n>1 else float('nan')
            hsums.append(dict(zip(hkeys,key),n=n,mean=mean,std=sd,ci95_low=mean-half,ci95_high=mean+half))
        with out.with_name('horizon_mean_std_ci.csv').open('w',newline='',encoding='utf-8') as fp:
            w=csv.DictWriter(fp,fieldnames=list(hsums[0])); w.writeheader(); w.writerows(hsums)
    if spatial_rows:
        spatial=out.with_name('spatial_metrics.csv')
        with spatial.open('w',newline='',encoding='utf-8') as fp:
            w=csv.DictWriter(fp,fieldnames=list(spatial_rows[0])); w.writeheader(); w.writerows(spatial_rows)
    # Seed-level mean, sample SD, and t-based 95% CI; keep raw rows above.
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in ('experiment','model','history','horizon','feature_mode','target','metric'))].append(float(r['value']))
    summary=out.with_name('mean_std_ci.csv'); sums=[]
    for key,vals in groups.items():
        n=len(vals); mean=sum(vals)/n; sd=math.sqrt(sum((x-mean)**2 for x in vals)/(n-1)) if n>1 else 0.
        half=float(student_t.ppf(.975,n-1))*sd/math.sqrt(n) if n>1 else float('nan')
        sums.append(dict(zip(('experiment','model','history','horizon','feature_mode','target','metric'),key),n=n,mean=mean,std=sd,ci95_low=mean-half,ci95_high=mean+half))
    with summary.open('w',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=list(sums[0])); w.writeheader(); w.writerows(sums)
    tex=out.with_name('mean_std_ci.tex')
    with tex.open('w',encoding='utf-8') as fp:
        fp.write('% Requires \\usepackage{booktabs,longtable}\n')
        fp.write('\\begin{longtable}{llllrrrl}\n\\toprule\nExperiment & Features & Model & Target & H & P & Mean $\\pm$ SD & 95\\% CI \\\\\n\\midrule\n\\endhead\n')
        for r in sums:
            esc=lambda x:str(x).replace('_',r'\\_')
            fp.write(f"{esc(r['experiment'])} & {esc(r['feature_mode'])} & {esc(r['model'])} & {r['target']} & {r['history']} & {r['horizon']} & {r['mean']:.4g} $\\pm$ {r['std']:.3g} & [{r['ci95_low']:.4g}, {r['ci95_high']:.4g}] \\\\\n")
        fp.write('\\bottomrule\n\\end{longtable}\n')
    print(f'Wrote raw seed rows: {out}; summary: {summary}; LaTeX: {tex}')
if __name__=='__main__': main()
