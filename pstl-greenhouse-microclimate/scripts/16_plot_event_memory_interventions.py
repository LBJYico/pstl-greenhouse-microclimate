from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from scipy.stats import t, sem

parser=argparse.ArgumentParser()
parser.add_argument('--results-root',default='results/selective_event_memory_v2/full')
parser.add_argument('--output',required=True)
args=parser.parse_args()
ROOT=Path(args.results_root)
daily=pd.read_csv(ROOT/'event_date_metrics.csv')
tests=pd.read_csv(ROOT/'paired_statistics.csv')
events=['sunrise','noon','cloudy_transition'];labels=['Sunrise','Solar noon','Cloud transition']
styles={'gated':('#2369A8','o'),'ungated':('#C87819','s'),'no_recurrence':('#666666','^')}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none'})
fig=plt.figure(figsize=(6.25,6.45),constrained_layout=True)
gs=fig.add_gridspec(2,2,height_ratios=[1,1],hspace=.1,wspace=.1)
a=fig.add_subplot(gs[0,:]);b=fig.add_subplot(gs[1,0]);c=fig.add_subplot(gs[1,1])

def average_ci(v):
 m=float(v.mean());ci=float(t.ppf(.975,len(v)-1)*sem(v)) if len(v)>1 else 0
 return m,ci
for j,(mode,(color,marker)) in enumerate(styles.items()):
 for i,event in enumerate(events):
  vals=daily[(daily['mode']==mode)&(daily.event==event)].normal_recall_mse.to_numpy()
  x=i+(j-1)*.2;m,ci=average_ci(vals)
  a.scatter(np.full(len(vals),x),vals,s=10,color=color,marker=marker,alpha=.4,zorder=2)
  a.errorbar(x,m,yerr=ci,fmt=marker,color=color,ms=5,capsize=3,lw=1,zorder=3,label=mode if i==0 else None)
a.set_title('(a) Delayed recall of event observations',loc='left',fontweight='bold')
a.set_ylabel('Recall MSE')
a.set_xticks(range(3),labels)
a.legend([plt.Line2D([0],[0],marker=v[1],linestyle='',color=v[0]) for v in styles.values()],['Event gated','Continuous writes','No recurrence'],ncol=3,frameon=False,fontsize=8,loc='upper left')
a.grid(axis='y',alpha=.2);a.set_ylim(bottom=-.1)
for ax,col,family,color,title,ylabel in [(b,'matching_state_reset_recall_delta','gated_state_dependence','#2369A8','(b) Reset effect on recall','Δ MSE: reset − intact'),(c,'matching_state_reset_forecast_delta','forecast_state_dependence','#A8433F','(c) Forecast reset effect','Δ MSE: reset − intact')]:
 for i,event in enumerate(events):
  vals=daily[(daily['mode']=='gated')&(daily.event==event)][col].to_numpy();m,ci=average_ci(vals)
  ax.scatter(np.full(len(vals),i),vals,s=15,color=color,alpha=.4,zorder=2)
  ax.errorbar(i,m,yerr=ci,fmt='o',color=color,ms=5,capsize=3,lw=1.2,zorder=3)
  p=tests[(tests.family==family)&tests.comparison.str.endswith(f'({event})')].iloc[0].holm_adjusted_p
  ax.text(i,.94,f'p={p:.3f}',ha='center',va='top',fontsize=8,transform=ax.get_xaxis_transform())
 ax.set_xticks(range(3),['Sunrise','Noon','Cloud'],rotation=0)
 ax.axhline(0,color='black',lw=.7);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_title(title,loc='left',fontweight='bold',fontsize=9);ax.margins(x=.24)
b.set_ylim(-.35,6.6);c.set_ylim(-.108,.055)
outdir=Path(args.output).parent
outdir.mkdir(parents=True,exist_ok=True)
for ext in ['png','svg','tif']:
 f=outdir/f'Figure_9_event_memory.{ext}'
 kwargs={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tif' else {}
 fig.savefig(f,dpi=600 if ext=='tif' else 400,**kwargs)
 print(f)
plt.close(fig)
