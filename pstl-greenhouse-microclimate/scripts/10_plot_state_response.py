#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--summary',required=True); ap.add_argument('--outdir',default='results/figures/state_response'); a=ap.parse_args()
    df=pd.read_csv(a.summary); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    for event,g in df.groupby('event'):
        fig,ax=plt.subplots(figsize=(7,4.5))
        for layer,lg in g.groupby('layer'):
            lg=lg.sort_values('history_step'); ax.plot(lg.history_step,lg['mean'],label=f'Layer {layer}')
            if 'std' in lg: ax.fill_between(lg.history_step,lg['mean']-lg['std'],lg['mean']+lg['std'],alpha=.15)
        ax.set_xlabel('Historical input step'); ax.set_ylabel('Normalized state update intensity'); ax.set_title(event.replace('_',' ').title()); ax.legend(); fig.tight_layout()
        fig.savefig(out/f'{event}.pdf'); fig.savefig(out/f'{event}.svg'); plt.close(fig)
    # Combined paper figure keeps the three event windows on identical log axes.
    event_order=['sunrise','noon','cloudy_transition']
    titles={'sunrise':'Sunrise','noon':'Solar noon','cloudy_transition':'Cloudy transition'}
    colors={1:'#1768AC',2:'#E07A1F'}
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharex=True,sharey=True)
    for ax,event in zip(axes,event_order):
        g=df[df.event==event]
        if g.empty:
            ax.set_visible(False); continue
        n=int(g[g.layer==g.layer.iloc[0]]['count'].max())
        for layer in sorted(g.layer.unique()):
            lg=g[g.layer==layer].sort_values('history_step'); x=lg.history_step.to_numpy()
            mean=lg['mean'].to_numpy(); std=lg['std'].fillna(0).to_numpy()
            lo=(mean-std).clip(min=1e-16); hi=(mean+std).clip(min=1e-16)
            ax.plot(x,mean,color=colors.get(layer),marker='o',markersize=3,linewidth=1.8,label=f'Layer {layer}')
            ax.fill_between(x,lo,hi,color=colors.get(layer),alpha=.16,linewidth=0)
        ax.set_title(f'{titles[event]} (n={n})'); ax.set_xlabel('Historical input step')
        ax.set_yscale('log'); ax.grid(True,which='both',alpha=.2)
    axes[0].set_ylabel('Normalized state-update intensity (log scale)')
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.03))
    fig.suptitle('Selective state response across environmental event windows',y=1.13,fontsize=13)
    fig.tight_layout(); fig.savefig(out/'state_response_comparison.pdf',bbox_inches='tight'); fig.savefig(out/'state_response_comparison.svg',bbox_inches='tight'); plt.close(fig)
    (out/'interpretation.txt').write_text(
        'Diagnostic from one trained checkpoint (seed 42), averaged over 27 sensor nodes. '
        'Matched event windows: sunrise n=10, solar noon n=11, cloudy transition n=6. '
        'The normalized state-update intensity decays strongly across the 12 input steps, '
        'while the three event-class curves nearly overlap. This diagnostic does not show '
        'clear event-specific state modulation and should not be described as proof of '
        'selective memory effectiveness.\n',encoding='utf-8')
    print(f'Figures saved to {out}')
if __name__=='__main__': main()
