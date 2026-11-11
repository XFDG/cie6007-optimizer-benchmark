"""All figures and tables derive from immutable per-run CSVs, not hand-entered values."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils import ROOT, config

NAMES=['sgd','adam','adamw','muon']
LABELS={'sgd':'SGD-M','adam':'Adam','adamw':'AdamW','muon':'Muon + AdamW'}
COLORS=dict(zip(NAMES,['#3477b8','#e08221','#32906b','#9a55a3']))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'figure.dpi':140})


def save(fig,name):
    fig.tight_layout()
    for ext in ['png','pdf']:
        p=ROOT/f'results/figures/{name}.{ext}'
        fig.savefig(p,dpi=220,bbox_inches='tight')
        shutil.copy2(p,ROOT/f'paper/figures/{name}.{ext}')
    plt.close(fig)


def main():
    cfg=config()
    paths=sorted((ROOT/'results/runs').glob('*/metrics.csv'))
    raw=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    raw.to_csv(ROOT/'results/metrics.csv',index=False)
    final=raw.query("phase=='final'")
    evals=pd.read_csv(ROOT/'results/final_evaluations.csv')
    records=[]
    for run_id,r in final.groupby('run_id'):
        r=r.sort_values('epoch')
        hit=r[r.val_accuracy>=cfg['threshold']]
        low=r[r.train_loss<=.3]
        e=evals[(evals.run_id==run_id)&(evals.selection=='best')].iloc[0]
        f=evals[(evals.run_id==run_id)&(evals.selection=='final')].iloc[0]
        b=evals[(evals.run_id==run_id)&(evals.selection=='budget')]
        records.append(dict(run_id=run_id,optimizer=r.optimizer.iloc[0],seed=r.seed.iloc[0],
            lr=r.lr.iloc[0],best_epoch=e.epoch,best_val_accuracy=e.val_accuracy,
            train_accuracy=e.train_accuracy,test_accuracy=e.test_accuracy,
            generalization_gap=e.generalization_gap,final_test_accuracy=f.test_accuracy,
            final_val_accuracy=f.val_accuracy,final_train_accuracy=f.train_accuracy,
            total_train_seconds=r.train_seconds.sum(),avg_epoch_seconds=r.train_seconds.mean(),
            total_wall_seconds=r.cumulative_wall_seconds.iloc[-1],
            samples_per_second=cfg['train_size']*len(r)/r.train_seconds.sum(),
            peak_memory_mb=r.peak_memory_mb.max(),
            threshold_epoch=hit.epoch.iloc[0] if len(hit) else np.nan,
            threshold_seconds=hit.cumulative_wall_seconds.iloc[0] if len(hit) else np.nan,
            loss_threshold_epoch=low.epoch.iloc[0] if len(low) else np.nan,
            budget_test_accuracy=b.test_accuracy.iloc[0] if len(b) else np.nan,
            budget_val_accuracy=b.val_accuracy.iloc[0] if len(b) else np.nan,
            budget_epoch=b.epoch.iloc[0] if len(b) else np.nan))
    runs=pd.DataFrame(records)
    runs.to_csv(ROOT/'results/run_summary.csv',index=False)
    metrics=[c for c in runs.columns if c not in ['run_id','optimizer','seed','lr']]
    summary=runs.groupby('optimizer')[metrics].agg(['mean','std','count']).reindex(NAMES)
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(ROOT/'results/summary.csv')
    tuning=raw.query("phase=='tune'").groupby(['optimizer','lr']).agg(
        best_val_accuracy=('val_accuracy','max'),final_val_accuracy=('val_accuracy','last'))
    tuning.to_csv(ROOT/'results/lr_sensitivity.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.0,2.55))
    for name in NAMES:
        r=final[final.optimizer==name].groupby('epoch')
        for ax,col,ylabel in zip(axs,['train_loss','val_accuracy'],['Training cross-entropy','Validation accuracy (%)']):
            mean=r[col].mean(); std=r[col].std(); scale=100 if col=='val_accuracy' else 1
            ax.plot(mean.index,mean*scale,label=LABELS[name],color=COLORS[name])
            ax.fill_between(mean.index,(mean-std)*scale,(mean+std)*scale,color=COLORS[name],alpha=.15)
            ax.set(xlabel='Epoch',ylabel=ylabel)
            ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    save(fig,'convergence')
    for i,(col,ylabel,filename) in enumerate([
        ('train_loss','Training cross-entropy','training_loss'),
        ('val_accuracy','Validation accuracy (%)','validation_accuracy')]):
        fig,ax=plt.subplots(figsize=(4.8,3.1))
        for name in NAMES:
            r=final[final.optimizer==name].groupby('epoch')[col]
            m,s=r.mean(),r.std(); scale=100 if col=='val_accuracy' else 1
            ax.plot(m.index,m*scale,label=LABELS[name],color=COLORS[name])
            ax.fill_between(m.index,(m-s)*scale,(m+s)*scale,alpha=.15,color=COLORS[name])
        ax.set(xlabel='Epoch',ylabel=ylabel);ax.legend(fontsize=8);ax.grid(alpha=.2)
        save(fig,filename)
    fig,axs=plt.subplots(1,2,figsize=(7.0,2.45))
    for name in NAMES:
        r=final[final.optimizer==name].groupby('epoch')
        axs[0].plot(r.cumulative_wall_seconds.mean(),r.val_accuracy.mean()*100,
                    label=LABELS[name],color=COLORS[name])
        d=tuning.loc[name].sort_index()
        axs[1].semilogx(d.index,d.best_val_accuracy*100,'o-',label=LABELS[name],color=COLORS[name])
    axs[0].axvline(cfg['time_budget_s'],color='gray',linestyle=':',lw=1)
    axs[0].set(xlabel='Cumulative train + validation time (s)',ylabel='Validation accuracy (%)')
    axs[1].set(xlabel='Main optimizer learning rate',ylabel='Best validation accuracy (%)')
    axs[1].legend(fontsize=7)
    for ax in axs: ax.grid(alpha=.2)
    save(fig,'efficiency_sensitivity')
    fig,axs=plt.subplots(1,2,figsize=(7.0,2.6))
    x=np.arange(4)
    axs[0].bar(x,summary.total_train_seconds_mean,yerr=summary.total_train_seconds_std,
               color=[COLORS[n] for n in NAMES],capsize=3)
    axs[0].set(ylabel='15-epoch training time (s)',xticks=x,xticklabels=['SGD-M','Adam','AdamW','Muon+'])
    for j,(col,label) in enumerate([('train_accuracy','Train'),('best_val_accuracy','Validation'),('test_accuracy','Test')]):
        axs[1].bar(x+(j-1)*.24,summary[col+'_mean']*100,width=.24,
                   yerr=summary[col+'_std']*100,label=label,capsize=2)
    axs[1].set(ylabel='Accuracy (%)',ylim=(75,100),xticks=x,xticklabels=['SGD-M','Adam','AdamW','Muon+'])
    axs[1].legend(fontsize=8)
    save(fig,'time_generalization')
    print(summary[[c for c in summary if c.endswith('_mean') or c=='test_accuracy_std']].to_string())

if __name__=='__main__': main()
