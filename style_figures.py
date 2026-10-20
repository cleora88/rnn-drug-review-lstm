"""Restyle report figures from recorded data without retraining."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).parent/'results'
ink,teal,gold='#18364B','#007F7A','#B26D16'
plt.rcParams.update({'font.family':'Arial','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':ink,'text.color':ink,'xtick.color':ink,'ytick.color':ink,'axes.titleweight':'bold','figure.facecolor':'white','axes.edgecolor':'#9AA7AD'})
tr=pd.read_csv(R/'train_split.csv')
fig,ax=plt.subplots(1,2,figsize=(11,3.8))
ax[0].bar(['Negative','Neutral','Positive'],tr.label.value_counts().sort_index().values,color=[ink,gold,teal],width=.55)
ax[0].set(title='Training sentiment distribution',ylabel='Reviews')
for i,n in enumerate(tr.label.value_counts().sort_index().values):ax[0].text(i,n+150,f'{n:,}',ha='center',fontweight='bold')
ax[0].set_ylim(0,9000)
ax[1].hist(tr.text.str.split().str.len(),bins=40,color=teal,alpha=.8)
ax[1].axvline(96,color=gold,linewidth=2,label='96-token limit');ax[1].legend(frameon=False);ax[1].set(title='Review length before truncation',xlabel='Tokens',ylabel='Reviews')
fig.tight_layout(pad=1.6);fig.savefig(R/'eda.png',dpi=190);plt.close(fig)
runs=json.loads((R/'experiments.json').read_text())
fig,ax=plt.subplots(1,3,figsize=(11,3.7),sharey=True)
for a,kind in zip(ax,['vanilla','cifg','wmc']):
    for r,color,style in zip([r for r in runs if r['model']==kind],[ink,teal,gold],['-','--',':']):
        a.plot([h['epoch'] for h in r['history']],[h['macro_f1'] for h in r['history']],marker='o',markersize=4,color=color,linestyle=style,linewidth=2,label=f"LR {r['learning_rate']} / batch {r['batch_size']}")
    a.set(title=kind.upper(),xlabel='Epoch',xticks=[1,2,3,4],ylim=(.3,.55));a.grid(axis='y',alpha=.15);a.legend(frameon=False,fontsize=8,loc='lower right')
ax[0].set_ylabel('Validation macro-F1');fig.tight_layout(pad=1.5);fig.savefig(R/'learning_curves.png',dpi=190);plt.close(fig)
for kind in ['vanilla','cifg','wmc']:
    pred=pd.read_csv(R/f'{kind}_predictions.csv');cm=pd.crosstab(pred.actual,pred.predicted).reindex(index=[0,1,2],columns=[0,1,2],fill_value=0).to_numpy()
    from matplotlib.colors import LinearSegmentedColormap
    fig,a=plt.subplots(figsize=(5,4));a.imshow(cm,cmap=LinearSegmentedColormap.from_list('uir',['#F4F5EF',teal]))
    for (i,j),n in np.ndenumerate(cm):a.text(j,i,f'{n:,}',ha='center',va='center',fontweight='bold',color='white' if n>cm.max()*.55 else ink)
    a.set(xticks=[0,1,2],yticks=[0,1,2],xticklabels=['Negative','Neutral','Positive'],yticklabels=['Negative','Neutral','Positive'],xlabel='Predicted class',ylabel='Actual class',title=f'{kind.upper()} / held-out test errors')
    fig.tight_layout(pad=1.5);fig.savefig(R/f'{kind}_confusion.png',dpi=190);plt.close(fig)
print('Report figures restyled from saved results')
