import os, re, html, json, random, time, copy, platform
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import Classifier

ROOT = Path(__file__).parent
os.chdir(ROOT)
Path('results').mkdir(exist_ok=True)
def save(name, value):
    Path('results', name+'.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
def clean(s):
    return ' '.join(re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[!?]", html.unescape(str(s)).lower()))
def seed(n=42):
    random.seed(n); np.random.seed(n); torch.manual_seed(n)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(n)

def main():
    seed(); torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save('environment', dict(python=platform.python_version(), torch=torch.__version__, device=device,
        gpu=torch.cuda.get_device_name(0) if device=='cuda' else None))
    print('Environment:', device, flush=True)
    train = pd.read_csv('data/drugsComTrain_raw.tsv', sep='\t')
    test = pd.read_csv('data/drugsComTest_raw.tsv', sep='\t')
    original = dict(train=len(train), test=len(test))
    for df in (train,test):
        df['text'] = df.review.map(clean)
        df['label'] = np.select([df.rating<=4, df.rating<=6], [0,1], default=2)
    train = train[train.text.str.len()>0].drop_duplicates('text').copy()
    test = test[test.text.str.len()>0].drop_duplicates('text').copy()
    overlap = test.text.isin(set(train.text))
    overlap_count = int(overlap.sum()); test = test[~overlap].copy()
    tr, va = train_test_split(train, test_size=.15, stratify=train.label, random_state=42)
    # Fixed stratified subsets make manual recurrent-cell experiments practical locally.
    def subset(df, n):
        return train_test_split(df, train_size=n, stratify=df.label, random_state=42)[0] if len(df)>n else df
    tr, va, te = subset(tr,12000), subset(va,2000), subset(test,4000)
    vocab = {'<pad>':0,'<unk>':1}
    vocab.update({w:i+2 for i,(w,_) in enumerate(Counter(' '.join(tr.text).split()).most_common(11998))})
    save('vocabulary',vocab)
    splits = {'train':tr,'validation':va,'test':te}
    for name,df in splits.items(): df.to_csv('results/'+name+'_split.csv',index=False)
    eda = dict(original=original, removed_test_overlap=overlap_count, cleaned_train=len(train),cleaned_test=len(test),
        splits={n:dict(size=len(d),classes={str(k):int(v) for k,v in d.label.value_counts().sort_index().items()}) for n,d in splits.items()},
        vocabulary=len(vocab),max_length=96,labels={'0':'negative (1-4)','1':'neutral (5-6)','2':'positive (7-10)'},
        train_truncation_fraction=float((tr.text.str.split().str.len()>96).mean()))
    save('data_summary',eda); print(json.dumps(eda,indent=2),flush=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    tr.label.value_counts().sort_index().plot.bar(ax=axes[0],color=['#dc6262','#e5b44d','#49a58d'])
    axes[0].set(xticklabels=['Negative','Neutral','Positive'],title='Training class distribution',ylabel='Reviews')
    axes[1].hist(tr.text.str.split().str.len(),bins=45,color='#386bac'); axes[1].axvline(96,color='red',label='96-token limit'); axes[1].legend(); axes[1].set(title='Review lengths',xlabel='Tokens')
    fig.tight_layout(); fig.savefig('results/eda.png',dpi=160); plt.close(fig)
    def encode(df):
        x=torch.zeros((len(df),96),dtype=torch.long)
        for j,s in enumerate(df.text):
            ids=[vocab.get(w,1) for w in s.split()[:96]]; x[j,:len(ids)]=torch.tensor(ids)
        return TensorDataset(x,torch.tensor(df.label.to_numpy(),dtype=torch.long))
    datasets={n:encode(d) for n,d in splits.items()}
    weights=torch.tensor(len(tr)/(3*np.bincount(tr.label)),dtype=torch.float32,device=device)
    criterion=nn.CrossEntropyLoss(weight=weights)
    def evaluate(model, name):
        model.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for x,y in DataLoader(datasets[name],batch_size=256):
                ps.extend(model(x.to(device)).argmax(1).cpu().tolist()); ys.extend(y.tolist())
        return dict(accuracy=accuracy_score(ys,ps),macro_f1=f1_score(ys,ps,average='macro',zero_division=0)),ys,ps
    runs=[]; finalists={}
    for kind in ['vanilla','cifg','wmc']:
        best=-1
        for lr,batch_size in [(.001,128),(.003,128),(.001,64)]:
            seed(); model=Classifier(len(vocab),kind).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=lr)
            history=[]; best_state=None; best_score=-1; started=time.time()
            for epoch in range(1,5):
                model.train(); total=0
                for x,y in DataLoader(datasets['train'],batch_size=batch_size,shuffle=True):
                    x,y=x.to(device),y.to(device); optimizer.zero_grad()
                    loss=criterion(model(x),y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step(); total+=loss.item()*len(y)
                metrics,_,_=evaluate(model,'validation')
                history.append(dict(epoch=epoch,loss=total/len(tr),**metrics))
                print(kind,lr,batch_size,history[-1],flush=True)
                if metrics['macro_f1']>best_score:
                    best_score=metrics['macro_f1']; best_state=copy.deepcopy(model.state_dict())
                save('progress',dict(model=kind,learning_rate=lr,batch_size=batch_size,history=history))
            run=dict(model=kind,learning_rate=lr,batch_size=batch_size,optimizer='Adam',initialization='PyTorch defaults',
                seed=42,epochs=4,embedding=64,hidden=64,parameters=sum(p.numel() for p in model.parameters()),
                best_validation_macro_f1=best_score,seconds=time.time()-started,history=history)
            runs.append(run); save('experiments',runs)
            if best_score>best:
                best=best_score; finalists[kind]=(run,best_state)
        torch.save(finalists[kind][1],'results/'+kind+'.pt')
    # Only selected configurations are evaluated on the held-out test set.
    final=[]
    for kind,(run,state) in finalists.items():
        model=Classifier(len(vocab),kind).to(device); model.load_state_dict(state)
        metrics,ys,ps=evaluate(model,'test')
        final.append(dict(model=kind,learning_rate=run['learning_rate'],batch_size=run['batch_size'],parameters=run['parameters'],**metrics))
        save(kind+'_classification',classification_report(ys,ps,target_names=['negative','neutral','positive'],output_dict=True,zero_division=0))
        pd.DataFrame(dict(text=te.text.tolist(),actual=ys,predicted=ps)).to_csv('results/'+kind+'_predictions.csv',index=False)
        fig,ax=plt.subplots(figsize=(5,4)); cm=confusion_matrix(ys,ps); ax.imshow(cm,cmap='Blues')
        for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=['Neg','Neu','Pos'],yticklabels=['Neg','Neu','Pos'],xlabel='Predicted',ylabel='Actual',title=kind.upper()+' test confusion matrix')
        fig.tight_layout();fig.savefig('results/'+kind+'_confusion.png',dpi=150);plt.close(fig)
    save('final_results',final); pd.DataFrame(final).to_csv('results/comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4))
    for run in runs: ax.plot([h['epoch'] for h in run['history']],[h['macro_f1'] for h in run['history']],label=f"{run['model']} lr={run['learning_rate']} bs={run['batch_size']}")
    ax.set(xlabel='Epoch',ylabel='Validation macro-F1',title='Architecture and learning-rate comparison');ax.legend();fig.tight_layout();fig.savefig('results/learning_curves.png',dpi=160);plt.close(fig)
    save('baseline',dict(majority_accuracy=float((te.label==tr.label.mode()[0]).mean()),majority_macro_f1=f1_score(te.label,[tr.label.mode()[0]]*len(te),average='macro')))
    print('COMPLETE',json.dumps(final,indent=2),flush=True)

if __name__=='__main__': main()
