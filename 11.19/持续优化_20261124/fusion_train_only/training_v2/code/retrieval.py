"""Train attribute-prompt retrieval scores across whole images and candidate regions."""
import itertools
import json

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from inputs import BASE,OUT
from train import fit_ridge,ridge,retrieval_metric


def features(data):
    prompt=torch.load(OUT/'features/query_features.pt',weights_only=True)
    qwen=json.loads((BASE/'training_v1/features/qwen_all_training_gallery.json').read_text(encoding='utf-8'))
    result={}
    for q in range(len(data.texts)):
        for i,row in enumerate(data.records):
            values=[data.retrieval[f'{q}:{i}']['scalar']]
            n=len(row['boxes'])
            for name in data.models:
                visual=data.visual[name];start=visual['records'][i]['start']
                for mode in ('ensemble','attribute','centered'):
                    text=prompt[name][mode][q]
                    whole=float(visual['visual'][start]@text)
                    scores=[]
                    for offset in (1,1+n):
                        regional=visual['visual'][start+offset:start+offset+n]@text
                        if n:
                            prior=regional+.1*row['scores'].clamp_min(1e-6).log()
                            scores.extend([regional.max(),regional.mean(),regional.topk(min(2,n)).values.mean(),prior.max()])
                        else:scores.extend([whole]*4)
                    values.append(torch.tensor([whole]+scores))
            values.append(torch.tensor([n,float(torch.tensor(qwen[f'{q}:{i}']/5).tanh())]))
            result[f'{q}:{i}']=torch.cat(values)
    return result


def fit(data,inputs,ids,config):
    assert set(ids)<=set(data.fit_ids),'Retrieval training outside original fit images'
    queries=sorted(set(data.texts.index(t) for i in ids for t in data.records[i]['texts']))
    x=torch.stack([inputs[f'{q}:{i}'] for q in queries for i in ids])
    y=torch.tensor([data.records[i]['texts'].count(data.texts[q]) for q in queries for i in ids],dtype=torch.float32)
    if config['family']=='ridge':return fit_ridge(x,y,config['alpha'])
    model=ExtraTreesRegressor(n_estimators=120,max_depth=config['depth'],min_samples_leaf=3,n_jobs=4,random_state=20261123)
    model.fit(x.numpy(),y.numpy())
    return model


def predict(data,inputs,ids,model,config):
    rankings=[]
    for q in range(len(data.texts)):
        x=torch.stack([inputs[f'{q}:{i}'] for i in ids])
        local=ridge(model,x) if config['family']=='ridge' else torch.from_numpy(model.predict(x.numpy())).float()
        base=torch.tensor([data.retrieval[f'{q}:{i}']['base_score'] for i in ids])
        scores=base+config['amount']*local
        rankings.append([ids[j] for j in scores.argsort(descending=True,stable=True)[:5].tolist()])
    return rankings


def select(data):
    inputs=features(data);trials=[];best=None
    configs=[{'family':'ridge','alpha':alpha} for alpha in (.01,.1,1.,10.,100.)]
    configs+=[{'family':'extra','depth':depth} for depth in (2,4)]
    for config in configs:
        models=[]
        for val in data.folds():
            train=[i for i in data.fit_ids if i not in val]
            models.append(fit(data,inputs,train,config))
        for amount in (.03,.1,.3,1.,3.):
            trial={**config,'amount':amount}
            metrics=[retrieval_metric(data,val,predict(data,inputs,val,model,trial)) for val,model in zip(data.folds(),models)]
            score=sum(r['weighted_retrieval']*r['queries'] for r in metrics)/sum(r['queries'] for r in metrics)
            trial['cv_retrieval']=score;trials.append(trial)
            if best is None or score>best['cv_retrieval']+1e-9:best=trial
        print(f'Attribute retrieval {config}: selected fit-side weighted recall {best["cv_retrieval"]:.4f}',flush=True)
    return inputs,fit(data,inputs,data.fit_ids,best),best,trials
