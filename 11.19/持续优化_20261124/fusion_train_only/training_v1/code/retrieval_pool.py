"""Train full-candidate retrieval aggregation without using grounding assignments."""
import itertools
import json
from datetime import datetime,timezone

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor

from data import BASE,Data
from common import write_json,digest
from train import fit_ridge,ridge,retrieval_metric
from evaluate import predict
from score_round import measure

OUT=BASE/'training_v1'


def pool_features(data):
    result={}
    for q in range(len(data.texts)):
        for i,row in enumerate(data.records):
            parts=[data.retrieval[f'{q}:{i}']['scalar']]
            n=len(row['boxes'])
            for name in data.models:
                v=data.visual[name];start=v['records'][i]['start'];text=data.text_features[name][q]
                whole=float(v['visual'][start]@text)
                for offset in (1,1+n):
                    region=v['visual'][start+offset:start+offset+n]@text
                    if n:
                        ordered=region.sort(descending=True).values
                        weighted=region+.1*row['scores'].clamp_min(1e-6).log()
                        summaries=[region.max(),region.mean(),region.std(unbiased=False),
                                   ordered[:2].mean(),ordered[:3].mean(),weighted.max(),
                                   torch.logsumexp(region*20,0)/20,np.float32(whole)]
                    else: summaries=[whole]*8
                    parts.append(torch.tensor(summaries))
            if n:
                boxes=row['boxes']/torch.tensor([row['width'],row['height'],row['width'],row['height']])
                area=(boxes[:,2:]-boxes[:,:2]).prod(1)
                detection=torch.tensor([n,row['scores'].max(),row['scores'].mean(),area.max(),area.mean()])
            else: detection=torch.zeros(5)
            parts.append(detection)
            result[f'{q}:{i}']=torch.cat(parts).float()
    return result


def fit(data,features,ids,family,alpha):
    assert set(ids)<=set(data.fit_ids)
    queries=sorted(set(data.texts.index(t) for i in ids for t in data.records[i]['texts']))
    x=torch.stack([features[f'{q}:{i}'] for q in queries for i in ids])
    y=torch.tensor([data.records[i]['texts'].count(data.texts[q]) for q in queries for i in ids],dtype=torch.float32)
    if family=='ridge': return fit_ridge(x,y,alpha)
    if family=='extra': model=ExtraTreesRegressor(n_estimators=120,max_depth=int(alpha),min_samples_leaf=3,max_features=.8,n_jobs=4,random_state=20261123)
    else: model=GradientBoostingRegressor(n_estimators=100,max_depth=int(alpha),learning_rate=.03,min_samples_leaf=3,random_state=20261123)
    model.fit(x.numpy(),y.numpy())
    return model


def rankings(data,features,ids,model,family,fraction):
    result=[]
    for q in range(len(data.texts)):
        x=torch.stack([features[f'{q}:{i}'] for i in ids])
        score=ridge(model,x) if family=='ridge' else torch.from_numpy(model.predict(x.numpy())).float()
        base=torch.tensor([data.retrieval[f'{q}:{i}']['base_score'] for i in ids])
        score=base+fraction*score
        result.append([ids[j] for j in score.argsort(descending=True,stable=True)[:5].tolist()])
    return result


def main():
    torch.set_num_threads(4)
    data=Data();features=pool_features(data);old=torch.load(OUT/'weights/trained_fusion_final.pt',weights_only=True)
    best={'family':'old','cv_retrieval':old['config']['retrieval']['cv_retrieval']};trials=[]
    for family,values in (('ridge',(.01,.1,1.,10.,100.)),('extra',(2,4,6)),('boost',(2,3))):
        for alpha in values:
            models=[]
            for val in data.folds():
                ids=[i for i in data.fit_ids if i not in val]
                models.append(fit(data,features,ids,family,alpha))
            for fraction in (.03,.1,.3,1.,3.):
                metrics=[retrieval_metric(data,val,rankings(data,features,val,model,family,fraction)) for val,model in zip(data.folds(),models)]
                score=sum(r['weighted_retrieval']*r['queries'] for r in metrics)/sum(r['queries'] for r in metrics)
                trial={'family':family,'alpha':alpha,'fraction':fraction,'cv_retrieval':score};trials.append(trial)
                if score>best['cv_retrieval']+1e-9: best=trial
            print(f'Complete-candidate retrieval CV {family} {alpha}: best {best["cv_retrieval"]:.4f}',flush=True)
    config={'retrieval':best,'selection_used':'three grouped folds within 33 fit images only',
            'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'grounding_assignment_used_for_retrieval':False,'official_validation_or_test_used':False}
    write_json(OUT/'results/retrieval_pool_cv_trials.json',trials)
    write_json(OUT/'results/retrieval_pool_frozen_config.json',config)
    model=None if best['family']=='old' else fit(data,features,data.fit_ids,best['family'],best['alpha'])
    checkpoint=OUT/'weights/trained_retrieval_aggregation.joblib'
    joblib.dump({'model':model,'config':config},checkpoint)
    predictions=predict(data,data.dev_ids,old)
    if model is not None: predictions['rankings']=rankings(data,features,data.dev_ids,model,best['family'],best['fraction'])
    result,details=measure(data.cache,data.split['dev'],predictions)
    write_json(OUT/'results/retrieval_pool_internal_dev_score.json',{
        **result,'target_score':.6,'target_met':result['score_out_of_1']>=.6,
        'fit_cross_validation_score_out_of_1':best['cv_retrieval']+old['config']['grounding']['cv_grounding'],
        'official_validation_or_test_used':False,'checkpoint_sha256':digest(checkpoint),
        'frozen_config':config,'scored_at_utc':datetime.now(timezone.utc).isoformat()})
    write_json(OUT/'results/retrieval_pool_internal_dev_predictions.json',details)
    print('Complete-candidate aggregation internal dev:',result,flush=True)


if __name__=='__main__':main()
