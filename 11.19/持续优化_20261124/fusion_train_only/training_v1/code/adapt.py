"""Fit regularized visual adapters using only fit-side augmented target crops."""
import itertools
import json
from datetime import datetime,timezone

import torch

from data import BASE,Data,iou
from common import digest,write_json
from train import chosen_boxes,grounding_metric,ridge
from evaluate import predict
from score_round import measure

OUT=BASE/'training_v1'


def augmented_features(data,aug,ids,kind):
    selected=[j for j,r in enumerate(aug['jobs']) if r['image'] in ids]
    assert set(ids)<=set(data.fit_ids)
    visual=torch.cat([aug[name][selected] for name in data.models],1)
    if kind=='visual': return visual
    queries=[aug['jobs'][j]['query'] for j in selected]
    text=torch.cat([data.text_features[name][queries] for name in data.models],1)
    return torch.cat([visual,visual*text],1)


def model_fit(data,aug,ids,kind,alpha,weight):
    xs,ys=[],[]
    for key in data.keys(ids):
        row=data.grounding[key]
        truth=data.truths(row['image'],row['query'],ids)
        overlap=torch.stack([iou(box,truth) for box in row['boxes']]).max(1).values
        xs.append(row['conditioned'][:,:1536 if kind=='visual' else 3072])
        ys.append(overlap)
    x,y=torch.cat(xs),torch.cat(ys)
    positive=augmented_features(data,aug,ids,kind)
    all_x=torch.cat([x,positive]);all_y=torch.cat([y,torch.ones(len(positive))])
    sample_weights=torch.cat([torch.ones(len(x)),torch.full((len(positive),),weight*len(x)/len(positive))])
    mean,std=all_x.mean(0),all_x.std(0,unbiased=False).clamp_min(.01)
    normalized=(all_x-mean)/std/all_x.shape[1]**.5
    bias=(all_y*sample_weights).sum()/sample_weights.sum()
    weighted=normalized*sample_weights.sqrt()[:,None]
    values=(all_y-bias)*sample_weights.sqrt()
    coefficients=torch.linalg.solve(weighted@weighted.T+alpha*torch.eye(len(all_x)),values)
    return {'mean':mean,'std':std,'weight':weighted.T@coefficients,'bias':bias}


def scores(data,ids,model,kind,amount):
    output={}
    for key in data.keys(ids):
        r=data.grounding[key]
        x=r['conditioned'][:,:1536 if kind=='visual' else 3072]
        output[key]=r['base_score']+amount*ridge(model,x)
    return output


def main():
    torch.set_num_threads(4)
    data=Data();aug=torch.load(OUT/'features/augmented_fit_regions.pt',weights_only=True)
    old=torch.load(OUT/'weights/trained_fusion_final.pt',weights_only=True)
    best={'family':'old','cv_grounding':old['config']['grounding']['cv_grounding']}
    trials=[]
    for kind,alpha,weight in itertools.product(('visual','conditioned'),(.01,.1,1.,10.),(.1,.5,1.)):
        oof={}
        for val in data.folds():
            ids=[i for i in data.fit_ids if i not in val]
            model=model_fit(data,aug,ids,kind,alpha,weight)
            oof.update(scores(data,val,model,kind,1.))
        for amount,scale in itertools.product((.1,.25,.5,1.,2.),(.85,.95,1.,1.05,1.15)):
            mixed={k:data.grounding[k]['base_score']+amount*(v-data.grounding[k]['base_score']) for k,v in oof.items()}
            metric=grounding_metric(data,data.fit_ids,chosen_boxes(data,data.fit_ids,mixed,scale))['weighted_grounding']
            config={'family':'visual_adapter','kind':kind,'alpha':alpha,'augmentation_weight':weight,
                    'amount':amount,'scale':scale,'cv_grounding':metric}
            trials.append(config)
            if metric>best['cv_grounding']+1e-9: best=config
        print(f'Augmented visual CV {kind} alpha={alpha} weight={weight}: best grounding {best["cv_grounding"]:.4f}',flush=True)
    config={'grounding':best,'selection_used':'three grouped folds within 33 fit images; augmented crops exclude each validation group',
            'official_validation_or_test_used':False,'frozen_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(OUT/'results/adapter_cv_trials.json',trials)
    write_json(OUT/'results/adapter_frozen_config.json',config)
    bundle={'config':config}
    if best['family']!='old':
        bundle['model']=model_fit(data,aug,data.fit_ids,best['kind'],best['alpha'],best['augmentation_weight'])
    checkpoint=OUT/'weights/augmented_visual_adapter.pt'
    torch.save(bundle,checkpoint)
    predictions=predict(data,data.dev_ids,old)
    if 'model' in bundle:
        local=scores(data,data.dev_ids,bundle['model'],best['kind'],best['amount'])
        predictions['grounding']=chosen_boxes(data,data.dev_ids,local,best['scale'])
    result,details=measure(data.cache,data.split['dev'],predictions)
    write_json(OUT/'results/adapter_internal_dev_score.json',{
        **result,'target_score':.6,'target_met':result['score_out_of_1']>=.6,
        'fit_cross_validation_score_out_of_1':best['cv_grounding']+old['config']['retrieval']['cv_retrieval'],
        'official_validation_or_test_used':False,'checkpoint_sha256':digest(checkpoint),
        'scored_at_utc':datetime.now(timezone.utc).isoformat(),'frozen_config':config})
    write_json(OUT/'results/adapter_internal_dev_predictions.json',details)
    print('Augmented visual adapter internal dev',result,flush=True)


if __name__=='__main__': main()
