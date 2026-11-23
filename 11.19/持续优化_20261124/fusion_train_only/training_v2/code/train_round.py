"""Train shared candidate selection on fit-only folds, then score fixed internal dev."""
import itertools
import json
from datetime import datetime,timezone

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from inputs import BASE,OUT,write_json,digest,iou
from fusion_features import Features,box_predictions
from train import fit_ridge,ridge,grounding_metric
from evaluate import predict
from score_round import measure
import retrieval


def targets(features,ids,mode):
    data=features.data
    assert set(ids)<=set(data.fit_ids),'Supervision outside original fit split'
    result={}
    for key in features.keys(ids):
        row=features.rows[key]
        truth=data.truths(row['image'],row['query'],ids)
        overlap=torch.stack([iou(box,truth) for box in row['boxes']])
        result[key]=(overlap.mean(1) if mode=='iou' else
                     (.6*(overlap>=.5).float()+.1*(overlap>=.75).float()).mean(1))
    return result


def train_model(features,ids,config):
    keys=features.keys(ids);ys=targets(features,ids,config['target'])
    x=torch.cat([features.features(k,config['kind']) for k in keys])
    y=torch.cat([ys[k] for k in keys])
    if config['family']=='ridge':return fit_ridge(x,y,config['regularization'])
    model=ExtraTreesRegressor(n_estimators=160,max_depth=config['depth'],min_samples_leaf=3,
                              max_features=.9,n_jobs=4,random_state=20261123)
    model.fit(x.numpy(),y.numpy())
    return model


def model_score(model,x,family):
    return ridge(model,x) if family=='ridge' else torch.from_numpy(model.predict(x.numpy())).float()


def scores(features,ids,model,config):
    result={}
    for key in features.keys(ids):
        value=features.rows[key]['base']
        if model is not None:
            value=value+config['amount']*model_score(model,features.features(key,config['kind']),config['family'])
        result[key]=value
    return result


def main():
    torch.set_num_threads(4)
    features=Features();data=features.data
    protocol={'original_fit_images':data.split['fit'],'original_internal_dev_images':data.split['dev'],
              'hyperparameters_selected_using':'three scene-group folds within 33 fit images only',
              'official_validation_or_test_used':False,'target_score':.6,
              'feature_names':features.names,'frozen_before_training_utc':datetime.now(timezone.utc).isoformat(),
              'known_limitation':'Development has already consulted the fixed internal-dev scores in earlier rounds.'}
    write_json(OUT/'results/protocol.json',protocol)
    base={k:r['base'] for k,r in features.rows.items()}
    base_metric=grounding_metric(data,data.fit_ids,box_predictions(features,data.fit_ids,base))
    best={'family':'base','kind':'all','target':'utility','amount':0.,'scale':1.,'cv_grounding':base_metric['weighted_grounding']}
    trials=[]
    models=[]
    for kind,mode,alpha in itertools.product(('all','semantic'),('iou','utility'),(.01,.1,1.,10.)):
        models.append({'family':'ridge','kind':kind,'target':mode,'regularization':alpha})
    for kind,mode,depth in itertools.product(('all','semantic'),('iou','utility'),(2,4)):
        models.append({'family':'extra','kind':kind,'target':mode,'depth':depth})
    for number,config in enumerate(models,1):
        oof={}
        for val in data.folds():
            train_ids=[i for i in data.fit_ids if i not in val]
            model=train_model(features,train_ids,config)
            for key in features.keys(val):
                oof[key]=model_score(model,features.features(key,config['kind']),config['family'])
        for amount,scale in itertools.product((.1,.25,.5,1.,2.),(.9,.95,1.,1.05)):
            combined={k:base[k]+amount*v for k,v in oof.items()}
            value=grounding_metric(data,data.fit_ids,box_predictions(features,data.fit_ids,combined,scale))['weighted_grounding']
            trial={**config,'amount':amount,'scale':scale,'cv_grounding':value};trials.append(trial)
            if value>best['cv_grounding']+1e-9:best=trial
        if number%5==0 or number==len(models):
            print(f'Multi-view candidate training {number}/{len(models)}: best fit-side grounding {best["cv_grounding"]:.4f}',flush=True)
    write_json(OUT/'results/cv_trials.json',trials)
    retrieval_inputs,retrieval_model,retrieval_config,retrieval_trials=retrieval.select(data)
    write_json(OUT/'results/retrieval_cv_trials.json',retrieval_trials)
    frozen={'selected':best,'selection_used':'33 fit images only; held-out scene groups for head selection',
            'retrieval':retrieval_config,
            'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'input_sha256':{p.name:digest(p) for p in (OUT/'features').iterdir() if p.is_file()},
            'code_sha256':{p.name:digest(p) for p in (OUT/'code').glob('*.py')},
            'official_validation_or_test_used':False}
    write_json(OUT/'results/frozen_config.json',frozen)
    model=None if best['family']=='base' else train_model(features,data.fit_ids,best)
    checkpoint=OUT/'weights/multiview_candidate_ranker.joblib';checkpoint.parent.mkdir(exist_ok=True)
    joblib.dump({'model':model,'retrieval_model':retrieval_model,'config':frozen},checkpoint)
    # Retrieval remains the trained head used in the previous supervised round.
    previous=torch.load(BASE/'training_v1/weights/trained_fusion_final.pt',weights_only=True)
    predictions=predict(data,data.dev_ids,previous)
    original=predictions['grounding']
    improved=box_predictions(features,data.dev_ids,scores(features,data.dev_ids,model,best),best['scale'])
    predictions['grounding']=improved
    localization_result,localization_details=measure(data.cache,data.split['dev'],predictions)
    new_rankings=retrieval.predict(data,retrieval_inputs,data.dev_ids,retrieval_model,retrieval_config)
    predictions['rankings']=new_rankings
    result,details=measure(data.cache,data.split['dev'],predictions)
    predictions['grounding']=original
    retrieval_result,_=measure(data.cache,data.split['dev'],predictions)
    write_json(OUT/'results/ablations.json',{'new_localization_only':localization_result,
        'new_retrieval_only':retrieval_result,'combined':result})
    write_json(OUT/'results/localization_only_predictions.json',localization_details)
    report={**result,'target_score':.6,'target_met':result['score_out_of_1']>=.6,
            'official_validation_or_test_used':False,'fit_images':33,'internal_dev_images':16,
            'checkpoint_sha256':digest(checkpoint),'frozen_config_sha256':digest(OUT/'results/frozen_config.json'),
            'scored_at_utc':datetime.now(timezone.utc).isoformat(),'selected':best,
            'previous_best_same_internal_dev_score':.5037037037037037}
    write_json(OUT/'results/internal_dev_score.json',report)
    write_json(OUT/'results/internal_dev_predictions.json',details)
    print('Multi-view internal development score:',result,flush=True)


if __name__=='__main__':main()
