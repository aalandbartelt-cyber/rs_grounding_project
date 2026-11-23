"""Nonlinear candidate quality models with fit-side group cross-validation."""
import itertools
import json
from datetime import datetime,timezone

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.decomposition import PCA

from data import BASE,Data,iou
from common import digest,write_json
from train import chosen_boxes,grounding_metric,ridge
from evaluate import predict
from score_round import measure

OUT=BASE/"training_v1"


def extra_features(data):
    semantic=json.loads((OUT/"features/qwen_semantic.json").read_text(encoding="utf-8"))
    result={}
    for key,r in data.grounding.items():
        s=r["scalar"]
        x,y=s[:,5],s[:,6]
        w,h=s[:,7].clamp_min(.001),s[:,8].clamp_min(.001)
        score=s[:,4]
        ranks=score.argsort(descending=True).argsort().float()/max(1,len(score)-1)
        highest=int(score.argmax())
        overlaps=iou(r["boxes"][highest],r["boxes"])
        raw=torch.tensor(semantic["grounding"][key])/5
        text=data.texts[r["query"]]
        text_flags=torch.tensor([float('四' in text),float('两' in text or '双发' in text),float('上单翼' in text),
                                 float('翼上' in text),float('大型' in text),float('小型' in text)])
        geometry=torch.stack([x*x,y*y,(x-.5).abs(),(y-.5).abs(),w.log(),h.log(),
                              (w*h).log(),(w/h).log(),ranks,overlaps,raw,raw-raw.max(),
                              score-score.max(),torch.full_like(score,len(score))],1)
        centered=(s-s.mean(0))/s.std(0,unbiased=False).clamp_min(.01)
        result[key]=torch.cat([s,geometry,centered,text_flags.expand(len(s),-1)],1).numpy()
    return result


def get_model(family,depth,leaf):
    if family=='extra':
        return ExtraTreesRegressor(n_estimators=160,max_depth=depth,min_samples_leaf=leaf,
                                    max_features=.8,n_jobs=4,random_state=20261123)
    return GradientBoostingRegressor(n_estimators=120,max_depth=depth,min_samples_leaf=leaf,
        learning_rate=.04,loss='huber',random_state=20261123)


def targets(data,ids,mode):
    result={}
    for key in data.keys(ids):
        r=data.grounding[key]
        truth=data.truths(r['image'],r['query'],ids)
        overlaps=torch.stack([iou(b,truth) for b in r['boxes']])
        result[key]=(overlaps.mean(1) if mode=='iou' else
                      (.6*(overlaps>=.5).float()+.1*(overlaps>=.75).float()).mean(1)).numpy()
    return result


def main():
    torch.set_num_threads(4)
    data=Data();folds=data.folds();features=extra_features(data)
    old=torch.load(OUT/'weights/trained_fusion_final.pt',weights_only=True)
    old_summary=json.loads((OUT/'results/final_training_summary.json').read_text(encoding='utf-8'))
    old_config=old['config']['grounding']
    best={'family':'old','cv_grounding':old_config['cv_grounding'],'scale':old_config['scale'],'fraction':0.}
    base={key:r['base_score'] for key,r in data.grounding.items()}
    trials=[]
    for family,depth,leaf,mode in itertools.product(('extra','boost'),(2,4),(2,5),('iou','utility')):
        scores={}
        for val in folds:
            fit=[i for i in data.fit_ids if i not in val]
            keys=data.keys(fit); ys=targets(data,fit,mode)
            model=get_model(family,depth,leaf)
            model.fit(np.concatenate([features[k] for k in keys]),np.concatenate([ys[k] for k in keys]))
            for key in data.keys(val):
                scores[key]=torch.from_numpy(model.predict(features[key])).float()
        for fraction,scale in itertools.product((.1,.25,.5,1.,2.,5.),(.8,.9,.95,1.,1.05,1.15)):
            combined={k:base[k]+fraction*v for k,v in scores.items()}
            score=grounding_metric(data,data.fit_ids,chosen_boxes(data,data.fit_ids,combined,scale))['weighted_grounding']
            config={'family':family,'depth':depth,'leaf':leaf,'target':mode,'fraction':fraction,'scale':scale,'cv_grounding':score}
            trials.append(config)
            if score>best['cv_grounding']+1e-9:
                best=config
        print(f"Tree CV {family} depth={depth} leaf={leaf} target={mode}: selected grounding {best['cv_grounding']:.4f}",flush=True)
    frozen={'grounding':best,'selection_used':'three grouped fit folds; no dev label used for these parameters',
            'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'previous_internal_dev_score_seen':.5,
            'limitation':'Internal-dev error analysis motivated this model family; result is iterative development.'}
    write_json(OUT/'results/tree_cv_trials.json',trials)
    write_json(OUT/'results/tree_frozen_config.json',frozen)
    tree=None
    if best['family']!='old':
        keys=data.keys(data.fit_ids);ys=targets(data,data.fit_ids,best['target'])
        tree=get_model(best['family'],best['depth'],best['leaf'])
        tree.fit(np.concatenate([features[k] for k in keys]),np.concatenate([ys[k] for k in keys]))
    checkpoint=OUT/'weights/trained_candidate_ensemble.joblib'
    joblib.dump({'model':tree,'config':frozen},checkpoint)
    write_json(OUT/'results/tree_training_summary.json',{
        'fit_cross_validation_score_out_of_1':best['cv_grounding']+old['config']['retrieval']['cv_retrieval'],
        'checkpoint_sha256':digest(checkpoint),'config':frozen,
        'training_images':data.split['fit'],'official_validation_or_test_used':False})
    if tree is None:
        predictions=predict(data,data.dev_ids,old)
    else:
        scores={k:base[k]+best['fraction']*torch.from_numpy(tree.predict(features[k])).float() for k in data.keys(data.dev_ids)}
        predictions=predict(data,data.dev_ids,old)
        predictions['grounding']=chosen_boxes(data,data.dev_ids,scores,best['scale'])
    result,details=measure(data.cache,data.split['dev'],predictions)
    write_json(OUT/'results/tree_internal_dev_score.json',{
        **result,'target_score':.6,'target_met':result['score_out_of_1']>=.6,
        'official_validation_or_test_used':False,'training_images':33,'internal_dev_images':16,
        'frozen_at_utc':frozen['frozen_at_utc'],'scored_at_utc':datetime.now(timezone.utc).isoformat(),
        'checkpoint_sha256':digest(checkpoint),
        'fit_cross_validation_score_out_of_1':best['cv_grounding']+old['config']['retrieval']['cv_retrieval']})
    write_json(OUT/'results/tree_internal_dev_predictions.json',details)
    print('Tree internal dev:',result,flush=True)


if __name__=='__main__':
    main()
