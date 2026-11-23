"""Three scene-group folds over the 49 original-training civilian images."""
import json
from datetime import datetime,timezone

import torch

from data import BASE,Data
from common import digest,write_json
from train import fit_ridge,supervision,target,retrieval_supervision
from evaluate import predict
from score_round import measure

OUT=BASE/'training_v1'


def scene_folds(data):
    lookup={r['label']:i for i,r in enumerate(data.records)}
    groups=[[lookup[label] for label in group] for group in data.split['overlap_groups']]
    groups.sort(key=lambda g:-(sum(len(data.records[i]['texts']) for i in g)+len(g)))
    folds=[[],[],[]]
    for group in groups:
        index=min(range(3),key=lambda j:sum(len(data.records[i]['texts']) for i in folds[j])+len(folds[j]))
        folds[index].extend(group)
    return folds


def train_bundle(data,ids,config):
    x,y=supervision(data,ids,config['grounding']['kind'])
    ground=fit_ridge(x,target(y,config['grounding']['target']),config['grounding']['alpha'])
    x,y=retrieval_supervision(data,ids,config['retrieval']['kind'])
    retrieval=fit_ridge(x,y,config['retrieval']['alpha'])
    return {'config':config,'grounding':ground,'retrieval':retrieval}


def main():
    torch.set_num_threads(4)
    data=Data()
    previous=torch.load(OUT/'weights/trained_fusion_final.pt',weights_only=True)
    config=previous['config']
    assert config['grounding']['family']==config['retrieval']['family']=='ridge'
    evidence=json.loads((OUT/'features/qwen_all_training_gallery.json').read_text(encoding='utf-8'))
    assert len(evidence)==len(data.records)*len(data.texts)
    for key,row in data.retrieval.items():
        value=float(torch.tensor(evidence[key]/5).tanh())
        for kind in ('scalar','vision','conditioned'): row[kind][-1]=value
    folds=scene_folds(data)
    # Every source image remains original train. The former internal dev is now rotated through folds.
    data.fit_ids=list(range(len(data.records)))
    protocol={'source':'49 reviewed original public-training images only','fold_count':3,
              'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
              'folds':[[data.records[i]['label'] for i in f] for f in folds],
              'scene_groups_do_not_cross_folds':True,'config_fixed_from_previous_training':config,
              'official_validation_or_test_used':False,
              'old_16_image_dev_rotated_into_training_folds':True,
              'interpretation':'Exploratory grouped cross-validation, not an independent final test or the fixed 16-image dev score.'}
    write_json(OUT/'results/all49_cv_protocol.json',protocol)
    totals={'R@1':0,'R@5':0,'Acc@0.5':0,'Acc@0.75':0};queries=0;results=[];all_details=[]
    for number,validation in enumerate(folds,1):
        training=[i for i in data.fit_ids if i not in validation]
        assert not set(training)&set(validation)
        bundle=train_bundle(data,training,config)
        bundle['training_labels']=[data.records[i]['label'] for i in training]
        checkpoint=OUT/f'weights/all49_cv_fold_{number}.pt';torch.save(bundle,checkpoint)
        predictions=predict(data,validation,bundle)
        labels=[data.records[i]['label'] for i in validation]
        result,details=measure(data.cache,labels,predictions)
        for name in totals: totals[name]+=result['counts'][name]
        queries+=result['queries_per_task']
        results.append({'fold':number,'training_images':len(training),'validation_images':len(validation),
                        'checkpoint_sha256':digest(checkpoint),**result})
        all_details.extend([{'fold':number,**row} for row in details])
        print(f"Scene fold {number}: {result['score_out_of_1']:.4f}/1, gallery={len(validation)}, queries={result['queries_per_task']}",flush=True)
    score=(totals['R@1']+2*totals['R@5']+6*totals['Acc@0.5']+totals['Acc@0.75'])/(10*queries)
    report={'scope':'grouped three-fold cross-validation on all 49 original-training images',
        'score_out_of_1':score,'counts':totals,'queries_per_task':queries,
        **{name:value/queries for name,value in totals.items()},
        'folds':results,'target_score':.6,'target_met':score>=.6,
        'old_fixed_16_image_dev_best_score':.5,
        'official_validation_or_test_used':False,'every_image_scored_out_of_fold':True,
        'score_is_not_comparable_to_old_gallery':True,'protocol_sha256':digest(OUT/'results/all49_cv_protocol.json'),
        'scored_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(OUT/'results/all49_grouped_cv_score.json',report)
    write_json(OUT/'results/all49_grouped_cv_predictions.json',all_details)
    # Final training uses the full original-training subset only after OOF predictions are saved.
    full=train_bundle(data,data.fit_ids,config)
    full['training_labels']=[r['label'] for r in data.records]
    full['evaluation']='Score comes from separate out-of-fold models, never this all-data model on its own training images.'
    torch.save(full,OUT/'weights/all49_trained_fusion.pt')
    print(f'All 49 original-training grouped CV: {score:.4f}/1 {totals}',flush=True)


if __name__=='__main__':main()
