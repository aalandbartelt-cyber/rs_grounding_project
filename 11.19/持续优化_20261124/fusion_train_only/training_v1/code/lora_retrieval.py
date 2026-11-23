"""Integrate the trained visual adapter into full-gallery image retrieval."""
import json
from datetime import datetime,timezone

import torch
from torch.nn.functional import normalize

from data import BASE,Data
from common import TrainArchive,crop,digest,write_json
from lora_visual import load_model,restore
from train import fit_ridge,retrieval_predictions,retrieval_supervision,chosen_boxes
from lora_visual import combined_scores
from evaluate import predict
from score_round import measure

OUT=BASE/'training_v1'


@torch.inference_mode()
def encode(model,transform,data):
    archive=TrainArchive();output={}
    for i,row in enumerate(data.records):
        image=archive.image(row['label'])
        views=[transform(image)]
        for box in row['boxes']:
            views.extend(transform(crop(image,box,s)) for s in (1.,1.5))
        values=[]
        for start in range(0,len(views),4):
            with torch.autocast('cuda',dtype=torch.float16):
                value=normalize(model.encode_image(torch.stack(views[start:start+4]).cuda()).float(),dim=-1)
            values.append(value.cpu())
        values=torch.cat(values)
        output[i]={'whole':values[0], 'regions':values[1:].reshape(len(row['boxes']),2,-1).mean(1)}
    archive.audit(OUT/'results/visual_lora_retrieval_features_data_access.json');archive.close()
    return output


def main():
    torch.set_num_threads(4)
    data=Data();lora=torch.load(OUT/'weights/RemoteCLIP_visual_lora.pt',weights_only=True)
    model,transform=load_model();restore(model,lora['state']);model.eval()
    visual=encode(model,transform,data)
    torch.save(visual,OUT/'features/RemoteCLIP_lora_retrieval.pt')
    # The retrieval-head form and regularization stay fixed from fit-side CV.
    for q in range(len(data.texts)):
        txt=data.text_features['RemoteCLIP'][q]
        for i,row in enumerate(data.records):
            remote=visual[i];geo=data.images[i]['regions'][1];geo_text=data.text_features['GeoRSCLIP'][q]
            global_sim=torch.tensor([float(remote['whole']@txt),float(data.images[i]['whole'][1]@geo_text)])
            if len(row['boxes']):
                local=torch.stack([remote['regions']@txt,geo@geo_text],1)
                values=local.mean(1)+.1*row['scores'].clamp_min(1e-6).log()
                best=int(values.argmax());region_sim=local[best];confidence=row['scores'][best].clamp_min(1e-6).log()
            else: region_sim=global_sim;confidence=torch.tensor(0.)
            original=data.retrieval[f'{q}:{i}']['scalar']
            scalar=torch.cat([global_sim,region_sim,confidence[None],original[-1:]])
            data.retrieval[f'{q}:{i}']['scalar']=scalar
            data.retrieval[f'{q}:{i}']['base_score']=float(.5*global_sim.mean()+.5*(region_sim.mean()+.1*confidence))
    old=torch.load(OUT/'weights/trained_fusion_final.pt',weights_only=True)
    config=old['config']['retrieval']
    x,y=retrieval_supervision(data,data.fit_ids,'scalar')
    head=fit_ridge(x,y,config['alpha'])
    frozen={'selection_used':'retrieval head settings fixed from prior fit-only CV; visual adapter fitted on 33 fit images',
            'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'retrieval_config':config,
            'visual_adapter_sha256':digest(OUT/'weights/RemoteCLIP_visual_lora.pt'),
            'official_validation_or_test_used':False}
    write_json(OUT/'results/visual_lora_full_frozen_config.json',frozen)
    checkpoint=OUT/'weights/RemoteCLIP_lora_retrieval_head.pt';torch.save({'head':head,'config':frozen},checkpoint)
    predictions=predict(data,data.dev_ids,old)
    predictions['rankings']=retrieval_predictions(data,data.dev_ids,head,'scalar',config['fraction'])
    local=torch.load(OUT/'features/RemoteCLIP_lora_dev_regions.pt',weights_only=True)
    predictions['grounding']=chosen_boxes(data,data.dev_ids,combined_scores(data,data.dev_ids,local),.95)
    result,details=measure(data.cache,data.split['dev'],predictions)
    write_json(OUT/'results/visual_lora_full_internal_dev_score.json',{
        **result,'target_score':.6,'target_met':result['score_out_of_1']>=.6,'official_validation_or_test_used':False,
        'checkpoint_sha256':digest(checkpoint),'config':frozen,'scored_at_utc':datetime.now(timezone.utc).isoformat()})
    write_json(OUT/'results/visual_lora_full_internal_dev_predictions.json',details)
    print('Full integration of visual LoRA',result,flush=True)


if __name__=='__main__':main()
