"""Supervised LoRA of the final RemoteCLIP vision blocks on civilian train only."""
import gc
import json
import random
from datetime import datetime,timezone

import open_clip
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from PIL import ImageEnhance,ImageOps

from data import BASE,Data,iou
from common import TrainArchive,crop,digest,write_json
from train import chosen_boxes,grounding_metric
from evaluate import predict
from score_round import measure

OUT=BASE/'training_v1'


class LowRank(nn.Module):
    def __init__(self,shape,rank=8):
        super().__init__()
        self.a=nn.Parameter(torch.randn(rank,shape[1])*.01)
        self.b=nn.Parameter(torch.zeros(shape[0],rank))
    def forward(self,weight):
        return weight+self.b@self.a


def load_model():
    model,_,transform=open_clip.create_model_and_transforms('ViT-L-14',pretrained=None)
    path=BASE/'weights/RemoteCLIP/RemoteCLIP-ViT-L-14.pt'
    state=torch.load(path,weights_only=True,mmap=True,map_location='cpu')
    state=state.get('state_dict',state.get('model',state))
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    del state
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for block in model.visual.transformer.resblocks[-2:]:
        for module,name in ((block.attn,'in_proj_weight'),(block.attn.out_proj,'weight'),
                            (block.mlp.c_fc,'weight'),(block.mlp.c_proj,'weight')):
            parametrize.register_parametrization(module,name,LowRank(getattr(module,name).shape))
    return model.cuda(),transform


def sample_specs(data,ids):
    assert set(ids)<=set(data.fit_ids)
    samples=[]
    for image in ids:
        row=data.records[image]
        for annotation,text in enumerate(row['texts']):
            samples.append({'image':image,'query':data.texts.index(text),'box':row['truth'][annotation],
                            'label':1.,'weight':1.})
        queries=sorted(set(data.texts.index(t) for t in row['texts']))
        for query in queries:
            truth=data.truths(image,query,ids)
            for box in row['boxes']:
                overlap=float(iou(box,truth).max())
                if overlap>=.5 or overlap<.1:
                    samples.append({'image':image,'query':query,'box':box,'label':float(overlap>=.5),'weight':.5})
    return samples


def adapter_state(model):
    return {name:param.detach().cpu().clone() for name,param in model.named_parameters() if param.requires_grad}


def restore(model,state):
    with torch.no_grad():
        for name,parameter in model.named_parameters():
            if name in state: parameter.copy_(state[name].to(parameter.device))


def train_epochs(data,ids,epochs,validation=None):
    torch.manual_seed(20261123);rng=random.Random(20261123)
    model,transform=load_model();archive=TrainArchive()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda')
    specs=sample_specs(data,ids);images={i:archive.image(data.records[i]['label']) for i in ids}
    text=data.text_features['RemoteCLIP'].cuda()
    history=[];best_score=-1.;selected_epoch=epochs
    for epoch in range(1,epochs+1):
        model.train();rng.shuffle(specs);total=0.;count=0
        for start in range(0,len(specs),4):
            rows=specs[start:start+4];views=[]
            for spec in rows:
                view=crop(images[spec['image']],spec['box'],rng.choice((1.,1.25,1.5)))
                if rng.random()<.5: view=ImageOps.mirror(view)
                view=view.rotate(rng.choice((0,90,180,270)),expand=True)
                view=ImageEnhance.Brightness(view).enhance(rng.uniform(.9,1.1))
                views.append(transform(view))
            batch=torch.stack(views).cuda()
            labels=torch.tensor([r['label'] for r in rows],device='cuda')
            weights=torch.tensor([r['weight'] for r in rows],device='cuda')
            queries=torch.tensor([r['query'] for r in rows],device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                visual=F.normalize(model.encode_image(batch).float(),dim=-1)
                similarity=(visual*text[queries]).sum(1)
                logits=20*(similarity-.25)
                loss=(F.binary_cross_entropy_with_logits(logits,labels,reduction='none')*weights).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            scaler.step(optimizer);scaler.update()
            total+=float(loss.detach())*len(rows);count+=len(rows)
        record={'epoch':epoch,'mean_loss':total/count}
        if validation is not None:
            local=encode_grounding(model,transform,data,archive,validation)
            scores=combined_scores(data,validation,local)
            metric=grounding_metric(data,validation,chosen_boxes(data,validation,scores,.95))['weighted_grounding']
            record['inner_validation_weighted_grounding']=metric
            if metric>best_score+1e-9:
                best_score=metric;selected_epoch=epoch
        history.append(record)
        print('RemoteCLIP visual LoRA',record,flush=True)
    state=adapter_state(model)
    archive.audit(OUT/'results/visual_lora_training_data_access.json')
    archive.close()
    return model,transform,state,history,selected_epoch


@torch.inference_mode()
def encode_grounding(model,transform,data,archive,ids):
    model.eval();vectors={}
    for key in data.keys(ids):
        row=data.grounding[key];image=archive.image(data.records[row['image']]['label'])
        views=[transform(crop(image,box,scale)) for box in row['boxes'] for scale in (1.,1.5)]
        encoded=[]
        for start in range(0,len(views),4):
            with torch.autocast('cuda',dtype=torch.float16):
                value=F.normalize(model.encode_image(torch.stack(views[start:start+4]).cuda()).float(),dim=-1)
            encoded.append(value.cpu())
        vectors[key]=torch.cat(encoded).reshape(len(row['boxes']),2,-1).mean(1)
    return vectors


def combined_scores(data,ids,visual):
    scores={}
    for key in data.keys(ids):
        row=data.grounding[key];s=row['scalar'];q=row['query']
        similarity=visual[key]@data.text_features['RemoteCLIP'][q]
        scores[key]=.5*similarity+.5*s[:,1]+.1*s[:,4]+.05*s[:,12]+.05*s[:,13]
    return scores


def main():
    torch.set_num_threads(4)
    data=Data();validation=data.folds()[0];training=[i for i in data.fit_ids if i not in validation]
    model,transform,state,history,epoch=train_epochs(data,training,6,validation)
    write_json(OUT/'results/visual_lora_inner_selection.json',{
        'selection_used':'one scene-group inner fold of original 33 fit images',
        'selected_epoch':epoch,'history':history,'training_images':training,'validation_images':validation})
    del model;gc.collect();torch.cuda.empty_cache()
    config={'rank':8,'vision_blocks':2,'learning_rate':1e-4,'selected_epoch':epoch,
            'training_images':data.split['fit'],'original_internal_dev_used_for_epoch_selection':False,
            'frozen_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(OUT/'results/visual_lora_frozen_config.json',config)
    model,transform,state,history,_=train_epochs(data,data.fit_ids,epoch)
    checkpoint=OUT/'weights/RemoteCLIP_visual_lora.pt'
    torch.save({'state':state,'config':config},checkpoint)
    write_json(OUT/'results/visual_lora_training_history.json',history)
    archive=TrainArchive();visual=encode_grounding(model,transform,data,archive,data.dev_ids)
    archive.audit(OUT/'results/visual_lora_inference_data_access.json');archive.close()
    torch.save(visual,OUT/'features/RemoteCLIP_lora_dev_regions.pt')
    old=torch.load(OUT/'weights/trained_fusion_final.pt',weights_only=True)
    predictions=predict(data,data.dev_ids,old)
    predictions['grounding']=chosen_boxes(data,data.dev_ids,combined_scores(data,data.dev_ids,visual),.95)
    result,details=measure(data.cache,data.split['dev'],predictions)
    write_json(OUT/'results/visual_lora_internal_dev_score.json',{
        **result,'target_score':.6,'target_met':result['score_out_of_1']>=.6,'official_validation_or_test_used':False,
        'checkpoint_sha256':digest(checkpoint),'config':config,
        'trained_parameters':sum(t.numel() for t in state.values()),'scored_at_utc':datetime.now(timezone.utc).isoformat()})
    write_json(OUT/'results/visual_lora_internal_dev_predictions.json',details)
    print('RemoteCLIP visual LoRA internal dev',result,flush=True)


if __name__=='__main__':main()
