"""Encode actual predicted boxes consistently across frozen and adapted CLIP streams."""
import gc
import json

import open_clip
import torch
from torch.nn.functional import normalize

from inputs import BASE,OUT,Data,TrainArchive,crop,write_json,digest


def candidates(data):
    views=json.loads((OUT/'features/geoground_views.json').read_text(encoding='utf-8'))
    result={}
    for key,row in data.grounding.items():
        boxes=row['boxes'].tolist()
        source=['dino']*(len(boxes)-int(bool(row['scalar'][:,11].max())))
        if len(source)<len(boxes):source.append('geoground_original')
        for rotation in (90,180,270):
            box=views[key][str(rotation)]['mapped_box_xyxy']
            if box is not None:
                boxes.append(box);source.append('geoground_'+str(rotation))
        result[key]={'query':row['query'],'image':row['image'],'boxes':torch.tensor(boxes),
                     'source':source,'original_candidate_count':len(row['boxes'])}
    return result


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    data=Data();regions=candidates(data);archive=TrainArchive()
    path=OUT/'features/candidates.pt'
    signature={'views_sha256':digest(OUT/'features/geoground_views.json'),
               'remote_adapter_sha256':digest(BASE/'training_v1/weights/RemoteCLIP_visual_lora.pt'),
               'views':['tight_actual_predicted_box','context_1.5']}
    if path.exists():
        existing=torch.load(path,weights_only=True)
        if existing['signature']==signature:
            print('Reusing consistent candidate encodings',flush=True);return
    result={'signature':signature,'regions':regions}
    for name in ('RemoteCLIP','RemoteCLIP_LoRA','GeoRSCLIP'):
        if name=='RemoteCLIP_LoRA':
            from lora_visual import load_model,restore
            model,transform=load_model()
            restore(model,torch.load(BASE/'training_v1/weights/RemoteCLIP_visual_lora.pt',weights_only=True)['state'])
        else:
            architecture='ViT-L-14' if name=='RemoteCLIP' else 'ViT-L-14-336'
            checkpoint=(BASE/'weights/RemoteCLIP/RemoteCLIP-ViT-L-14.pt' if name=='RemoteCLIP' else
                        BASE/'weights/GeoRSCLIP/ckpt/RS5M_ViT-L-14-336.pt')
            model,_,transform=open_clip.create_model_and_transforms(architecture,pretrained=None)
            state=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
            state=state.get('state_dict',state.get('model',state))
            model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
            del state
            model=model.cuda()
        model.eval();encoded={}
        for number,(key,row) in enumerate(regions.items(),1):
            image=archive.image(data.records[row['image']]['label'])
            values=[];batch=[]
            for box in row['boxes']:
                batch.extend(transform(crop(image,box,scale)) for scale in (1.,1.5))
            for start in range(0,len(batch),4):
                with torch.autocast('cuda',dtype=torch.float16):
                    value=normalize(model.encode_image(torch.stack(batch[start:start+4]).cuda()).float(),dim=-1)
                values.append(value.cpu())
            encoded[key]=torch.cat(values).reshape(len(row['boxes']),2,-1)
        result[name]=encoded
        print(f'{name}: consistently encoded {len(regions)} candidate sets',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    torch.save(result,path)
    archive.audit(OUT/'results/candidate_encoding_data_access.json');archive.close()


if __name__=='__main__':main()
