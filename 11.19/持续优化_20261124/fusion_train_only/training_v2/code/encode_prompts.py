"""Domain terminology and attribute-focused text prompts, derived from queries only."""
import gc
import json
import re

import open_clip
import torch
from torch.nn.functional import normalize

from inputs import BASE,OUT,Data,write_json


def prompts(text,translation):
    literal=translation.replace('biplane configuration','two-main-wing arrangement').replace('upper-wing','high-wing')
    color='blue and white' if '蓝白' in text else 'white' if '白色' in text else ''
    count=None
    for number,pattern in ((4,r'四(?:台|发|引擎)'),(3,r'三(?:台|发|引擎)'),(2,r'两台|双发|双引擎')):
        if re.search(pattern,text):count=number;break
    engine=f'with {count} engines' if count is not None else 'with engines'
    if '翼上' in text:engine+=' above the wings'
    elif any(term in text for term in ('翼下','翼下方','翼吊式')):engine+=' under the wings'
    location='in flight' if '飞行' in text or '空中' in text else 'parked on an airport apron' if '停机坪' in text else ''
    concise=f'A satellite photo of a {color} passenger airplane {engine}, {location}.'.replace('  ',' ')
    attribute=f'A {color} civilian aircraft {engine}.'.replace('  ',' ')
    return {'literal':literal,'concise':concise,'attribute':attribute,
            'engine_count_explicit_in_query':count}


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    data=Data();translations=json.loads((BASE/'features/translations.json').read_text(encoding='utf-8'))
    descriptions=[prompts(text,translations[text]) for text in data.texts]
    write_json(OUT/'features/query_prompts.json',{'texts':data.texts,'prompts':descriptions,
        'derived_only_from_query_text':True,'image_ids_or_target_boxes_used':False})
    output={'texts':data.texts}
    for name,architecture,checkpoint in (
        ('RemoteCLIP','ViT-L-14',BASE/'weights/RemoteCLIP/RemoteCLIP-ViT-L-14.pt'),
        ('GeoRSCLIP','ViT-L-14-336',BASE/'weights/GeoRSCLIP/ckpt/RS5M_ViT-L-14-336.pt')):
        model=open_clip.create_model(architecture,pretrained=None)
        state=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
        state=state.get('state_dict',state.get('model',state))
        model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
        del state
        model=model.cuda().eval();tokenizer=open_clip.get_tokenizer(architecture)
        modes={}
        for mode in ('literal','concise','attribute'):
            rows=[]
            for start in range(0,len(descriptions),16):
                batch=tokenizer([row[mode] for row in descriptions[start:start+16]]).cuda()
                with torch.autocast('cuda',dtype=torch.float16):
                    rows.append(normalize(model.encode_text(batch).float(),dim=-1).cpu())
            modes[mode]=torch.cat(rows)
        modes['ensemble']=normalize((modes['literal']+modes['concise'])/2,dim=-1)
        fit_queries=sorted(set(data.texts.index(t) for i in data.fit_ids for t in data.records[i]['texts']))
        mean=modes['ensemble'][fit_queries].mean(0)
        modes['centered']=normalize(modes['ensemble']-.75*mean,dim=-1)
        output[name]=modes
        print(f'{name}: encoded literal, concise and attribute prompts',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    torch.save(output,OUT/'features/query_features.pt')


if __name__=='__main__':main()
