"""Compare candidate pairs in both display orders using frozen Qwen logits."""
import itertools
import json

import torch
from PIL import Image,ImageDraw,ImageFont

from inputs import BASE,OUT,Data,TrainArchive,crop,write_json,iou
from vlm import Qwen


def comparison_image(image,boxes,first,second):
    canvas=Image.new('RGB',(768,288),'white')
    overview=image.copy();overview.thumbnail((256,256));canvas.paste(overview,(0,32))
    draw=ImageDraw.Draw(canvas);font=ImageFont.load_default(size=20)
    sx,sy=overview.width/image.width,overview.height/image.height
    for slot,index,color in ((0,first,'blue'),(1,second,'red')):
        box=boxes[index].tolist();label=('A','B')[slot]
        draw.rectangle([box[0]*sx,box[1]*sy+32,box[2]*sx,box[3]*sy+32],outline=color,width=3)
        draw.text((box[0]*sx,max(32,box[1]*sy+10)),label,fill=color,font=font)
        view=crop(image,box,1.25);view.thumbnail((250,250))
        x=256+slot*256;canvas.paste(view,(x+3,35))
        draw.text((x+8,3),label,fill=color,font=font)
    return canvas


@torch.inference_mode()
def logit_difference(model,image,text):
    prompt=('左侧是卫星图像中的两个候选框，右侧是候选 A 和 B 的局部图。'
            '比较可见的发动机数量、机翼布局、涂装和周边环境，判断哪个候选更符合下列描述。'
            '只回答 A 或 B。描述：'+text)
    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
    batch=model.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                              return_dict=True,return_tensors='pt').to('cuda')
    logits=model.model(**batch,logits_to_keep=1).logits[0,-1].float()
    a=model.tokenizer.encode('A',add_special_tokens=False);b=model.tokenizer.encode('B',add_special_tokens=False)
    assert len(a)==len(b)==1
    return float((logits[a[0]]-logits[b[0]]).cpu())


def main():
    torch.set_num_threads(4)
    data=Data();archive=TrainArchive();model=Qwen()
    semantic=json.loads((BASE/'training_v1/features/qwen_semantic.json').read_text(encoding='utf-8'))
    path=OUT/'features/qwen_pair_preferences.json'
    output=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for number,(key,row) in enumerate(data.grounding.items(),1):
        n=len(row['boxes']);s=row['scalar']
        prior=row['base_score']+.03*torch.tensor(semantic['grounding'][key]).div(5).tanh()
        order=prior.argsort(descending=True,stable=True).tolist()
        candidates=[]
        for index in order:
            if candidates and float(iou(row['boxes'][index],row['boxes'][candidates]).max())>=.7:
                continue
            candidates.append(index)
            if len(candidates)==6:break
        record=output.get(key,{'candidate_indices':candidates,'pairs':{}})
        assert record['candidate_indices']==candidates
        image=archive.image(data.records[row['image']]['label'])
        for first,second in itertools.combinations(candidates,2):
            pair=f'{first}:{second}'
            if pair in record['pairs']:continue
            forward=logit_difference(model,comparison_image(image,row['boxes'],first,second),data.texts[row['query']])
            reverse=logit_difference(model,comparison_image(image,row['boxes'],second,first),data.texts[row['query']])
            record['pairs'][pair]={'A_minus_B_forward':forward,'A_minus_B_reverse':reverse,
                                  'symmetric_preference':(forward-reverse)/2}
            output[key]=record;write_json(path,output)
        if key not in output:output[key]=record;write_json(path,output)
        if number%5==0 or number==len(data.grounding):
            print(f'Qwen symmetric pair comparison {number}/{len(data.grounding)}',flush=True)
    archive.audit(OUT/'results/qwen_pair_data_access.json');archive.close()


if __name__=='__main__':main()
