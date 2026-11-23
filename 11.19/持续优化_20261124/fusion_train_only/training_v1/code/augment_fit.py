"""Encode augmented GT crops of fit images only, for supervised visual adaptation."""
import gc
import random

import open_clip
import torch
from PIL import Image,ImageEnhance,ImageOps
from torch.nn.functional import normalize

from data import BASE,Data
from common import TrainArchive,crop,write_json
from encode_retrieval import SPECS


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    data=Data();archive=TrainArchive();rng=random.Random(20261123)
    jobs=[]
    for i in data.fit_ids:
        for j,text in enumerate(data.records[i]['texts']):
            for variant in range(8):
                jobs.append({'image':i,'annotation':j,'query':data.texts.index(text),
                             'rotation':90*(variant%4),'flip':variant>=4,
                             'brightness':rng.uniform(.9,1.1),'scale':rng.uniform(.9,1.1)})
    output={'jobs':jobs,'source':'original 33 fit images only','official_validation_or_test_used':False}
    for name,(architecture,checkpoint) in SPECS.items():
        model,_,transform=open_clip.create_model_and_transforms(architecture,pretrained=None)
        state=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
        state=state.get('state_dict',state.get('model',state))
        model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
        del state
        model=model.cuda().eval()
        vectors=[]
        for number,job in enumerate(jobs,1):
            row=data.records[job['image']]
            assert job['image'] in data.fit_ids
            image=archive.image(row['label'])
            box=row['truth'][job['annotation']]
            views=[]
            for scale in (1.,1.5):
                view=crop(image,box,scale*job['scale'])
                view=view.rotate(job['rotation'],expand=True)
                if job['flip']: view=ImageOps.mirror(view)
                view=ImageEnhance.Brightness(view).enhance(job['brightness'])
                views.append(transform(view))
            batch=torch.stack(views).cuda()
            with torch.autocast('cuda',dtype=torch.float16):
                vector=normalize(model.encode_image(batch).float(),dim=-1).mean(0).cpu()
            vectors.append(vector)
            if number%100==0 or number==len(jobs):
                print(f'{name} training augmentation {number}/{len(jobs)}',flush=True)
        output[name]=torch.stack(vectors)
        del model
        gc.collect();torch.cuda.empty_cache()
    torch.save(output,BASE/'training_v1/features/augmented_fit_regions.pt')
    archive.audit(BASE/'training_v1/results/augmentation_data_access.json')
    archive.close()


if __name__=='__main__': main()
