"""Complete frozen Qwen features across all original-training galleries."""
import json

import torch

from data import BASE,Data
from common import TrainArchive,write_json
from vlm import Qwen


def main():
    torch.set_num_threads(4)
    data=Data();archive=TrainArchive()
    path=BASE/'training_v1/features/qwen_all_training_gallery.json'
    known=json.loads((BASE/'training_v1/features/qwen_semantic.json').read_text(encoding='utf-8'))['retrieval']
    if path.exists(): known.update(json.loads(path.read_text(encoding='utf-8')))
    model=Qwen();total=len(data.texts)*len(data.records)
    for q,text in enumerate(data.texts):
        for i,record in enumerate(data.records):
            key=f'{q}:{i}'
            if key not in known:
                image=archive.image(record['label'])
                prompt=('Is there at least one civilian passenger aircraft in this satellite image that '
                        'matches the following description? Compare visible engine count, paint, wing '
                        'arrangement and surroundings. Answer only Yes or No. Description: '+text)
                known[key]=model.match_score(prompt,image)
                write_json(path,known)
            number=q*len(data.records)+i+1
            if number%100==0 or number==total:
                print(f'Full original-training gallery features {number}/{total}',flush=True)
    write_json(path,known)
    archive.audit(BASE/'training_v1/results/full_gallery_features_data_access.json')
    archive.close()


if __name__=='__main__':main()
