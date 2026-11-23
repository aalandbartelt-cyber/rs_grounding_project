"""Generate rotated-view GeoGround proposals without reading target boxes."""
import json

import torch

from inputs import BASE,OUT,Data,TrainArchive,parse,inverse_box,write_json
from vlm import GeoGround


def main():
    torch.set_num_threads(4)
    data=Data();archive=TrainArchive();model=GeoGround()
    output=OUT/'features/geoground_views.json'
    result=json.loads(output.read_text(encoding='utf-8')) if output.exists() else {}
    translations=json.loads((BASE/'features/translations.json').read_text(encoding='utf-8'))
    for number,(key,job) in enumerate(data.jobs.items(),1):
        if key not in result: result[key]={}
        image=archive.image(data.records[job['image']]['label'])
        for rotation in (90,180,270):
            if str(rotation) in result[key]:continue
            view=image.rotate(rotation,expand=True)
            answer=model.answer(translations[data.texts[job['query']]],view)
            box=parse(answer,view.width,view.height,'1000')
            box=inverse_box(box,image.width,image.height,rotation) if box is not None else None
            result[key][str(rotation)]={'answer':answer,'mapped_box_xyxy':box.tolist() if box is not None else None}
            write_json(output,result)
        if number%5==0 or number==len(data.jobs):
            print(f'GeoGround three rotated views {number}/{len(data.jobs)}',flush=True)
    archive.audit(OUT/'results/geoground_views_data_access.json');archive.close()


if __name__=='__main__':main()
