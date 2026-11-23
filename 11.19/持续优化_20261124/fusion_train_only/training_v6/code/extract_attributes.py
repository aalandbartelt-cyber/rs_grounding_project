"""Query-independent aircraft attributes from predicted regions, without target boxes."""
import json
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, TrainArchive, crop, digest, iou, write_json
from vlm import Qwen

OUT = BASE/'training_v6'


def jobs(data, regions):
    unique, by_image = [], {}
    mapping = {'gallery': {}, 'grounding': {}}

    def add(image, box):
        for index in by_image.get(image, []):
            if float(iou(torch.tensor(box), torch.tensor([unique[index]['box']]))[0]) >= .9:
                return index
        index = len(unique)
        unique.append({'image': image, 'box': list(map(float, box))})
        by_image.setdefault(image, []).append(index)
        return index

    for image, row in enumerate(data.records):
        mapping['gallery'][str(image)] = [add(image, box.tolist()) for box in row['boxes']]
    for key, row in regions.items():
        mapping['grounding'][key] = [add(row['image'], box.tolist()) for box in row['boxes']]
    return unique, mapping


@torch.inference_mode()
def option_logits(model, image, prompt, options):
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
    batch = model.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 return_dict=True, return_tensors='pt').to('cuda')
    logits = model.model(**batch, logits_to_keep=1).logits[0, -1].float()
    ids = [model.tokenizer.encode(option, add_special_tokens=False) for option in options]
    assert all(len(value) == 1 for value in ids), 'Attribute choices must be single tokens'
    return logits[[value[0] for value in ids]].cpu().tolist()


def main():
    torch.set_num_threads(4)
    for name in ('features', 'results'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    data = Data()
    path = BASE/'training_v2/features/candidates.pt'
    candidates = torch.load(path, weights_only=True)['regions']
    regions, mapping = jobs(data, candidates)
    signature = {'candidates_sha256': digest(path), 'code_sha256': digest(Path(__file__)),
                 'queries_and_target_annotations_enter_attribute_prompts': False}
    destination = OUT/'features/aircraft_attributes.json'
    output = json.loads(destination.read_text(encoding='utf-8')) if destination.exists() else {
        'signature': signature, 'regions': regions, 'mapping': mapping, 'answers': {}}
    assert output['signature'] == signature
    if len(output['answers']) == len(regions):
        print('Reusing complete frozen aircraft attribute evidence', flush=True)
        return
    model, archive = Qwen(), TrainArchive()
    images = {}
    print(f'Classifying {len(regions)} distinct predicted aircraft regions', flush=True)
    for number, row in enumerate(regions):
        key = str(number)
        if key in output['answers']:
            continue
        image_id = row['image']
        if image_id not in images:
            images[image_id] = archive.image(data.records[image_id]['label'])
        image = crop(images[image_id], row['box'], 1.15)
        count = option_logits(model, image,
            'Inspect the central aircraft in this satellite crop. How many engines are visibly present on this aircraft? '
            'Answer only 2, 3, or 4. Answer 0 when no aircraft is present or engine count cannot be seen. '
            'Use visible image evidence only.', ('2', '3', '4', '0'))
        paint = option_logits(model, image,
            'Inspect the central aircraft paint in this satellite crop. Classify its fuselage and wings. '
            'W: predominantly white. B: substantial blue and white paint. O: another color or cannot determine. '
            'Ignore a small colored tail logo. Answer only W, B, or O.', ('W', 'B', 'O'))
        complete = model.match_score(
            'Does this satellite crop clearly contain a complete civilian passenger airplane, with its fuselage '
            'and both main wings inside the crop? Answer only Yes or No.', image)
        output['answers'][key] = {'engine_logits_2_3_4_unknown': count, 'paint_logits_white_blue_other': paint,
                                  'complete_aircraft_yes_minus_no': complete}
        write_json(destination, output)
        if (number+1) % 20 == 0 or number+1 == len(regions):
            print(f'Aircraft attribute evidence {number+1}/{len(regions)}', flush=True)
    archive.audit(OUT/'results/attribute_extraction_data_access.json')
    archive.close()


if __name__ == '__main__':
    main()
