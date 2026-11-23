"""Offline SigLIP 2 encoding of train-only predicted regions and query text."""
import json
import sys
from pathlib import Path

import torch
from torch.nn.functional import normalize
from transformers import AutoModel, AutoProcessor

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, TrainArchive, crop, digest, write_json

OUT = BASE/'training_v18'
MODEL = BASE/'weights/SigLIP2-base-512'
REVISION = 'a89f5c5093f902bf39d3cd4d81d2c09867f0724b'


@torch.inference_mode()
def encode_images(model, processor, views):
    output = []
    for start in range(0, len(views), 4):
        batch = processor(images=views[start:start+4], return_tensors='pt')
        with torch.autocast('cuda', dtype=torch.float16):
            encoded = model.get_image_features(pixel_values=batch['pixel_values'].cuda())
        output.append(normalize(encoded.float(), dim=-1).cpu())
    return torch.cat(output)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    (OUT/'features').mkdir(parents=True, exist_ok=True)
    (OUT/'results').mkdir(parents=True, exist_ok=True)
    data = Data()
    prompts = json.loads((BASE/'training_v2/features/query_prompts.json').read_text(encoding='utf-8'))
    assert prompts['texts'] == data.texts and prompts['derived_only_from_query_text']
    assert prompts['image_ids_or_target_boxes_used'] is False
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True).eval().cuda()
    text_features = {}
    for mode in ('chinese', 'literal', 'concise', 'attribute'):
        texts = data.texts if mode == 'chinese' else [row[mode] for row in prompts['prompts']]
        batch = processor(text=texts, padding='max_length', truncation=True, max_length=64,
                          return_tensors='pt')
        with torch.autocast('cuda', dtype=torch.float16):
            encoded = model.get_text_features(input_ids=batch['input_ids'].cuda())
        text_features[mode] = normalize(encoded.float(), dim=-1).cpu()
    text_features['ensemble'] = normalize(text_features['chinese']+text_features['concise'], dim=-1)

    archive = TrainArchive()
    regions, gallery = {}, {}
    for image_id, record in enumerate(data.records):
        image = archive.image(record['label'])
        keys = data.keys([image_id])
        views, slots = [image], [('whole',)]
        seen = set()
        for box in record['boxes']:
            sig = tuple(round(float(value), 3) for value in box.tolist())
            for scale in (1., 1.5):
                if (sig, scale) not in seen:
                    seen.add((sig, scale))
                    views.append(crop(image, box, scale))
                    slots.append((sig, scale))
        for key in keys:
            for box in data.grounding[key]['boxes']:
                sig = tuple(round(float(value), 3) for value in box.tolist())
                for scale in (1., 1.5):
                    if (sig, scale) not in seen:
                        seen.add((sig, scale))
                        views.append(crop(image, box, scale))
                        slots.append((sig, scale))
        encoded = encode_images(model, processor, views)
        lookup = {slot: encoded[index] for index, slot in enumerate(slots)}
        gallery_rows = []
        for box in record['boxes']:
            sig = tuple(round(float(value), 3) for value in box.tolist())
            gallery_rows.append(torch.stack([lookup[(sig, 1.)], lookup[(sig, 1.5)]]))
        gallery[image_id] = {'whole': encoded[0],
                             'regions': torch.stack(gallery_rows) if gallery_rows else torch.empty(0, 2, 768)}
        for key in keys:
            candidates = []
            for box in data.grounding[key]['boxes']:
                sig = tuple(round(float(value), 3) for value in box.tolist())
                candidates.append(torch.stack([lookup[(sig, 1.)], lookup[(sig, 1.5)]]))
            regions[key] = torch.stack(candidates)
        print(f'SigLIP2 encoded original-train image {image_id+1}/{len(data.records)}', flush=True)
    destination = OUT/'features/siglip2_train_features.pt'
    torch.save({'texts': data.texts, 'text_features': text_features,
                'regions': regions, 'gallery': gallery, 'revision': REVISION}, destination)
    archive.audit(OUT/'results/siglip2_data_access.json')
    archive.close()
    write_json(OUT/'results/siglip2_features_manifest.json', {
        'source': 'google/siglip2-base-patch16-512', 'revision': REVISION,
        'license': 'Apache-2.0', 'weights_sha256': digest(MODEL/'model.safetensors'),
        'features_sha256': digest(destination), 'code_sha256': digest(Path(__file__)),
        'official_validation_or_test_used': False,
        'ground_truth_used_in_encoding': False,
        'image_scope': '49 whitelisted original-train images',
        'gallery_uses_query_independent_detector_regions': True,
        'text_inputs_derived_from_query_text_only': True})
    print('Saved SigLIP 2 features:', destination, flush=True)


if __name__ == '__main__':
    main()
