"""Matched image/text GeoRSCLIP QuickGELU ablation, using original train only."""
import json
import sys
from pathlib import Path

import open_clip
import torch
from torch.nn.functional import normalize

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, TrainArchive, crop, digest, write_json

OUT = BASE/'training_v5'


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    for name in ('features', 'results'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    data = Data()
    checkpoint = BASE/'weights/GeoRSCLIP/ckpt/RS5M_ViT-L-14-336.pt'
    model, _, transform = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained=None,
                                                               force_quick_gelu=True)
    state = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
    state = state.get('state_dict', state.get('model', state))
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    del state
    model = model.cuda().eval()
    assert type(model.visual.transformer.resblocks[0].mlp.gelu).__name__ == 'QuickGELU'
    tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    queries = json.loads((BASE/'training_v2/features/query_prompts.json').read_text(encoding='utf-8'))
    output = {'signature': {'checkpoint_sha256': digest(checkpoint), 'activation': 'QuickGELU',
                            'visual_and_text_activation_matched': True,
                            'query_prompts_sha256': digest(BASE/'training_v2/features/query_prompts.json')},
              'text': {}, 'gallery': {}, 'grounding': {}}
    for mode in ('literal', 'concise', 'attribute'):
        tokens = tokenizer([row[mode] for row in queries['prompts']]).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            output['text'][mode] = normalize(model.encode_text(tokens).float(), dim=-1).cpu()
    output['text']['ensemble'] = normalize((output['text']['literal']+output['text']['concise'])/2, dim=-1)
    fit_queries = sorted({data.texts.index(t) for i in data.fit_ids for t in data.records[i]['texts']})
    output['text']['centered'] = normalize(output['text']['ensemble']-.75*output['text']['ensemble'][fit_queries].mean(0), dim=-1)
    regions = torch.load(BASE/'training_v2/features/candidates.pt', weights_only=True)['regions']
    archive = TrainArchive()

    def encode(views):
        values = []
        for start in range(0, len(views), 4):
            with torch.autocast('cuda', dtype=torch.float16):
                value = normalize(model.encode_image(torch.stack(views[start:start+4]).cuda()).float(), dim=-1)
            values.append(value.cpu())
        return torch.cat(values)

    for i, row in enumerate(data.records):
        image = archive.image(row['label'])
        views = [transform(image)]
        views.extend(transform(crop(image, box, scale)) for box in row['boxes'] for scale in (1., 1.5))
        values = encode(views)
        output['gallery'][i] = {'whole': values[0], 'regions': values[1:].reshape(len(row['boxes']), 2, -1)}
        if (i+1) % 10 == 0:
            print(f'QuickGELU whole-gallery encoding {i+1}/{len(data.records)}', flush=True)
    for number, (key, row) in enumerate(regions.items(), 1):
        image = archive.image(data.records[row['image']]['label'])
        views = [transform(crop(image, box, scale)) for box in row['boxes'] for scale in (1., 1.5)]
        output['grounding'][key] = encode(views).reshape(len(row['boxes']), 2, -1)
        if number % 10 == 0 or number == len(regions):
            print(f'QuickGELU regional encoding {number}/{len(regions)}', flush=True)
    torch.save(output, OUT/'features/GeoRSCLIP_QuickGELU.pt')
    archive.audit(OUT/'results/activation_encoding_data_access.json')
    archive.close()
    write_json(OUT/'results/activation_ablation.json', {
        **output['signature'], 'source': 'Official README initializes one GeoRSCLIP variant from openai; this L/336 activation choice remains an experimental ablation.',
        'official_validation_or_test_used': False, 'all_data_from_original_training_whitelist': True})


if __name__ == '__main__':
    main()
