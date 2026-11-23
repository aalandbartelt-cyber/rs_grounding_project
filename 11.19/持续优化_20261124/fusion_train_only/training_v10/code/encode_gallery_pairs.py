"""Cross-compare the top gallery images with Qwen using query-visible evidence."""
import itertools
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[2]
OUT = BASE/'training_v10'
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, TrainArchive, crop, digest, write_json
from vlm import Qwen
import retrieval


def top_images(data, inputs, q, ids, model, config, limit=8):
    x = torch.stack([inputs[f'{q}:{image}'] for image in ids])
    local = torch.from_numpy(model.predict(x.numpy())).float()
    base = torch.tensor([data.retrieval[f'{q}:{image}']['base_score'] for image in ids])
    scores = base+config['amount']*local
    return [ids[index] for index in scores.argsort(descending=True, stable=True)[:limit].tolist()]


def preferred_box(data, query, image):
    record = data.records[image]
    if not len(record['boxes']):
        return None
    text = [data.text_features[name][query] for name in data.models]
    streams = data.images[image]['regions']
    similarity = .5*(streams[0]@text[0]+streams[1]@text[1])
    score = similarity+.1*record['scores'].clamp_min(1e-6).log()
    return record['boxes'][int(score.argmax())]


def comparison_image(images, boxes, first, second, reverse=False):
    order = (second, first) if reverse else (first, second)
    canvas = Image.new('RGB', (768, 320), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    for slot, index in enumerate(order):
        x = slot*384
        image = images[index]
        overview = image.copy()
        overview.thumbnail((178, 266))
        canvas.paste(overview, (x+4, 36))
        box = boxes[index]
        if box is None:
            detail = image.copy()
        else:
            scale_x = overview.width/image.width
            scale_y = overview.height/image.height
            coords = [float(box[0])*scale_x+x+4, float(box[1])*scale_y+36,
                      float(box[2])*scale_x+x+4, float(box[3])*scale_y+36]
            draw.rectangle(coords, outline='red', width=3)
            detail = crop(image, box, 1.35)
        detail.thumbnail((178, 266))
        canvas.paste(detail, (x+190, 36))
        draw.text((x+8, 8), ('A', 'B')[slot], fill='black', font=font)
        draw.text((x+190, 8), 'aircraft crop', fill='black', font=font)
    return canvas


@torch.inference_mode()
def compare(model, image, query):
    prompt = ('Two satellite-image candidates are shown side by side. In each half, the left pane is the full scene '
              'and the right pane enlarges the predicted aircraft. Which image shows an aircraft that best matches '
              'the visible paint, engine-count, and wing-layout attributes in this description? Ignore small text and '
              'choose by visible image evidence. Answer only A or B. Description: '+query)
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                            {'type': 'text', 'text': prompt}]}]
    batch = model.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                return_dict=True, return_tensors='pt').to('cuda')
    logits = model.model(**batch, logits_to_keep=1).logits[0, -1].float()
    a, b = model.tokenizer.encode('A', add_special_tokens=False), model.tokenizer.encode('B', add_special_tokens=False)
    if len(a) != 1 or len(b) != 1:
        raise ValueError('A/B answer labels must be single tokens')
    return float((logits[a[0]]-logits[b[0]]).cpu())


def main():
    torch.set_num_threads(4)
    (OUT/'features').mkdir(parents=True, exist_ok=True)
    data = Data()
    raw = retrieval.features(data)
    selected = json.loads((BASE/'training_v2/results/frozen_config.json').read_text(encoding='utf-8'))['retrieval']
    galleries = {f'cv{fold}': ids for fold, ids in enumerate(data.folds())}
    galleries['dev'] = data.dev_ids
    selections, pairs = {}, set()
    for name, ids in galleries.items():
        train_ids = data.fit_ids if name == 'dev' else [i for i in data.fit_ids if i not in ids]
        model = retrieval.fit(data, raw, train_ids, selected)
        queries = sorted({data.texts.index(text) for i in ids for text in data.records[i]['texts']})
        for q in queries:
            candidates = top_images(data, raw, q, ids, model, selected, 8)
            selections[f'{name}:{q}'] = candidates
            pairs.update((q, min(a, b), max(a, b)) for a, b in itertools.combinations(candidates, 2))
    job_payload = {'galleries': {key: list(map(int, value)) for key, value in galleries.items()},
                   'selections': selections, 'pair_count': len(pairs)}
    signature = {'jobs_sha256': __import__('hashlib').sha256(
                     json.dumps(job_payload, sort_keys=True).encode()).hexdigest(),
                 'qwen_weights_sha256': digest(BASE/'weights/Qwen3-VL-4B/model.safetensors.index.json'),
                 'code_sha256': digest(Path(__file__)), 'gallery_limit': 8,
                 'official_validation_or_test_used': False}
    destination = OUT/'features/qwen_gallery_pairs.json'
    if destination.exists():
        result = json.loads(destination.read_text(encoding='utf-8'))
        if result.get('signature') != signature:
            raise ValueError('Existing image-pair predictions were made for different jobs or prompts')
    else:
        result = {'signature': signature, 'jobs': job_payload, 'preferences': {}}
        write_json(destination, result)
    if len(result['preferences']) == len(pairs):
        print(f'Reusing all {len(pairs)} Qwen image-pair preferences', flush=True)
        return
    model, archive = Qwen(), TrainArchive()
    images, boxes = {}, {}

    def load(index):
        if index not in images:
            images[index] = archive.image(data.records[index]['label'])
            boxes[index] = preferred_box(data, 0, index)
        return images[index]

    for number, (q, first, second) in enumerate(sorted(pairs), 1):
        key = f'{q}:{first}:{second}'
        if key in result['preferences']:
            continue
        load(first); load(second)
        # Query-conditioned aircraft crops change by query, so compute them for this pair.
        boxes[first] = preferred_box(data, q, first)
        boxes[second] = preferred_box(data, q, second)
        prompt_image = comparison_image(images, boxes, first, second)
        reversed_image = comparison_image(images, boxes, first, second, reverse=True)
        forward = compare(model, prompt_image, data.texts[q])
        reverse = compare(model, reversed_image, data.texts[q])
        result['preferences'][key] = {'forward_A_minus_B': forward,
                                      'reverse_A_minus_B': reverse,
                                      'symmetric_preference': (forward-reverse)/2,
                                      'order_bias': (forward+reverse)/2}
        if number % 10 == 0 or number == len(pairs):
            write_json(destination, result)
        if number % 50 == 0 or number == len(pairs):
            print(f'Qwen gallery image comparisons {number}/{len(pairs)}', flush=True)
    archive.audit(OUT/'results/gallery_pair_data_access.json')
    archive.close()
    write_json(destination, result)


if __name__ == '__main__':
    main()
