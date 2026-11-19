"""Offline retrieval and grounding for civilian passenger-aircraft descriptions."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from PIL import Image
from safetensors.torch import load_file
from torch.nn.functional import linear, normalize
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          ChineseCLIPModel, ChineseCLIPProcessor)


ROOT = Path(__file__).resolve().parent
PROMPT = 'civilian passenger airliner.'
CIVIL_AIRCRAFT = re.compile(r'民航客机|民用客机|民用航班客机|civilian passenger airliner', re.I)
EXCLUDED = re.compile(r'军[用机事]|作战|武装|侦察|战斗机|轰炸机|军舰|导弹|炸弹|武器|military|weapon|missile', re.I)
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.bmp'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def image_catalog(root):
    info = read_json(root / 'info.json')
    if isinstance(info, dict) and 'images' in info:
        info = info['images']
    if isinstance(info, dict):
        info = [dict(value, image_id=value.get('image_id', key)) if isinstance(value, dict)
                else {'image_id': key, 'file_name': value} for key, value in info.items()]
    if not isinstance(info, list) or not info:
        raise ValueError('info.json must contain a nonempty image catalog')
    folder = (root / 'images').resolve()
    by_stem = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            by_stem.setdefault(path.stem, []).append(path)
    result = {}
    for row in info:
        image_id = row['image_id'] if 'image_id' in row else row['id']
        key = str(image_id)
        if key in result:
            raise ValueError(f'Duplicate image ID {image_id!r}')
        name = next((row[k] for k in ('file_name', 'filename', 'image_name', 'image_path', 'path') if row.get(k)), None)
        if name is None:
            matches = by_stem.get(key, [])
            if len(matches) != 1:
                raise ValueError(f'Cannot resolve filename for image ID {image_id!r}')
            path = matches[0]
        else:
            relative = Path(name.replace('\\', '/'))
            if relative.parts[:1] == ('images',):
                relative = Path(*relative.parts[1:])
            path = folder / relative
        path = path.resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError(f'Invalid image path for image ID {image_id!r}: {path}')
        result[key] = (image_id, path)
    return result


def approved(text):
    return CIVIL_AIRCRAFT.search(text) is not None and EXCLUDED.search(text) is None


def resize_boxes(boxes, scale, width, height):
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    radius = (boxes[:, 2:] - boxes[:, :2]) * scale / 2
    out = torch.cat((centers - radius, centers + radius), dim=1)
    out[:, 0::2].clamp_(0, width)
    out[:, 1::2].clamp_(0, height)
    return out


class Inference:
    def __init__(self, weights, device):
        self.weights = weights
        self.device = device
        self.clip = ChineseCLIPModel.from_pretrained(weights / 'chinese-clip-vit-base-patch16',
                                                      local_files_only=True, use_safetensors=True).to(device).eval()
        self.processor = ChineseCLIPProcessor.from_pretrained(weights / 'chinese-clip-vit-base-patch16',
                                                               local_files_only=True)
        self.retrieval_adapter = load_file(str(weights / 'retrieval_adapter.safetensors'), device=str(device))
        self.detect = None

    def encode_text(self, texts):
        encoded = []
        for start in range(0, len(texts), 16):
            inputs = self.processor(text=texts[start:start + 16], return_tensors='pt',
                                    padding=True, truncation=True, max_length=128).to(self.device)
            encoded.append(normalize(self.clip.get_text_features(**inputs), dim=-1))
        return dict(zip(texts, torch.cat(encoded))) if encoded else {}

    def encode_visual(self, images, adapter):
        pixels = self.processor(images=images, return_tensors='pt')['pixel_values'].to(self.device)
        pooled = self.clip.vision_model(pixel_values=pixels, return_dict=True).pooler_output
        residual = linear(linear(pooled, adapter['down.weight']), adapter['up.weight'])
        return normalize(self.clip.visual_projection(pooled) + residual, dim=-1)

    def gallery(self, catalog):
        paths = [path for _, path in catalog.values()]
        encoded = []
        for start in range(0, len(paths), 8):
            images = []
            for path in paths[start:start + 8]:
                with Image.open(path) as opened:
                    images.append(opened.convert('RGB'))
            encoded.append(self.encode_visual(images, self.retrieval_adapter))
        return torch.cat(encoded)

    def candidates(self, path):
        if self.detect is None:
            directory = self.weights / 'grounding-dino-tiny'
            self.detect = AutoModelForZeroShotObjectDetection.from_pretrained(
                directory, local_files_only=True, use_safetensors=True,
                disable_custom_kernels=True).to(self.device).eval()
            self.dino_processor = AutoProcessor.from_pretrained(directory, local_files_only=True, use_fast=False)
            self.context_adapter = load_file(str(self.weights / 'context_adapter.safetensors'), device=str(self.device))
        with Image.open(path) as opened:
            image = opened.convert('RGB')
        w, h = image.size
        batch = self.dino_processor(images=image, text=PROMPT, return_tensors='pt').to(self.device)
        pred = self.dino_processor.post_process_grounded_object_detection(
            self.detect(**batch), batch.input_ids, threshold=0.15, text_threshold=0.25,
            target_sizes=[(h, w)])[0]
        boxes, scores = pred['boxes'].float().cpu(), pred['scores'].float().cpu()
        boxes[:, 0::2].clamp_(0, w)
        boxes[:, 1::2].clamp_(0, h)
        keep = boxes.isfinite().all(1) & scores.isfinite() & (boxes[:, 2:] > boxes[:, :2]).all(1)
        boxes, scores = boxes[keep], scores[keep]
        order = scores.argsort(descending=True, stable=True)[:20]
        boxes, scores = boxes[order], scores[order]
        if not len(boxes):
            return None
        crops = [image.crop((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)))
                 for x1, y1, x2, y2 in resize_boxes(boxes, 1.5, w, h).tolist()]
        visual = torch.cat([self.encode_visual(crops[i:i + 8], self.context_adapter)
                            for i in range(0, len(crops), 8)])
        return resize_boxes(boxes, 0.9, w, h), scores, visual


@torch.inference_mode()
def predict(queries, catalog, weights, device):
    kinds = {'retrieval', 'grounding'}
    if any(q['type'] not in kinds or not isinstance(q['text'], str) for q in queries):
        raise ValueError('Expected retrieval or grounding queries with text strings')
    supported = list(dict.fromkeys(q['text'].strip() for q in queries if approved(q['text'])))
    embeddings, rankings, candidates = {}, {}, {}
    if supported:
        model = Inference(weights, device)
        embeddings = model.encode_text(supported)
        retrieval_text = list(dict.fromkeys(q['text'].strip() for q in queries
                            if q['type'] == 'retrieval' and q['text'].strip() in embeddings))
        if retrieval_text:
            if len(catalog) < 5:
                raise ValueError('The retrieval gallery must contain at least five images')
            ids = [image_id for image_id, _ in catalog.values()]
            gallery = model.gallery(catalog)
            ranks = (torch.stack([embeddings[t] for t in retrieval_text]) @ gallery.T).argsort(
                dim=1, descending=True, stable=True)[:, :5].tolist()
            rankings = {t: [ids[i] for i in row] for t, row in zip(retrieval_text, ranks)}
    results = []
    for query in queries:
        text = query['text'].strip()
        row = {'query_id': query['query_id'], 'type': query['type']}
        if query['type'] == 'retrieval':
            row['ranking'] = rankings.get(text, [])
        else:
            image_id = query['image_id']
            row.update(image_id=image_id, category_id=4, score=0.0, bbox=[0.0] * 4)
            if text in embeddings:
                key = str(image_id)
                if key not in candidates:
                    candidates[key] = model.candidates(catalog[key][1])
                prediction = candidates[key]
                if prediction is not None:
                    boxes, scores, visual = prediction
                    best = int((visual @ embeddings[text] + 0.1 * scores.to(device).log()).argmax())
                    x1, y1, x2, y2 = boxes[best].tolist()
                    row.update(score=float(scores[best]), bbox=[x1, y1, x2 - x1, y2 - y1])
        results.append(row)
    return results, sum(q['text'].strip() not in embeddings for q in queries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('/input_path'))
    parser.add_argument('--output', type=Path, default=Path('/output_path'))
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    queries = read_json(args.input / 'queries.json')
    rows, skipped = predict(queries, image_catalog(args.input), ROOT / 'weights', device)
    args.output.mkdir(parents=True, exist_ok=True)
    pending = args.output / 'pred.json.tmp'
    pending.write_text(json.dumps(rows, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(args.output / 'pred.json')
    print(f'{len(rows)} predictions, {skipped} out-of-scope queries, {device}, '
          f'{time.perf_counter() - started:.2f} seconds', flush=True)


if __name__ == '__main__':
    main()
