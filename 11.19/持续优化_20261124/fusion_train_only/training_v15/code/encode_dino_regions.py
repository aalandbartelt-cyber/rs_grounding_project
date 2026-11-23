"""Encode prediction-only candidate crops with the pinned DINOv2 checkpoint."""
import sys
from pathlib import Path

import torch
from torch.nn.functional import normalize
from transformers import AutoImageProcessor, Dinov2Model

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, TrainArchive, crop, digest, write_json

OUT = BASE/'training_v15'
MODEL_DIR = BASE/'weights/DINOv2-base'
REVISION = 'f9e44c814b77203eaa57a6bdbbd535f21ede1415'
SIZE = 518
BATCH_SIZE = 4


def embed_views(model, processor, views):
    outputs = []
    for start in range(0, len(views), BATCH_SIZE):
        batch = processor(images=views[start:start+BATCH_SIZE], return_tensors='pt',
                          size={'height': SIZE, 'width': SIZE},
                          crop_size={'height': SIZE, 'width': SIZE})
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            hidden = model(pixel_values=batch['pixel_values'].cuda(),
                           interpolate_pos_encoding=True).last_hidden_state.float()
        cls = normalize(hidden[:, 0], dim=-1)
        patch_mean = normalize(hidden[:, 1:].mean(1), dim=-1)
        outputs.extend(torch.cat([cls, patch_mean], 1).cpu())
    return outputs


def main():
    torch.set_num_threads(4)
    (OUT/'features').mkdir(parents=True, exist_ok=True)
    (OUT/'results').mkdir(parents=True, exist_ok=True)
    data = Data()
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=False)
    model = Dinov2Model.from_pretrained(MODEL_DIR, local_files_only=True).eval().cuda()
    archive = TrainArchive()
    encoded = {}

    for image_id in range(len(data.records)):
        keys = data.keys([image_id])
        if not keys:
            continue
        image = archive.image(data.records[image_id]['label'])
        signatures, views, view_indices = [], [], []
        seen = set()
        for key in keys:
            for box in data.grounding[key]['boxes']:
                signature = tuple(round(float(value), 2) for value in box.tolist())
                signatures.append(signature)
                if signature not in seen:
                    seen.add(signature)
                    for scale in (1., 1.5):
                        views.append(crop(image, box, scale))
                        view_indices.append((signature, scale))
        vectors = embed_views(model, processor, views)
        cache = {}
        for index, (signature, scale) in enumerate(view_indices):
            cache[(signature, scale)] = vectors[index]
        for key in keys:
            rows = []
            for box in data.grounding[key]['boxes']:
                signature = tuple(round(float(value), 2) for value in box.tolist())
                tight, context = cache[(signature, 1.)], cache[(signature, 1.5)]
                rows.append(torch.cat([tight[:768], context[:768], tight[768:], context[768:]]))
            encoded[key] = torch.stack(rows)
        print(f'DINOv2 encoded train image {image_id+1}/{len(data.records)} '
              f'({len(keys)} query-image pairs)', flush=True)

    checkpoint = MODEL_DIR/'model.safetensors'
    feature_path = OUT/'features/dinov2_region_features.pt'
    torch.save({'features': encoded, 'source_model_revision': REVISION,
                'input_size': SIZE, 'views': ['tight', '1.5x context'],
                'feature_order': ['tight_cls', 'context_cls', 'tight_patch_mean', 'context_patch_mean']},
               feature_path)
    archive.audit(OUT/'results/dino_region_data_access.json')
    archive.close()
    write_json(OUT/'results/dino_features_manifest.json', {
        'source': 'facebook/dinov2-base', 'license': 'Apache-2.0',
        'revision': REVISION, 'weights_sha256': digest(checkpoint),
        'features_sha256': digest(feature_path), 'code_sha256': digest(Path(__file__)),
        'source_images': '49 whitelisted original-train images only',
        'official_validation_or_test_used': False,
        'internal_dev_images_used_for_unlabeled_feature_extraction': True,
        'ground_truth_used_during_feature_extraction': False,
        'candidate_source': 'frozen predicted boxes from the original-train feature cache',
        'input_size': SIZE, 'views': ['tight', '1.5x context'],
        'feature_width_per_candidate': 3072})
    print(f'Saved {len(encoded)} query-image feature rows to {feature_path}', flush=True)


if __name__ == '__main__':
    main()
