"""Select a RemoteCLIP adapter using grouped folds within original-train fit images."""
import gc
import itertools
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import open_clip
import torch
from PIL import ImageEnhance, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, TrainArchive, crop, digest, write_json
sys.path.insert(0, str(BASE/'training_v1/code'))
import lora_visual
from train import chosen_boxes, grounding_metric
from score_round import measure
import retrieval

OUT = BASE/'training_v11'


def load_model(rank, blocks):
    model, _, transform = open_clip.create_model_and_transforms('ViT-L-14', pretrained=None)
    checkpoint = BASE/'weights/RemoteCLIP/RemoteCLIP-ViT-L-14.pt'
    state = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
    state = state.get('state_dict', state.get('model', state))
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    del state
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in model.visual.transformer.resblocks[-blocks:]:
        for module, name in ((block.attn, 'in_proj_weight'), (block.attn.out_proj, 'weight'),
                             (block.mlp.c_fc, 'weight'), (block.mlp.c_proj, 'weight')):
            parametrize.register_parametrization(module, name,
                                                  lora_visual.LowRank(getattr(module, name).shape, rank=rank))
    return model.cuda(), transform


def train(data, ids, config, archive):
    assert set(ids) <= set(data.fit_ids), 'LoRA supervision outside the original fit images'
    torch.manual_seed(20261123)
    rng = random.Random(20261123)
    model, transform = load_model(config['rank'], config['blocks'])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=1e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    specs = lora_visual.sample_specs(data, ids)
    images = {i: archive.image(data.records[i]['label']) for i in ids}
    texts = data.text_features['RemoteCLIP'].cuda()
    history = []
    for epoch in range(1, config['epochs']+1):
        model.train()
        rng.shuffle(specs)
        total = count = 0
        for start in range(0, len(specs), 4):
            rows = specs[start:start+4]
            views = []
            for row in rows:
                view = crop(images[row['image']], row['box'], rng.choice((1., 1.25, 1.5)))
                if rng.random() < .5:
                    view = ImageOps.mirror(view)
                view = view.rotate(rng.choice((0, 90, 180, 270)), expand=True)
                view = ImageEnhance.Brightness(view).enhance(rng.uniform(.9, 1.1))
                views.append(transform(view))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                visual = F.normalize(model.encode_image(torch.stack(views).cuda()).float(), dim=-1)
                query = torch.tensor([row['query'] for row in rows], device='cuda')
                labels = torch.tensor([row['label'] for row in rows], device='cuda')
                weights = torch.tensor([row['weight'] for row in rows], device='cuda')
                loss = (F.binary_cross_entropy_with_logits(20*(visual*texts[query]).sum(1)-5,
                       labels, reduction='none')*weights).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())*len(rows)
            count += len(rows)
        entry = {'epoch': epoch, 'mean_loss': total/count}
        history.append(entry)
    state = {name: value.detach().cpu().clone() for name, value in model.named_parameters()
             if value.requires_grad}
    return model, transform, state, history


def cv_trial(data, fold, config):
    validation = fold
    training = [image for image in data.fit_ids if image not in validation]
    archive = TrainArchive()
    model, transform, state, history = train(data, training, config, archive)
    visual = lora_visual.encode_grounding(model, transform, data, archive, validation)
    scores = lora_visual.combined_scores(data, validation, visual)
    metrics = []
    details = {}
    for scale in (.9, .95, 1.):
        predictions = chosen_boxes(data, validation, scores, scale)
        metric = grounding_metric(data, validation, predictions)
        metrics.append({**metric, 'scale': scale})
        details[scale] = predictions
    archive.audit(OUT/'results'/f'fold_{data.folds().index(validation)}_data_access.json')
    archive.close()
    del visual, model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, details, history, state


def main():
    torch.set_num_threads(4)
    for name in ('results', 'weights', 'features'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    data = Data()
    folds = data.folds()
    configs = [{'rank': rank, 'blocks': 2, 'epochs': 3} for rank in (4, 8, 16)]
    configs += [{'rank': 8, 'blocks': blocks, 'epochs': 3} for blocks in (1, 4)]
    trials = []
    best = None
    for number, config in enumerate(configs, 1):
        by_scale = {scale: [] for scale in (.9, .95, 1.)}
        for fold_index, fold in enumerate(folds):
            metrics, _, history, _ = cv_trial(data, fold, config)
            for row in metrics:
                by_scale[row['scale']].append(row)
            print(f'LoRA CV {config} fold {fold_index+1}/3: '
                  f'{[(r["weighted_grounding"]) for r in metrics]}', flush=True)
        for scale, fold_scores in by_scale.items():
            value = sum(row['weighted_grounding']*row['queries'] for row in fold_scores)/sum(row['queries'] for row in fold_scores)
            trial = {**config, 'scale': scale, 'cv_grounding': value,
                     'fold_weighted_grounding': [row['weighted_grounding'] for row in fold_scores],
                     'fold_queries': [row['queries'] for row in fold_scores]}
            trials.append(trial)
            if best is None or value > best['cv_grounding']+1e-9:
                best = trial
        print(f'LoRA grouped CV {number}/{len(configs)}: best fit-side grounding {best["cv_grounding"]:.4f}', flush=True)
    write_json(OUT/'results/cv_trials.json', trials)
    frozen = {'selected': best, 'selection_used': 'three SIFT-scene-group folds within 33 original-train fit images',
              'fit_images': data.split['fit'], 'internal_dev_images': data.split['dev'],
              'official_validation_or_test_used': False,
              'previously_selected_config': {'rank': 8, 'blocks': 2, 'epochs': 3, 'scale': .95},
              'limitation': 'Earlier rounds consulted the same internal-dev split; fold selection itself uses only fit images.',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': {str(p.relative_to(BASE)): digest(p) for p in
                 (Path(__file__), BASE/'training_v1/code/lora_visual.py')}}
    write_json(OUT/'results/frozen_config.json', frozen)
    archive = TrainArchive()
    model, transform, state, history = train(data, data.fit_ids, best, archive)
    checkpoint = OUT/'weights/RemoteCLIP_visual_lora_cvselected.pt'
    torch.save({'state': state, 'config': frozen}, checkpoint)
    write_json(OUT/'results/full_fit_training_history.json', history)
    visual = lora_visual.encode_grounding(model, transform, data, archive, data.dev_ids)
    archive.audit(OUT/'results/dev_inference_data_access.json')
    archive.close()
    torch.save(visual, OUT/'features/lora_cvselected_dev_regions.pt')
    previous = torch.load(BASE/'training_v1/weights/trained_fusion_final.pt', weights_only=True)
    predictions = __import__('evaluate').predict(data, data.dev_ids, previous)
    predictions['grounding'] = chosen_boxes(data, data.dev_ids,
        lora_visual.combined_scores(data, data.dev_ids, visual), best['scale'])
    ranker = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
    inputs = retrieval.features(data)
    predictions['rankings'] = retrieval.predict(data, inputs, data.dev_ids,
        ranker['retrieval_model'], ranker['config']['retrieval'])
    result, details = measure(data.cache, data.split['dev'], predictions)
    write_json(OUT/'results/internal_dev_predictions.json', details)
    report = {'combined_with_v2_retrieval': result, 'target_score': .6,
              'target_met': result['score_out_of_1'] >= .6,
              'official_validation_or_test_used': False,
              'checkpoint_sha256': digest(checkpoint), 'config': frozen,
              'scored_at_utc': datetime.now(timezone.utc).isoformat(),
              'previous_best_same_internal_dev_score': .5259259259259259}
    write_json(OUT/'results/internal_dev_score.json', report)
    print('CV-selected LoRA with v2 retrieval:', result, flush=True)


if __name__ == '__main__':
    main()

