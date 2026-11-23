"""Fit query/image grouped LoRA losses against hard detector negatives."""
import gc
import itertools
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import open_clip
import torch
from PIL import ImageEnhance, ImageOps
from torch.nn import functional as F
from torch.nn.utils import parametrize

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, TrainArchive, crop, digest, iou, write_json
sys.path.insert(0, str(BASE/'training_v1/code'))
import lora_visual
from train import chosen_boxes, grounding_metric
from score_round import measure
import retrieval

OUT = BASE/'training_v13'
EPOCHS = 3


def load_model():
    model, _, transform = open_clip.create_model_and_transforms('ViT-L-14', pretrained=None)
    checkpoint = BASE/'weights/RemoteCLIP/RemoteCLIP-ViT-L-14.pt'
    state = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
    state = state.get('state_dict', state.get('model', state))
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    del state
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in model.visual.transformer.resblocks[-2:]:
        for module, name in ((block.attn, 'in_proj_weight'), (block.attn.out_proj, 'weight'),
                             (block.mlp.c_fc, 'weight'), (block.mlp.c_proj, 'weight')):
            parametrize.register_parametrization(module, name,
                lora_visual.LowRank(getattr(module, name).shape, rank=8))
    return model.cuda(), transform


def groups(data, ids):
    assert set(ids) <= set(data.fit_ids), 'Pairwise adapter supervision outside the fit split'
    output = []
    for key in data.keys(ids):
        row = data.grounding[key]
        truth = data.truths(row['image'], row['query'], ids)
        overlap = torch.stack([iou(box, truth).max() for box in row['boxes']])
        negatives = (overlap < .3).nonzero().flatten()
        if len(negatives):
            order = row['base_score'][negatives].argsort(descending=True, stable=True)
            negatives = negatives[order[:4]]
        output.append({'image': row['image'], 'query': row['query'], 'positive': truth,
                       'negative': row['boxes'][negatives]})
    return output


def train(data, ids, pair_weight, archive):
    torch.manual_seed(20261123)
    torch.cuda.manual_seed_all(20261123)
    model, transform = load_model()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=1e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    rng = random.Random(20261123)
    examples = groups(data, ids)
    images = {i: archive.image(data.records[i]['label']) for i in ids}
    texts = data.text_features['RemoteCLIP'].cuda()
    history = []
    for epoch in range(1, EPOCHS+1):
        rng.shuffle(examples)
        model.train()
        total = 0.
        for example in examples:
            boxes = torch.cat([example['positive'], example['negative']])
            labels = torch.cat([torch.ones(len(example['positive'])),
                                torch.zeros(len(example['negative']))]).cuda()
            views = []
            for box in boxes:
                view = crop(images[example['image']], box, rng.choice((1., 1.25, 1.5)))
                if rng.random() < .5:
                    view = ImageOps.mirror(view)
                view = view.rotate(rng.choice((0, 90, 180, 270)), expand=True)
                view = ImageEnhance.Brightness(view).enhance(rng.uniform(.9, 1.1))
                views.append(transform(view))
            chunks = []
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(views), 4):
                with torch.autocast('cuda', dtype=torch.float16):
                    chunks.append(F.normalize(model.encode_image(torch.stack(views[start:start+4]).cuda()).float(), dim=-1))
            similarity = torch.cat(chunks)@texts[example['query']]
            logits = 20*(similarity-.25)
            binary = F.binary_cross_entropy_with_logits(logits, labels)
            n_positive = len(example['positive'])
            n_negative = len(example['negative'])
            if n_negative:
                pairwise = F.softplus(logits[n_positive:][None, :]-logits[:n_positive, None]+.4).mean()
                loss = (1-pair_weight)*binary+pair_weight*pairwise
            else:
                loss = binary
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        history.append({'epoch': epoch, 'mean_group_loss': total/len(examples)})
    state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
             if parameter.requires_grad}
    return model, transform, state, history


def main():
    torch.set_num_threads(4)
    for name in ('results', 'weights', 'features'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    data = Data()
    folds = data.folds()
    weights = (.25, .5, .75, 1.)
    trials, best = [], {'method': 'existing_v11_adapter', 'cv_grounding': .45925925925925926,
                       'scale': .95, 'pair_weight': 0.}
    for pair_weight in weights:
        by_scale = {scale: [] for scale in (.9, .95, 1.)}
        for fold_index, validation in enumerate(folds):
            data = Data()
            training = [image for image in data.fit_ids if image not in validation]
            archive = TrainArchive()
            model, transform, state, history = train(data, training, pair_weight, archive)
            visual = lora_visual.encode_grounding(model, transform, data, archive, validation)
            scores = lora_visual.combined_scores(data, validation, visual)
            for scale in by_scale:
                prediction = chosen_boxes(data, validation, scores, scale)
                by_scale[scale].append(grounding_metric(data, validation, prediction))
            archive.audit(OUT/'results'/f'pair{pair_weight}_fold{fold_index}_data_access.json')
            archive.close()
            del model, visual, state
            gc.collect()
            torch.cuda.empty_cache()
            print(f'Pairwise LoRA weight {pair_weight} fold {fold_index+1}/3: '
                  f'{[by_scale[s][-1]["weighted_grounding"] for s in by_scale]}', flush=True)
        for scale, rows in by_scale.items():
            score = sum(row['weighted_grounding']*row['queries'] for row in rows)/sum(row['queries'] for row in rows)
            trial = {'method': 'pairwise_lora', 'pair_weight': pair_weight,
                     'scale': scale, 'epochs': EPOCHS,
                     'cv_grounding': score, 'fold_scores': [row['weighted_grounding'] for row in rows],
                     'fold_queries': [row['queries'] for row in rows]}
            trials.append(trial)
            if score > best['cv_grounding']+1e-9:
                best = trial
        print(f'Pairwise loss {pair_weight}: best grouped CV {best["cv_grounding"]:.4f}', flush=True)
    write_json(OUT/'results/cv_trials.json', trials)
    frozen = {'selected': best,
              'selection_used': 'three SIFT-scene-group folds within 33 original-train fit images',
              'fit_images': data.split['fit'], 'internal_dev_images': data.split['dev'],
              'official_validation_or_test_used': False,
              'loss': 'binary image-text alignment mixed with per-query hard-negative pairwise ranking',
              'hard_negatives': 'top four frozen CLIP-scored predicted boxes with IoU below 0.3; used for fit targets only',
              'limitation': 'Earlier rounds consulted the same internal-dev split.',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': digest(Path(__file__))}
    write_json(OUT/'results/frozen_config.json', frozen)
    if best['method'] == 'existing_v11_adapter':
        retained = json.loads((BASE/'optimization_results/current_best.json').read_text(encoding='utf-8'))
        report = {'decision': 'retain_current_best',
                  'retained_method': retained['method'],
                  'fit_only_cv_reference': best,
                  'pairwise_candidates_beat_cv_reference': False,
                  'retained_development_score_out_of_1': retained['score_out_of_1'],
                  'retained_score_source': 'optimization_results/current_best.json',
                  'retained_score_recomputed_in_this_round': False,
                  'target_score': .6,
                  'target_met': retained['score_out_of_1'] >= .6,
                  'official_validation_or_test_used': False,
                  'conclusion': 'No pairwise loss weight beat the existing adapter on grouped fit folds; retained the existing best model without another internal-dev selection round.'}
        write_json(OUT/'results/selection.json', report)
        print('No replacement selected by fit-only grouped CV; kept v11/v2 retained baseline.', flush=True)
        return
    archive = TrainArchive()
    model, transform, state, history = train(data, data.fit_ids, best['pair_weight'], archive)
    checkpoint = OUT/'weights/RemoteCLIP_visual_lora_pairwise.pt'
    torch.save({'state': state, 'config': frozen}, checkpoint)
    write_json(OUT/'results/full_fit_training_history.json', history)
    visual = lora_visual.encode_grounding(model, transform, data, archive, data.dev_ids)
    torch.save(visual, OUT/'features/pairwise_lora_dev_regions.pt')
    archive.audit(OUT/'results/full_fit_and_dev_data_access.json')
    archive.close()
    predictions = __import__('evaluate').predict(data, data.dev_ids,
        torch.load(BASE/'training_v1/weights/trained_fusion_final.pt', weights_only=True))
    predictions['grounding'] = chosen_boxes(data, data.dev_ids,
        lora_visual.combined_scores(data, data.dev_ids, visual), best['scale'])
    ranker = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
    inputs = retrieval.features(data)
    predictions['rankings'] = retrieval.predict(data, inputs, data.dev_ids,
        ranker['retrieval_model'], ranker['config']['retrieval'])
    result, details = measure(data.cache, data.split['dev'], predictions)
    write_json(OUT/'results/internal_dev_predictions.json', details)
    write_json(OUT/'results/internal_dev_score.json', {
        'combined_with_v2_retrieval': result, 'target_score': .6,
        'target_met': result['score_out_of_1'] >= .6,
        'official_validation_or_test_used': False,
        'checkpoint_sha256': digest(checkpoint), 'config': frozen,
        'scored_at_utc': datetime.now(timezone.utc).isoformat(),
        'previous_best_same_internal_dev_score': .5259259259259259})
    print('Pairwise LoRA with v2 retrieval:', result, flush=True)


if __name__ == '__main__':
    main()
