"""Query-conditioned DINOv2/CLIP ranking with scene-grouped fit-only selection."""
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch
from torch.nn.functional import normalize

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v15/code'))
import dino_ranker
from inputs import Data, digest, write_json
from train import chosen_boxes, grounding_metric
from lora_visual import combined_scores
from score_round import measure
import retrieval

OUT = BASE/'training_v16'


def text_rows(data, ids):
    return torch.cat([normalize(data.text_features[name], dim=-1)
                      for name in ('RemoteCLIP', 'GeoRSCLIP')], dim=1)/math.sqrt(2)


def representations(data, dino, ids, config, stats=None):
    texts = text_rows(data, ids)
    if stats is None:
        queries = sorted({data.grounding[key]['query'] for key in data.keys(ids)})
        mean = texts[queries].mean(0)
        scalar = torch.cat([data.grounding[key]['scalar'] for key in data.keys(ids)])
        stats = {'text_mean': mean, 'scalar_std': scalar.std(0, unbiased=False).clamp_min(.05)}
    if config['query_mode'] == 'centered':
        texts = normalize(texts-stats['text_mean'], dim=-1)
    visual_rows, query_rows, scalar_rows, counts = [], [], [], []
    for key in data.keys(ids):
        row = data.grounding[key]
        visual = dino[key][:, :1536] if config['visual_kind'] == 'cls' else dino[key]
        visual = normalize(visual-visual.mean(0), dim=-1)
        scalar = row['scalar']
        scalar = (scalar-scalar.mean(0))/stats['scalar_std']/math.sqrt(scalar.shape[1])
        visual_rows.append(visual)
        query_rows.append(texts[row['query']].expand(len(visual), -1))
        scalar_rows.append(scalar)
        counts.append((key, len(visual)))
    return {'visual': torch.cat(visual_rows), 'query': torch.cat(query_rows),
            'scalar': torch.cat(scalar_rows)}, counts, stats


def kernel(left, right, config):
    appearance = left['visual']@right['visual'].T
    semantic = left['query']@right['query'].T
    # The product kernel learns how appearance changes meaning across query descriptions.
    return appearance*(.25+.75*semantic)+.1*(left['scalar']@right['scalar'].T)


def fit_model(data, dino, ids, config):
    assert set(ids) <= set(data.fit_ids), 'Conditional ranking supervision outside fit split'
    x, counts, stats = representations(data, dino, ids, config)
    truth = dino_ranker.targets(data, ids, config['target'])
    ys, sample_weights = [], []
    for key, count in counts:
        y = truth[key]
        ys.append(y-y.mean())
        sample_weights.append(torch.full((count,), 1/count))
    y = torch.cat(ys)
    weights = torch.cat(sample_weights)
    matrix = kernel(x, x, config)
    matrix = matrix+config['alpha']*torch.diag(1/weights)
    dual = torch.linalg.solve(matrix, y)
    return {'training_x': x, 'dual': dual, 'stats': stats, 'config': config}


def predictions(data, dino, ids, model):
    x, counts, _ = representations(data, dino, ids, model['config'], model['stats'])
    scores = kernel(x, model['training_x'], model['config'])@model['dual']
    output, start = {}, 0
    for key, count in counts:
        output[key] = scores[start:start+count]
        start += count
    return output


def main():
    torch.set_num_threads(4)
    (OUT/'results').mkdir(parents=True, exist_ok=True)
    (OUT/'weights').mkdir(parents=True, exist_ok=True)
    data = Data()
    dino = torch.load(BASE/'training_v15/features/dinov2_region_features.pt', weights_only=True)['features']
    lora_bundle = torch.load(BASE/'training_v15/features/remoteclip_lora_oof_scores.pt', weights_only=True)
    folds = data.folds()
    assert folds == lora_bundle['fold_image_ids']
    lora = lora_bundle['fold_scores']
    baseline = json.loads((BASE/'training_v3/results/frozen_config.json').read_text(encoding='utf-8'))['grounding']['cv_grounding']
    configs = [dict(visual_kind=kind, query_mode=mode, target=target, alpha=alpha)
               for kind, mode, target, alpha in itertools.product(
                   ('cls', 'all'), ('raw', 'centered'), ('iou', 'utility'), (.01, .1, 1., 10.))]
    trials, best_candidate = [], None
    selected = {'method': 'existing_v3_ridge', 'cv_grounding': baseline}
    for index, config in enumerate(configs, 1):
        fold_scores = {}
        for fold_index, validation in enumerate(folds):
            ids = [image for image in data.fit_ids if image not in validation]
            model = fit_model(data, dino, ids, config)
            fold_scores[fold_index] = predictions(data, dino, validation, model)
        for amount, scale in itertools.product((.25, .5, 1., 2., 4.), (.9, .95, 1., 1.05)):
            per_fold = []
            for fold_index, validation in enumerate(folds):
                scores = dino_ranker.combine_scores(data, validation, lora[fold_index],
                                                    fold_scores[fold_index], amount)
                boxes = chosen_boxes(data, validation, scores, scale)
                per_fold.append(grounding_metric(data, validation, boxes))
            score = sum(row['weighted_grounding']*row['queries'] for row in per_fold)/sum(row['queries'] for row in per_fold)
            trial = {**config, 'amount': amount, 'scale': scale, 'cv_grounding': score,
                     'fold_scores': [row['weighted_grounding'] for row in per_fold],
                     'fold_queries': [row['queries'] for row in per_fold]}
            trials.append(trial)
            if best_candidate is None or score > best_candidate['cv_grounding']+1e-9:
                best_candidate = trial
            if score > selected['cv_grounding']+1e-9:
                selected = trial
        print(f"Query-conditioned DINOv2 {index}/{len(configs)}: "
              f"best grouped CV {selected['cv_grounding']:.4f}", flush=True)

    frozen = {'selected': selected, 'best_candidate': best_candidate,
              'fit_only_baseline': baseline,
              'selection_used': 'three SIFT-scene-group folds within 33 original-train fit images',
              'fit_images': data.split['fit'], 'internal_dev_images': data.split['dev'],
              'official_validation_or_test_used': False,
              'limitation': 'Previous rounds consulted the same internal-dev split.',
              'source_feature_sha256': digest(BASE/'training_v15/features/dinov2_region_features.pt'),
              'oof_lora_sha256': digest(BASE/'training_v15/features/remoteclip_lora_oof_scores.pt'),
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': digest(Path(__file__))}
    write_json(OUT/'results/cv_trials.json', trials)
    write_json(OUT/'results/frozen_config.json', frozen)
    retained = json.loads((BASE/'optimization_results/current_best.json').read_text(encoding='utf-8'))
    if selected.get('method') == 'existing_v3_ridge':
        write_json(OUT/'results/selection.json', {
            'decision': 'retain_current_best', 'fit_only_baseline': baseline,
            'best_candidate': best_candidate, 'development_score_recomputed': False,
            'retained_development_score_out_of_1': retained['score_out_of_1'],
            'retained_score_source': 'optimization_results/current_best.json',
            'official_validation_or_test_used': False})
        print('No conditional kernel candidate beat grouped fit-only baseline.', flush=True)
        return

    model = fit_model(data, dino, data.fit_ids, selected)
    checkpoint = OUT/'weights/query_conditioned_DINOv2_ranker.pt'
    torch.save({'model': model, 'config': frozen}, checkpoint)
    visual = torch.load(BASE/'training_v11/features/lora_cvselected_dev_regions.pt', weights_only=True)
    lora_scores = combined_scores(data, data.dev_ids, visual)
    learned = predictions(data, dino, data.dev_ids, model)
    scores = dino_ranker.combine_scores(data, data.dev_ids, lora_scores, learned, selected['amount'])
    boxes = chosen_boxes(data, data.dev_ids, scores, selected['scale'])
    ranker = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
    rankings = retrieval.predict(data, retrieval.features(data), data.dev_ids,
                                 ranker['retrieval_model'], ranker['config']['retrieval'])
    result, details = measure(data.cache, data.split['dev'], {'grounding': boxes, 'rankings': rankings})
    write_json(OUT/'results/internal_dev_predictions.json', details)
    write_json(OUT/'results/internal_dev_score.json', {
        'combined_with_v2_retrieval': result, 'target_score': .6,
        'target_met': result['score_out_of_1'] >= .6,
        'official_validation_or_test_used': False,
        'checkpoint_sha256': digest(checkpoint), 'config': frozen,
        'scored_at_utc': datetime.now(timezone.utc).isoformat(),
        'previous_best_same_internal_dev_score': retained['score_out_of_1']})
    print('Query-conditioned DINOv2 with v2 retrieval:', result, flush=True)


if __name__ == '__main__':
    main()
