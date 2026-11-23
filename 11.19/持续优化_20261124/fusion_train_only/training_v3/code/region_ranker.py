"""Learn query-relative candidate quality from frozen visual representations."""
import itertools
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, digest, iou, write_json
from fusion_features import Features
from train import fit_ridge, ridge, grounding_metric
from data import encode_deltas, decode_deltas
from score_round import measure
import retrieval

OUT = BASE/'training_v3'


class RegionFeatures(Features):
    def __init__(self):
        super().__init__()
        encoded = torch.load(BASE/'training_v2/features/candidates.pt', weights_only=True)
        prompts = torch.load(BASE/'training_v2/features/query_features.pt', weights_only=True)
        for key, row in self.rows.items():
            scalar = row['features']
            centered = scalar - scalar.mean(0)
            standardized = centered / scalar.std(0, unbiased=False).clamp_min(.05)
            scalar = torch.cat([scalar, centered, standardized], 1)
            streams, conditioned = [], []
            for name in ('RemoteCLIP', 'GeoRSCLIP'):
                views = encoded[name][key]
                text = prompts[name]['ensemble'][row['query']]
                streams.extend([views[:, 0], views[:, 1]])
                conditioned.extend([views[:, 0]*text, views[:, 1]*text])
            visual = torch.cat(streams, 1)
            row['scalar_relative'] = scalar
            row['visual_rank'] = torch.cat([visual, scalar], 1)
            row['conditioned_rank'] = torch.cat([visual, *conditioned, scalar], 1)
            row['box_features'] = torch.cat([visual, scalar[:, 14:21]], 1)

    def x(self, key, kind):
        return self.rows[key][kind]


def utilities(features, ids, mode):
    assert set(ids) <= set(features.data.fit_ids), 'Supervision outside original fit split'
    result = {}
    for key in features.keys(ids):
        row = features.rows[key]
        truth = features.data.truths(row['image'], row['query'], ids)
        overlaps = torch.stack([iou(box, truth) for box in row['boxes']])
        values = overlaps if mode == 'iou' else .6*(overlaps >= .5).float()+.1*(overlaps >= .75).float()
        result[key] = values.mean(1)
    return result


def fit(features, ids, config):
    targets = utilities(features, ids, config['target'])
    xs, ys = [], []
    rng = random.Random(20261123)
    for key, y in targets.items():
        x = features.x(key, config['kind'])
        if config['objective'] == 'pairwise':
            pairs = [(i, j) for i, j in itertools.combinations(range(len(y)), 2)
                     if abs(float(y[i]-y[j])) > .03]
            rng.shuffle(pairs)
            pairs = pairs[:12]
            if not pairs:
                continue
            a, b = zip(*pairs)
            delta_x = x[list(a)]-x[list(b)]
            delta_y = y[list(a)]-y[list(b)]
            xs.extend([delta_x, -delta_x])
            ys.extend([delta_y, -delta_y])
        else:
            # Center within the query so image-specific score offsets cannot fit labels.
            xs.append(x-x.mean(0))
            ys.append(y-y.mean())
    return fit_ridge(torch.cat(xs), torch.cat(ys), config['alpha'])


def scores(features, ids, model, config):
    output = {}
    for key in features.keys(ids):
        value = features.rows[key]['base']
        if model is not None:
            value = value+config['amount']*ridge(model, features.x(key, config['kind']))
        output[key] = value
    return output


def box_regression(features, ids, alpha):
    assert set(ids) <= set(features.data.fit_ids)
    xs, ys = [], []
    for key in features.keys(ids):
        row = features.rows[key]
        truth = features.data.truths(row['image'], row['query'], ids)
        overlap = torch.stack([iou(box, truth) for box in row['boxes']])
        maximum, nearest = overlap.max(1)
        keep = maximum >= .35
        xs.append(row['box_features'][keep])
        ys.append(encode_deltas(row['boxes'][keep], truth[nearest[keep]]))
    return fit_ridge(torch.cat(xs), torch.cat(ys), alpha)


def boxes(features, ids, values, scale=1., regression=None, fraction=0.):
    output = {}
    for key in features.keys(ids):
        row = features.rows[key]
        image = features.data.records[row['image']]
        candidates = row['boxes']
        if regression is not None and fraction:
            candidates = decode_deltas(candidates, fraction*ridge(regression, row['box_features']),
                                       image['width'], image['height'])
        center = (candidates[:, :2]+candidates[:, 2:])/2
        half = (candidates[:, 2:]-candidates[:, :2])*scale/2
        candidates = torch.cat([center-half, center+half], 1)
        candidates[:, 0::2].clamp_(0, image['width'])
        candidates[:, 1::2].clamp_(0, image['height'])
        output[key] = candidates[int(values[key].argmax())].tolist()
    return output


def main():
    torch.set_num_threads(4)
    for name in ('results', 'weights'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    features = RegionFeatures()
    data = features.data
    folds = data.folds()
    baseline = {k: r['base'] for k, r in features.rows.items()}
    baseline_score = grounding_metric(data, data.fit_ids, boxes(features, data.fit_ids, baseline))['weighted_grounding']
    best = {'kind': 'base', 'amount': 0., 'scale': 1., 'cv_grounding': baseline_score}
    best_oof = baseline
    trials = []
    configs = list(itertools.product(('scalar_relative', 'visual_rank', 'conditioned_rank'),
                                    ('pointwise', 'pairwise'), ('iou', 'utility'), (.01, .1, 1., 10.)))
    for number, (kind, objective, target, alpha) in enumerate(configs, 1):
        config = {'kind': kind, 'objective': objective, 'target': target, 'alpha': alpha}
        oof = {}
        for val in folds:
            fit_ids = [i for i in data.fit_ids if i not in val]
            model = fit(features, fit_ids, config)
            for key in features.keys(val):
                oof[key] = ridge(model, features.x(key, kind))
        for amount, scale in itertools.product((.1, .25, .5, 1.), (.95, 1., 1.05)):
            combined = {k: baseline[k]+amount*v for k, v in oof.items()}
            score = grounding_metric(data, data.fit_ids, boxes(features, data.fit_ids, combined, scale))['weighted_grounding']
            trial = {**config, 'amount': amount, 'scale': scale, 'cv_grounding': score}
            trials.append(trial)
            if score > best['cv_grounding']+1e-9:
                best, best_oof = trial, combined
        if number % 8 == 0:
            print(f'Region ranking {number}/{len(configs)}: best fit-side grounding {best["cv_grounding"]:.4f}', flush=True)
    best_regression = {'alpha': 1., 'fraction': 0., 'cv_grounding': best['cv_grounding']}
    for alpha in (.1, 1., 10.):
        models = [box_regression(features, [i for i in data.fit_ids if i not in val], alpha) for val in folds]
        for fraction in (.25, .5, 1.):
            output = {}
            for val, model in zip(folds, models):
                output.update(boxes(features, val, best_oof, best['scale'], model, fraction))
            score = grounding_metric(data, data.fit_ids, output)['weighted_grounding']
            if score > best_regression['cv_grounding']+1e-9:
                best_regression = {'alpha': alpha, 'fraction': fraction, 'cv_grounding': score}
    write_json(OUT/'results/cv_trials.json', trials)
    frozen = {'grounding': best, 'regression': best_regression,
              'selection_used': 'three scene-group folds within the original 33 fit images',
              'fit_images': data.split['fit'], 'internal_dev_images': data.split['dev'],
              'fit_adapted_lora_features_in_cross_validation': False,
              'official_validation_or_test_used': False,
              'limitation': 'Iterative architecture development has consulted internal-dev scores.',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': digest(Path(__file__)),
              'inputs_sha256': {p.name: digest(p) for p in (BASE/'training_v2/features').iterdir() if p.is_file()}}
    write_json(OUT/'results/frozen_config.json', frozen)
    model = None if best['kind'] == 'base' else fit(features, data.fit_ids, best)
    regression = (box_regression(features, data.fit_ids, best_regression['alpha'])
                  if best_regression['fraction'] else None)
    checkpoint = OUT/'weights/region_ranker.pt'
    torch.save({'model': model, 'regression': regression, 'config': frozen}, checkpoint)
    retrieval_bundle = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
    predictions = {'grounding': boxes(features, data.dev_ids, scores(features, data.dev_ids, model, best),
                                     best['scale'], regression, best_regression['fraction']),
                   'rankings': retrieval.predict(data, retrieval.features(data), data.dev_ids,
                        retrieval_bundle['retrieval_model'], retrieval_bundle['config']['retrieval'])}
    result, details = measure(data.cache, data.split['dev'], predictions)
    write_json(OUT/'results/internal_dev_score.json', {
        **result, 'target_score': .6, 'target_met': result['score_out_of_1'] >= .6,
        'official_validation_or_test_used': False, 'config': frozen,
        'checkpoint_sha256': digest(checkpoint), 'scored_at_utc': datetime.now(timezone.utc).isoformat()})
    write_json(OUT/'results/internal_dev_predictions.json', details)
    print('Region ranker internal dev:', result, flush=True)


if __name__ == '__main__':
    main()
