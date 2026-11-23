"""Learn query-wise image ordering from fit-only positive/negative pairs."""
import itertools
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
import retrieval
from inputs import Data, digest, write_json
from train import retrieval_metric
from score_round import measure
from lora_visual import combined_scores
from train import chosen_boxes

OUT = BASE/'training_v8'


def compatible(query, annotated):
    def attrs(text):
        engines = next((n for n, pattern in ((4, r'四(?:台|发|引擎)'),
                         (3, r'三(?:台|发|引擎)'), (2, r'两台|双发|双引擎|双发动机'))
                       if re.search(pattern, text)), None)
        paint = 'blue_white' if '蓝白' in text else 'white' if '白色' in text else None
        engine_position = 'above' if '翼上' in text else 'below' if any(t in text for t in ('翼下','翼吊式','机翼下')) else None
        state = 'flight' if any(t in text for t in ('飞行','空中')) else 'ground' if any(t in text for t in ('静止','停放')) else None
        return engines, paint, engine_position, state
    a, b = attrs(query), attrs(annotated)
    return all(x is None or y is None or x == y for x, y in zip(a, b))


def matrix(data, raw, query, ids, representation):
    values = torch.stack([raw[f'{query}:{image}'] for image in ids]).float()
    if representation == 'raw':
        return values
    order = torch.argsort(torch.argsort(values, dim=0, stable=True), dim=0, stable=True).float()
    if len(ids) > 1:
        order /= len(ids)-1
    else:
        order.zero_()
    if representation == 'rank':
        return order
    if representation == 'raw_rank':
        return torch.cat([values, order], 1)
    raise ValueError(representation)


def labels(data, query, ids, target):
    output = []
    text = data.texts[query]
    for image in ids:
        descriptions = data.records[image]['texts']
        if target == 'exact':
            output.append(float(text in descriptions))
        else:
            output.append(float(any(compatible(text, value) for value in descriptions)))
    return torch.tensor(output)


def fit(data, raw, ids, config):
    assert set(ids) <= set(data.fit_ids), 'Retrieval training outside original fit split'
    queries = sorted({data.texts.index(text) for image in ids for text in data.records[image]['texts']})
    matrices, targets, pair_deltas, pair_weights = [], [], [], []
    for query in queries:
        x = matrix(data, raw, query, ids, config['representation'])
        y = labels(data, query, ids, config['target'])
        matrices.append(x)
        targets.append(y)
        if config['objective'] == 'pairwise':
            positive = (y > 0).nonzero().flatten().tolist()
            negative = (y == 0).nonzero().flatten().tolist()
            pairs = [(p, n) for p in positive for n in negative]
            if pairs:
                delta = x[[p for p, _ in pairs]]-x[[n for _, n in pairs]]
                pair_deltas.append(delta.numpy())
                pair_weights.extend([1./len(pairs)]*len(pairs))
    x = torch.cat(matrices).numpy().astype(np.float64)
    y = torch.cat(targets).numpy().astype(np.float64)
    mean, std = x.mean(0), x.std(0).clip(min=.05)
    z = (x-mean)/std
    alpha = config['alpha']
    if config['objective'] == 'pairwise':
        deltas = np.concatenate(pair_deltas).astype(np.float64)
        delta_z = deltas/std
        weight = np.asarray(pair_weights)

        def objective(coef):
            margin = delta_z@coef
            probability = 1/(1+np.exp(np.clip(margin, -40, 40)))
            loss = np.logaddexp(0., -margin)
            value = np.dot(weight, loss)+.5*alpha*np.dot(coef, coef)
            gradient = -(delta_z.T@(weight*probability))+alpha*coef
            return value, gradient

        result = minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B',
                          options={'maxiter': 120, 'ftol': 1e-9})
        coef = result.x
    else:
        positive = max(float(y.sum()), 1.)
        negative = max(float(len(y)-y.sum()), 1.)
        sample_weight = np.where(y > 0, .5/positive, .5/negative)

        def objective(coef):
            score = z@coef
            probability = 1/(1+np.exp(np.clip(score, -40, 40)))
            loss = np.logaddexp(0., score)-y*score
            value = np.dot(sample_weight, loss)+.5*alpha*np.dot(coef, coef)
            gradient = z.T@(sample_weight*(probability-y))+alpha*coef
            return value, gradient

        result = minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B',
                          options={'maxiter': 120, 'ftol': 1e-9})
        coef = result.x
    return {'mean': torch.from_numpy(mean).float(), 'std': torch.from_numpy(std).float(),
            'weight': torch.from_numpy(coef).float(), 'optimizer_success': bool(result.success),
            'optimizer_iterations': int(result.nit)}


def predict(data, raw, ids, model, config):
    output = []
    for query in range(len(data.texts)):
        x = matrix(data, raw, query, ids, config['representation'])
        z = (x-model['mean'])/model['std']
        learned = z@model['weight']
        baseline = torch.tensor([data.retrieval[f'{query}:{image}']['base_score'] for image in ids])
        scores = baseline+config['amount']*learned
        output.append([ids[i] for i in scores.argsort(descending=True, stable=True)[:5].tolist()])
    return output


def main():
    torch.set_num_threads(4)
    for name in ('results', 'weights'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    data = Data()
    raw = retrieval.features(data)
    folds = data.folds()
    trials, best = [], None
    configs = itertools.product(('pairwise','pointwise'), ('exact','semantic'),
                                ('raw','rank','raw_rank'), (.03,.3,3.,30.))
    configs = list(configs)
    for number, (objective, target, representation, alpha) in enumerate(configs, 1):
        models = []
        for val in folds:
            train_ids = [image for image in data.fit_ids if image not in val]
            config = {'objective': objective, 'target': target,
                      'representation': representation, 'alpha': alpha}
            model = fit(data, raw, train_ids, config)
            models.append(model)
        for amount in (.03,.1,.3,1.,3.):
            metrics = [retrieval_metric(data, val, predict(data, raw, val, model,
                       {**config, 'amount': amount})) for val, model in zip(folds, models)]
            weighted = sum(r['weighted_retrieval']*r['queries'] for r in metrics)/sum(r['queries'] for r in metrics)
            trial = {**config, 'amount': amount, 'cv_retrieval': weighted,
                     'fold_queries': [r['queries'] for r in metrics],
                     'fold_weighted_retrieval': [r['weighted_retrieval'] for r in metrics]}
            trials.append(trial)
            if best is None or weighted > best['cv_retrieval']+1e-9:
                best = trial
        if number % 6 == 0 or number == len(configs):
            print(f'Pairwise gallery ranking {number}/{len(configs)}: best fit-side weighted retrieval {best["cv_retrieval"]:.4f}', flush=True)
    write_json(OUT/'results/cv_trials.json', trials)
    frozen = {'selected': best, 'fit_images': data.split['fit'],
              'internal_dev_images': data.split['dev'],
              'selection_used': 'three SIFT-scene-group folds within the original 33 fit images',
              'official_validation_or_test_used': False,
              'query_labels_used': 'exact text matches or generic attribute compatibility, selected by fit-only CV',
              'gallery_relative_features': 'per-feature ranks among only the current prediction gallery',
              'limitation': 'Repeated internal-dev iterations informed research direction and retention.',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': digest(Path(__file__)),
              'input_features_sha256': {str(p.relative_to(BASE)): digest(p) for p in (
                  BASE/'training_v1/features/qwen_all_training_gallery.json',
                  BASE/'training_v2/features/query_features.pt',
                  BASE/'training_v2/features/candidates.pt')}}
    write_json(OUT/'results/frozen_config.json', frozen)
    model = fit(data, raw, data.fit_ids, best)
    checkpoint = OUT/'weights/pairwise_retriever.joblib'
    joblib.dump({'model': model, 'config': best, 'protocol': frozen}, checkpoint)
    rankings = predict(data, raw, data.dev_ids, model, best)
    metric = retrieval_metric(data, data.dev_ids, rankings)
    previous = torch.load(BASE/'training_v1/weights/trained_fusion_final.pt', weights_only=True)
    old = __import__('evaluate').predict(data, data.dev_ids, previous)
    old['rankings'] = rankings
    visual = torch.load(BASE/'training_v1/features/RemoteCLIP_lora_dev_regions.pt', weights_only=True)
    old['grounding'] = chosen_boxes(data, data.dev_ids, combined_scores(data, data.dev_ids, visual), .95)
    combined, details = measure(data.cache, data.split['dev'], old)
    write_json(OUT/'results/internal_dev_predictions.json', details)
    report = {'retrieval': metric, 'combined_with_retained_lora_localization': combined,
              'target_score': .6, 'target_met': combined['score_out_of_1'] >= .6,
              'official_validation_or_test_used': False,
              'checkpoint_sha256': digest(checkpoint), 'config': frozen,
              'scored_at_utc': datetime.now(timezone.utc).isoformat(),
              'previous_best_same_internal_dev_score': .5259259259259259}
    write_json(OUT/'results/internal_dev_score.json', report)
    print('Pairwise retrieval:', metric, flush=True)
    print('Combined with retained LoRA:', combined, flush=True)


if __name__ == '__main__':
    main()
