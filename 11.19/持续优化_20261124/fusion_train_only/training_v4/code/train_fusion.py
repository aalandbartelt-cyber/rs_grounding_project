"""Calibrate complementary Chinese/English evidence with scene-group fit folds."""
import itertools
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch
from sklearn.ensemble import ExtraTreesRegressor
from torch.nn.functional import normalize

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import digest, iou, write_json
from fusion_features import Features
from train import fit_ridge, ridge, supervision, target, grounding_metric, retrieval_metric
from score_round import measure
import retrieval

OUT = BASE/'training_v4'


def attributes(text):
    count = next((n for n, pattern in ((4, r'四(?:台|发|引擎)'), (3, r'三(?:台|发|引擎)'),
                                      (2, r'两台|双发|双引擎|双发动机')) if re.search(pattern, text)), None)
    color = 'blue_white' if '蓝白' in text else 'white' if '白色' in text else None
    location = 'above' if '翼上' in text else 'below' if any(t in text for t in ('翼下', '翼吊式', '机翼下')) else None
    flying = 'flight' if '飞行' in text or '空中' in text else 'ground' if '静止' in text or '停放' in text else None
    return count, color, location, flying


def compatible(query, annotated):
    """A generic query leaves an attribute unspecified; an explicit mismatch is negative."""
    q, a = attributes(query), attributes(annotated)
    return all(left is None or right is None or left == right for left, right in zip(q, a))


def chinese_evidence(data, query, image):
    row = data.records[image]
    visual = normalize(data.cache['initial'], dim=-1)
    text = normalize(data.cache['text'], dim=-1)[query]
    n, start = len(row['boxes']), row['start']
    if not n:
        return (visual[start]@text).repeat(1), visual[start]@text
    return ((visual[start+1:start+1+n]+visual[start+1+n:start+1+2*n])/2)@text, visual[start]@text


class ComplementaryFeatures(Features):
    def __init__(self):
        super().__init__()
        self.data.cache['initial'] = normalize(self.data.cache['initial'], dim=-1)
        self.data.cache['text'] = normalize(self.data.cache['text'], dim=-1)
        for key, row in self.rows.items():
            old = self.data.grounding[key]
            overlap = torch.stack([iou(box, old['boxes']) for box in row['boxes']])
            agreement, nearest = overlap.max(1)
            row['nearest_old'], row['nearest_agreement'] = nearest, agreement
            original = self.data.records[row['image']]
            chinese, _ = chinese_evidence(self.data, row['query'], row['image'])
            dino_overlap = torch.stack([iou(box, original['boxes']) for box in row['boxes']])
            maximum, closest = dino_overlap.max(1)
            chinese = chinese[closest]*maximum
            chinese = torch.stack([chinese, chinese-chinese.mean(), maximum], 1)
            x = torch.cat([row['features'], old['scalar'][nearest], chinese], 1)
            row['complementary'] = torch.cat([x, x-x.mean(0)], 1)
            row['old_base'] = old['base_score'][nearest]+.1*agreement.clamp_min(1e-6).log()
            row['original_count'] = len(old['boxes'])


def train_candidate(features, ids, config):
    assert set(ids) <= set(features.data.fit_ids)
    xs, ys = [], []
    for key in features.keys(ids):
        row = features.rows[key]
        count = row['original_count'] if config['policy'] == 'original' else len(row['boxes'])
        truth = features.data.truths(row['image'], row['query'], ids)
        overlap = torch.stack([iou(box, truth) for box in row['boxes'][:count]])
        y = (overlap if config['target'] == 'iou' else
             .6*(overlap >= .5).float()+.1*(overlap >= .75).float()).mean(1)
        xs.append(row['complementary'][:count])
        ys.append(y)
    return fit_ridge(torch.cat(xs), torch.cat(ys), config['alpha'])


def train_old(data, ids, config):
    x, y = supervision(data, ids, config['kind'])
    return fit_ridge(x, target(y, config['target']), config['alpha'])


def candidate_scores(features, ids, old_model, old_config, model=None):
    output = {}
    for key in features.keys(ids):
        row = features.rows[key]
        old_row = features.data.grounding[key]
        learned = ridge(old_model, old_row[old_config['kind']])[row['nearest_old']]
        value = row['old_base']+old_config['fraction']*learned*row['nearest_agreement']
        new = ridge(model, row['complementary']) if model is not None else torch.zeros_like(value)
        output[key] = (value, new)
    return output


def select_boxes(features, ids, scores, config):
    output = {}
    for key in features.keys(ids):
        row = features.rows[key]
        image = features.data.records[row['image']]
        count = row['original_count'] if config['policy'] == 'original' else len(row['boxes'])
        score = scores[key][0]+config['amount']*scores[key][1]
        box = row['boxes'][int(score[:count].argmax())]
        center = (box[:2]+box[2:])/2
        half = (box[2:]-box[:2])*config['scale']/2
        box = torch.cat([center-half, center+half])
        box[0::2].clamp_(0, image['width'])
        box[1::2].clamp_(0, image['height'])
        output[key] = box.tolist()
    return output


def retrieval_features(data):
    values = retrieval.features(data)
    for q in range(len(data.texts)):
        for i, row in enumerate(data.records):
            regional, whole = chinese_evidence(data, q, i)
            n = len(row['boxes'])
            prior = regional+.1*row['scores'].clamp_min(1e-6).log() if n else regional
            extra = torch.stack([whole, regional.max(), regional.mean(), regional.topk(min(2, len(regional))).values.mean(), prior.max()])
            values[f'{q}:{i}'] = torch.cat([values[f'{q}:{i}'], extra])
    return values


def train_retrieval(data, values, ids, config):
    assert set(ids) <= set(data.fit_ids)
    queries = sorted({data.texts.index(t) for i in ids for t in data.records[i]['texts']})
    x = torch.stack([values[f'{q}:{i}'] for q in queries for i in ids])
    y = torch.tensor([sum(t == data.texts[q] if config['target'] == 'exact' else compatible(data.texts[q], t)
                         for t in data.records[i]['texts']) for q in queries for i in ids], dtype=torch.float32)
    if config['family'] == 'ridge':
        return fit_ridge(x, y, config['alpha'])
    model = ExtraTreesRegressor(n_estimators=120, max_depth=config['depth'], min_samples_leaf=3,
                                n_jobs=4, random_state=20261123)
    model.fit(x.numpy(), y.numpy())
    return model


def main():
    torch.set_num_threads(4)
    for name in ('results', 'weights'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    features = ComplementaryFeatures()
    data = features.data
    previous = torch.load(BASE/'training_v1/weights/trained_fusion_final.pt', weights_only=True)
    old_config = previous['config']['grounding']
    folds = data.folds()
    old_models = [train_old(data, [i for i in data.fit_ids if i not in val], old_config) for val in folds]
    base_oof = {}
    for val, model in zip(folds, old_models):
        base_oof.update(candidate_scores(features, val, model, old_config))
    initial = {'policy': 'original', 'amount': 0., 'scale': old_config['scale']}
    initial['cv_grounding'] = grounding_metric(data, data.fit_ids, select_boxes(features, data.fit_ids, base_oof, initial))['weighted_grounding']
    best, trials = initial, []
    for policy, target_mode, alpha in itertools.product(('original', 'expanded'), ('iou', 'utility'), (.01, .1, 1., 10.)):
        config = {'policy': policy, 'target': target_mode, 'alpha': alpha}
        oof = {}
        for val, old_model in zip(folds, old_models):
            model = train_candidate(features, [i for i in data.fit_ids if i not in val], config)
            oof.update(candidate_scores(features, val, old_model, old_config, model))
        for amount, scale in itertools.product((.03, .1, .3, 1.), (.9, .95, 1., 1.05)):
            trial = {**config, 'amount': amount, 'scale': scale}
            trial['cv_grounding'] = grounding_metric(data, data.fit_ids, select_boxes(features, data.fit_ids, oof, trial))['weighted_grounding']
            trials.append(trial)
            if trial['cv_grounding'] > best['cv_grounding']+1e-9:
                best = trial
        print(f'Complementary candidate {config}: best fit-side grounding {best["cv_grounding"]:.4f}', flush=True)
    write_json(OUT/'results/candidate_cv_trials.json', trials)
    values, r_trials, r_best = retrieval_features(data), [], None
    for mode in ('exact', 'semantic'):
        configs = [{'family': 'ridge', 'alpha': a} for a in (.01, .1, 1., 10.)]
        configs += [{'family': 'extra', 'depth': d} for d in (2, 4)]
        for spec in configs:
            config = {**spec, 'target': mode}
            models = [train_retrieval(data, values, [i for i in data.fit_ids if i not in val], config) for val in folds]
            for amount in (.03, .1, .3, 1., 3.):
                trial = {**config, 'amount': amount}
                metrics = [retrieval_metric(data, val, retrieval.predict(data, values, val, model, trial)) for val, model in zip(folds, models)]
                trial['cv_retrieval'] = sum(r['weighted_retrieval']*r['queries'] for r in metrics)/sum(r['queries'] for r in metrics)
                r_trials.append(trial)
                if r_best is None or trial['cv_retrieval'] > r_best['cv_retrieval']+1e-9:
                    r_best = trial
        print(f'Chinese+English retrieval/{mode}: best fit-side recall {r_best["cv_retrieval"]:.4f}', flush=True)
    write_json(OUT/'results/retrieval_cv_trials.json', r_trials)
    frozen = {'grounding': best, 'retrieval': r_best, 'old_grounding': old_config,
              'fit_images': data.split['fit'], 'internal_dev_images': data.split['dev'],
              'selection_used': 'three scene-group folds within original 33 fit images only',
              'official_validation_or_test_used': False,
              'limitation': 'Architecture iteration and retention have consulted internal-dev scores.',
              'chinese_features': 'Frozen ChineseCLIP regional features transferred by predicted-box IoU',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': digest(Path(__file__))}
    write_json(OUT/'results/frozen_config.json', frozen)
    model = train_candidate(features, data.fit_ids, best) if best['amount'] else None
    old_model = train_old(data, data.fit_ids, old_config)
    r_model = train_retrieval(data, values, data.fit_ids, r_best)
    checkpoint = OUT/'weights/complementary_fusion.joblib'
    joblib.dump({'grounding': model, 'old_grounding': old_model, 'retrieval': r_model, 'config': frozen}, checkpoint)
    new_boxes = select_boxes(features, data.dev_ids, candidate_scores(features, data.dev_ids, old_model, old_config, model), best)
    v2 = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
    old_rankings = retrieval.predict(data, retrieval.features(data), data.dev_ids, v2['retrieval_model'], v2['config']['retrieval'])
    predictions = {'grounding': new_boxes, 'rankings': old_rankings}
    local, _ = measure(data.cache, data.split['dev'], predictions)
    predictions['rankings'] = retrieval.predict(data, values, data.dev_ids, r_model, r_best)
    combined, details = measure(data.cache, data.split['dev'], predictions)
    from lora_visual import combined_scores
    from train import chosen_boxes
    visual = torch.load(BASE/'training_v1/features/RemoteCLIP_lora_dev_regions.pt', weights_only=True)
    predictions['grounding'] = chosen_boxes(data, data.dev_ids, combined_scores(data, data.dev_ids, visual), .95)
    retrieval_only, retrieval_details = measure(data.cache, data.split['dev'], predictions)
    write_json(OUT/'results/internal_dev_predictions.json', details)
    write_json(OUT/'results/retrieval_only_predictions.json', retrieval_details)
    write_json(OUT/'results/internal_dev_score.json', {
        'combined': combined, 'new_localization_only': local, 'new_retrieval_only': retrieval_only,
        'target_score': .6, 'official_validation_or_test_used': False,
        'checkpoint_sha256': digest(checkpoint), 'config': frozen,
        'scored_at_utc': datetime.now(timezone.utc).isoformat()})
    print('Complementary fusion internal dev:', combined, flush=True)
    print('Chinese+English retrieval with retained LoRA localization:', retrieval_only, flush=True)


if __name__ == '__main__':
    main()
