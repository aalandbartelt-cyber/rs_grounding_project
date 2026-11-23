"""Fit compact geometry correction and prediction-only region consensus."""
import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch
from sklearn.linear_model import HuberRegressor

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v4/code'))
import train_fusion as fusion
from inputs import digest, iou, write_json
from data import encode_deltas, decode_deltas
from train import fit_ridge, ridge, grounding_metric
from lora_visual import combined_scores
from score_round import measure
import retrieval

OUT = BASE/'training_v7'


def scaled(row, scale):
    boxes = row['boxes']
    center = (boxes[:, :2]+boxes[:, 2:])/2
    half = (boxes[:, 2:]-boxes[:, :2])*scale/2
    return torch.cat([center-half, center+half], 1)


def geometry(row):
    scalar = row['features']
    # Geometry, detector confidence, source type and agreement are prediction inputs.
    return torch.cat([scalar[:, 2:3], scalar[:, 8:14], scalar[:, 16:21]], 1)


def fit(features, ids, family, scale):
    data = features.data
    assert set(ids) <= set(data.fit_ids), 'Box correction outside original fit images'
    xs, ys, sources = [], [], []
    seen = set()
    for key in features.keys(ids):
        row = features.rows[key]
        boxes = scaled(row, scale)
        truth = data.records[row['image']]['truth']
        overlaps = torch.stack([iou(box, truth) for box in boxes])
        maximum, nearest = overlaps.max(1)
        for index in (maximum >= .4).nonzero().flatten().tolist():
            signature = (row['image'], tuple(boxes[index].tolist()))
            if signature in seen:
                continue
            seen.add(signature)
            xs.append(geometry(row)[index])
            ys.append(encode_deltas(boxes[index:index+1], truth[nearest[index]:nearest[index]+1])[0])
            sources.append(int(row['features'][index, 20]))
    x, y = torch.stack(xs), torch.stack(ys)
    if family == 'median':
        source = torch.tensor(sources)
        return {'median': y.median(0).values,
                'by_source': {value: y[source == value].median(0).values for value in (0, 1) if (source == value).any()}}
    if family == 'ridge':
        return fit_ridge(x, y, 10.)
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(.05)
    normalized = (x-mean)/std
    models = []
    for column in range(4):
        model = HuberRegressor(alpha=10., max_iter=500)
        model.fit(normalized.numpy(), y[:, column].numpy())
        models.append(model)
    return {'mean': mean, 'std': std, 'models': models}


def correction(model, row, family):
    x = geometry(row)
    if family == 'median':
        values = [model['by_source'].get(int(source), model['median']) for source in row['features'][:, 20]]
        return torch.stack(values)
    if family == 'ridge':
        return ridge(model, x)
    x = ((x-model['mean'])/model['std']).numpy()
    return torch.stack([torch.from_numpy(m.predict(x)).float() for m in model['models']], 1)


def predict(features, ids, selections, config, model=None):
    output = {}
    for key in features.keys(ids):
        row = features.rows[key]
        image = features.data.records[row['image']]
        boxes = scaled(row, config['scale'])
        if model is not None and config['correction_amount']:
            deltas = config['correction_amount']*correction(model, row, config['family'])
            boxes = decode_deltas(boxes, deltas, image['width'], image['height'])
        box = boxes[selections[key]]
        if config['consensus_amount']:
            overlaps = iou(box, boxes)
            weights = overlaps.pow(4)*(overlaps >= config['consensus_iou'])
            merged = (boxes*weights[:, None]).sum(0)/weights.sum().clamp_min(1e-6)
            box = (1-config['consensus_amount'])*box+config['consensus_amount']*merged
        box[0::2].clamp_(0, image['width'])
        box[1::2].clamp_(0, image['height'])
        output[key] = box.tolist()
    return output


def main():
    torch.set_num_threads(4)
    for name in ('results', 'weights'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    features = fusion.ComplementaryFeatures()
    data = features.data
    previous = torch.load(BASE/'training_v1/weights/trained_fusion_final.pt', weights_only=True)
    old_config = previous['config']['grounding']
    folds = data.folds()
    selections = {}
    for val in folds:
        model = fusion.train_old(data, [i for i in data.fit_ids if i not in val], old_config)
        values = fusion.candidate_scores(features, val, model, old_config)
        for key in features.keys(val):
            selections[key] = int(values[key][0][:features.rows[key]['original_count']].argmax())
    best = {'family': 'none', 'scale': old_config['scale'], 'correction_amount': 0.,
            'consensus_amount': 0., 'consensus_iou': .5}
    best['cv_grounding'] = grounding_metric(data, data.fit_ids, predict(features, data.fit_ids, selections, best))['weighted_grounding']
    trials = []
    for scale, family in itertools.product((.9, .95, 1., 1.05), ('none', 'median', 'ridge', 'huber')):
        models = [None if family == 'none' else fit(features, [i for i in data.fit_ids if i not in val], family, scale) for val in folds]
        for amount, merge, threshold in itertools.product((0.,) if family == 'none' else (.25, .5, 1.), (0., .5, 1.), (.5, .7)):
            config = {'family': family, 'scale': scale, 'correction_amount': amount,
                      'consensus_amount': merge, 'consensus_iou': threshold}
            output = {}
            for val, model in zip(folds, models):
                output.update(predict(features, val, selections, config, model))
            score = grounding_metric(data, data.fit_ids, output)['weighted_grounding']
            trial = {**config, 'cv_grounding': score}
            trials.append(trial)
            if score > best['cv_grounding']+1e-9:
                best = trial
        print(f'Geometry correction {scale}/{family}: best fit-side grounding {best["cv_grounding"]:.4f}', flush=True)
    write_json(OUT/'results/cv_trials.json', trials)
    frozen = {'selected': best, 'fit_images': data.split['fit'], 'internal_dev_images': data.split['dev'],
              'selection_used': 'three scene-group folds within 33 fit images',
              'official_validation_or_test_used': False,
              'limitation': 'Development has consulted internal-dev scores; final localization retains the existing 33-fit-image LoRA selector.',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': {str(p.relative_to(BASE)): digest(p) for p in (Path(__file__), Path(fusion.__file__))}}
    write_json(OUT/'results/frozen_config.json', frozen)
    model = None if best['family'] == 'none' else fit(features, data.fit_ids, best['family'], best['scale'])
    checkpoint = OUT/'weights/geometry_correction.joblib'
    joblib.dump({'model': model, 'config': frozen}, checkpoint)
    visual = torch.load(BASE/'training_v1/features/RemoteCLIP_lora_dev_regions.pt', weights_only=True)
    scores = combined_scores(data, data.dev_ids, visual)
    selected = {key: int(value.argmax()) for key, value in scores.items()}
    retrieval_bundle = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
    predictions = {'grounding': predict(features, data.dev_ids, selected, best, model),
                   'rankings': retrieval.predict(data, retrieval.features(data), data.dev_ids,
                       retrieval_bundle['retrieval_model'], retrieval_bundle['config']['retrieval'])}
    result, details = measure(data.cache, data.split['dev'], predictions)
    write_json(OUT/'results/internal_dev_score.json', {**result, 'target_score': .6,
        'target_met': result['score_out_of_1'] >= .6, 'official_validation_or_test_used': False,
        'config': frozen, 'checkpoint_sha256': digest(checkpoint), 'scored_at_utc': datetime.now(timezone.utc).isoformat()})
    write_json(OUT/'results/internal_dev_predictions.json', details)
    print('Geometry refinement internal dev:', result, flush=True)


if __name__ == '__main__':
    main()
