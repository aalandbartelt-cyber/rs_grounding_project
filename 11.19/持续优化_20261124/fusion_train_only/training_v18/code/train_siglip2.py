"""Select SigLIP 2 evidence with scene-grouped original-train fit folds."""
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, digest, iou, write_json
from train import chosen_boxes, fit_ridge, grounding_metric, retrieval_metric, ridge
from lora_visual import combined_scores
from score_round import measure
import retrieval

OUT = BASE/'training_v18'
MODES = ('chinese', 'concise', 'ensemble')


def centered(values):
    return (values-values.mean())/values.std(unbiased=False).clamp_min(1e-6)


class Evidence:
    def __init__(self, data):
        self.data = data
        pack = torch.load(OUT/'features/siglip2_train_features.pt', weights_only=True)
        assert pack['texts'] == data.texts
        self.pack = pack
        self.rows = {}
        self.retrieval = retrieval.features(data)
        for key, row in data.grounding.items():
            visual = pack['regions'][key]
            similarities = torch.stack([visual@pack['text_features'][mode][row['query']]
                                        for mode in MODES], dim=2).flatten(1)
            scalar = torch.cat([row['scalar'], similarities], dim=1)
            scalar = torch.cat([scalar, scalar-scalar.mean(0)], dim=1)
            text = pack['text_features']['ensemble'][row['query']]
            interaction = (visual.mean(1)*text).float()
            self.rows[key] = {'scalar': scalar,
                              'interaction': torch.cat([scalar, interaction], dim=1),
                              'zero_shot': {mode: (visual@pack['text_features'][mode][row['query']]).mean(1)
                                            for mode in MODES}}
        for query in range(len(data.texts)):
            for image, record in enumerate(data.records):
                gallery = pack['gallery'][image]
                extras = []
                for mode in MODES:
                    text = pack['text_features'][mode][query]
                    global_score = gallery['whole']@text
                    regional = gallery['regions']@text
                    if len(regional):
                        extra = [global_score, regional[:, 0].max(), regional[:, 1].max(),
                                 regional.mean(), regional.mean(1).topk(min(len(regional), 2)).values.mean()]
                    else:
                        extra = [global_score]*5
                    extras.extend(extra)
                key = f'{query}:{image}'
                self.retrieval[key] = torch.cat([self.retrieval[key], torch.stack(extras)])


def fit_grounding(evidence, ids, config):
    data = evidence.data
    assert set(ids) <= set(data.fit_ids), 'SigLIP 2 grounding supervision outside fit split'
    xs, ys = [], []
    for key in data.keys(ids):
        row = data.grounding[key]
        truth = data.truths(row['image'], row['query'], ids)
        overlap = torch.stack([iou(box, truth) for box in row['boxes']])
        target = (overlap if config['target'] == 'iou' else
                  .6*(overlap >= .5).float()+.1*(overlap >= .75).float()).mean(1)
        x = evidence.rows[key][config['features']]
        xs.append(x-x.mean(0))
        ys.append(target-target.mean())
    return fit_ridge(torch.cat(xs), torch.cat(ys), config['alpha'])


def predict_grounding(evidence, ids, model, config, lora):
    output = {}
    for key in evidence.data.keys(ids):
        row = evidence.rows[key]
        if config['method'] == 'zero_shot':
            value = row['zero_shot'][config['mode']]
        else:
            x = row[config['features']]
            value = ridge(model, x-x.mean(0))
        output[key] = centered(lora[key])+config['amount']*centered(value)
    return output


def select_grounding(evidence, folds, lora):
    data = evidence.data
    baseline = json.loads((BASE/'training_v3/results/frozen_config.json').read_text(encoding='utf-8'))['grounding']['cv_grounding']
    best = {'method': 'retained_lora', 'cv_grounding': baseline}
    trials = []
    configs = [{'method': 'zero_shot', 'mode': mode} for mode in MODES]
    configs += [dict(method='ridge', features=features, target=target, alpha=alpha)
                for features, target, alpha in itertools.product(
                    ('scalar', 'interaction'), ('iou', 'utility'), (.01, .1, 1., 10.))]
    for number, config in enumerate(configs, 1):
        models = [None if config['method'] == 'zero_shot' else fit_grounding(
            evidence, [image for image in data.fit_ids if image not in validation], config)
                  for validation in folds]
        for amount, scale in itertools.product((.25, .5, 1., 2., 4., 16.), (.9, .95, 1., 1.05)):
            trial = {**config, 'amount': amount, 'scale': scale}
            metrics = []
            for fold, model in zip(folds, models):
                scores = predict_grounding(evidence, fold, model, trial, lora)
                boxes = chosen_boxes(data, fold, scores, scale)
                metrics.append(grounding_metric(data, fold, boxes))
            trial['cv_grounding'] = sum(row['weighted_grounding']*row['queries'] for row in metrics)/54
            trial['fold_scores'] = [row['weighted_grounding'] for row in metrics]
            trial['fold_queries'] = [row['queries'] for row in metrics]
            trials.append(trial)
            if trial['cv_grounding'] > best['cv_grounding']+1e-9:
                best = trial
        print(f'SigLIP2 grounding {number}/{len(configs)}: best fit CV {best["cv_grounding"]:.4f}', flush=True)
    return best, trials


def select_retrieval(evidence, folds):
    data = evidence.data
    previous = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')['config']['retrieval']
    old_features = retrieval.features(data)
    old_metrics = []
    for validation in folds:
        training = [image for image in data.fit_ids if image not in validation]
        model = retrieval.fit(data, old_features, training, previous)
        rankings = retrieval.predict(data, old_features, validation, model, previous)
        old_metrics.append(retrieval_metric(data, validation, rankings))
    baseline = sum(row['weighted_retrieval']*row['queries'] for row in old_metrics)/54
    best = {'method': 'retained_v2', 'cv_retrieval': baseline}
    trials = []
    configs = [{'family': 'ridge', 'alpha': alpha} for alpha in (.01, .1, 1., 10., 100.)]
    configs += [{'family': 'extra', 'depth': depth} for depth in (2, 4)]
    for config in configs:
        models = [retrieval.fit(data, evidence.retrieval,
                    [image for image in data.fit_ids if image not in validation], config) for validation in folds]
        for amount in (.03, .1, .3, 1., 3.):
            trial = {**config, 'amount': amount}
            metrics = [retrieval_metric(data, fold, retrieval.predict(
                data, evidence.retrieval, fold, model, trial)) for fold, model in zip(folds, models)]
            trial['cv_retrieval'] = sum(row['weighted_retrieval']*row['queries'] for row in metrics)/54
            trial['fold_scores'] = [row['weighted_retrieval'] for row in metrics]
            trial['fold_queries'] = [row['queries'] for row in metrics]
            trials.append(trial)
            if trial['cv_retrieval'] > best['cv_retrieval']+1e-9:
                best = trial
        print(f'SigLIP2 retrieval {config}: best fit CV {best["cv_retrieval"]:.4f}', flush=True)
    return best, trials, baseline


def main():
    torch.set_num_threads(4)
    (OUT/'weights').mkdir(parents=True, exist_ok=True)
    data = Data()
    evidence = Evidence(data)
    folds = data.folds()
    oof = torch.load(BASE/'training_v15/features/remoteclip_lora_oof_scores.pt', weights_only=True)
    assert oof['fold_image_ids'] == folds
    lora = {key: score for fold in oof['fold_scores'].values() for key, score in fold.items()}
    grounding, grounding_trials = select_grounding(evidence, folds, lora)
    rank_config, rank_trials, retrieval_baseline = select_retrieval(evidence, folds)
    frozen = {'grounding': grounding, 'retrieval': rank_config,
              'retrieval_fit_only_baseline': retrieval_baseline,
              'fit_images': data.split['fit'], 'internal_dev_images': data.split['dev'],
              'selection_used': 'three SIFT-scene-group folds within original-train fit images',
              'official_validation_or_test_used': False,
              'limitation': 'Previous rounds consulted the same internal-dev split.',
              'feature_sha256': digest(OUT/'features/siglip2_train_features.pt'),
              'code_sha256': digest(Path(__file__)),
              'frozen_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(OUT/'results/cv_trials.json', {'grounding': grounding_trials, 'retrieval': rank_trials})
    write_json(OUT/'results/frozen_config.json', frozen)
    retained = json.loads((BASE/'optimization_results/current_best.json').read_text(encoding='utf-8'))
    new_grounding = grounding['method'] != 'retained_lora'
    new_retrieval = rank_config.get('method') != 'retained_v2'
    if not (new_grounding or new_retrieval):
        write_json(OUT/'results/selection.json', {
            'decision': 'retain_current_best', 'development_score_recomputed': False,
            'retained_development_score_out_of_1': retained['score_out_of_1'],
            'best_grounding_candidate': max(grounding_trials, key=lambda row: row['cv_grounding']),
            'best_retrieval_candidate': max(rank_trials, key=lambda row: row['cv_retrieval']),
            'official_validation_or_test_used': False})
        print('SigLIP2 did not beat fit-only baselines; retained current best.', flush=True)
        return
    g_model = fit_grounding(evidence, data.fit_ids, grounding) if new_grounding and grounding['method'] == 'ridge' else None
    r_model = retrieval.fit(data, evidence.retrieval, data.fit_ids, rank_config) if new_retrieval else None
    checkpoint = OUT/'weights/siglip2_fusion_rankers.joblib'
    joblib.dump({'grounding_model': g_model, 'retrieval_model': r_model, 'config': frozen}, checkpoint)
    visual = torch.load(BASE/'training_v1/features/RemoteCLIP_lora_dev_regions.pt', weights_only=True)
    lora = combined_scores(data, data.dev_ids, visual)
    boxes = chosen_boxes(data, data.dev_ids, lora, .95)
    if new_grounding:
        scores = predict_grounding(evidence, data.dev_ids, g_model, grounding, lora)
        boxes = chosen_boxes(data, data.dev_ids, scores, grounding['scale'])
    if new_retrieval:
        rankings = retrieval.predict(data, evidence.retrieval, data.dev_ids, r_model, rank_config)
    else:
        previous = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
        rankings = retrieval.predict(data, retrieval.features(data), data.dev_ids,
                                     previous['retrieval_model'], previous['config']['retrieval'])
    result, details = measure(data.cache, data.split['dev'], {'grounding': boxes, 'rankings': rankings})
    write_json(OUT/'results/internal_dev_predictions.json', details)
    write_json(OUT/'results/internal_dev_score.json', {
        **result, 'target_score': .6, 'target_met': result['score_out_of_1'] >= .6,
        'official_validation_or_test_used': False, 'config': frozen,
        'checkpoint_sha256': digest(checkpoint), 'scored_at_utc': datetime.now(timezone.utc).isoformat(),
        'previous_best_same_internal_dev_score': retained['score_out_of_1']})
    print('SigLIP2 fusion internal-development score:', result, flush=True)


if __name__ == '__main__':
    main()
