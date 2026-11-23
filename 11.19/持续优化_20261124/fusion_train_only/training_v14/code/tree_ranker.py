"""Cross-validate nonlinear candidate ranking on original-train fit images."""
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v3/code'))
import region_ranker
from inputs import digest, write_json
import retrieval
from score_round import measure

OUT = BASE/'training_v14'
FEATURE_KIND = 'scalar_relative'


def fit_model(features, ids, config):
    targets = region_ranker.utilities(features, ids, config['target'])
    xs, ys, weights = [], [], []
    for key, target in targets.items():
        x = features.rows[key][FEATURE_KIND]
        xs.append((x-x.mean(0)).numpy())
        ys.append((target-target.mean()).numpy())
        weights.append(np.full(len(target), 1/max(len(target), 1), dtype=np.float32))
    x, y, sample_weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
    if config['family'] == 'extra_trees':
        model = ExtraTreesRegressor(n_estimators=256, max_depth=config['max_depth'],
                                    min_samples_leaf=config['min_samples_leaf'],
                                    max_features=.8, n_jobs=4, random_state=20261124)
    else:
        model = HistGradientBoostingRegressor(max_iter=120, learning_rate=.05,
                                               max_leaf_nodes=config['max_leaf_nodes'],
                                               min_samples_leaf=5, l2_regularization=1.,
                                               random_state=20261124)
    model.fit(x, y, sample_weight=sample_weight)
    return model


def tree_predictions(features, ids, model):
    output = {}
    for key in features.keys(ids):
        x = features.rows[key][FEATURE_KIND]
        centered = x-x.mean(0)
        output[key] = torch.as_tensor(model.predict(centered.numpy()), dtype=x.dtype)
    return output


def normalized(values):
    return (values-values.mean())/values.std(unbiased=False).clamp_min(1e-6)


def score_regions(features, ids, model, amount):
    predictions = tree_predictions(features, ids, model)
    output = {}
    for key in features.keys(ids):
        baseline = features.rows[key]['base']
        output[key] = normalized(baseline)+amount*normalized(predictions[key])
    return output


def baseline_oof(features, folds, config):
    output = {}
    for validation in folds:
        training = [image for image in features.data.fit_ids if image not in validation]
        model = region_ranker.fit(features, training, config)
        for key in features.keys(validation):
            row = features.rows[key]
            output[key] = row['base']+config['amount']*region_ranker.ridge(
                model, features.x(key, config['kind']))
    return output


def main():
    torch.set_num_threads(4)
    (OUT/'results').mkdir(parents=True, exist_ok=True)
    (OUT/'weights').mkdir(parents=True, exist_ok=True)
    features = region_ranker.RegionFeatures()
    data, folds = features.data, features.data.folds()
    previous = json.loads((BASE/'training_v3/results/frozen_config.json').read_text(encoding='utf-8'))['grounding']
    baseline_predictions = baseline_oof(features, folds, previous)
    baseline = region_ranker.grounding_metric(
        data, data.fit_ids,
        region_ranker.boxes(features, data.fit_ids, baseline_predictions, previous['scale']))
    baseline_score = baseline['weighted_grounding']

    configs = []
    for target in ('iou', 'utility'):
        for depth, leaf in itertools.product((3, 5, None), (2, 4)):
            configs.append({'family': 'extra_trees', 'target': target,
                            'max_depth': depth, 'min_samples_leaf': leaf})
        for leaves in (3, 7):
            configs.append({'family': 'hist_gradient', 'target': target,
                            'max_leaf_nodes': leaves})

    trials, best_tree = [], None
    best = {'method': 'existing_v3_ridge', 'cv_grounding': baseline_score}
    best_config = None
    for index, config in enumerate(configs, 1):
        oof, fold_ids = {}, []
        for fold_index, validation in enumerate(folds):
            training = [image for image in data.fit_ids if image not in validation]
            model = fit_model(features, training, config)
            oof.update(tree_predictions(features, validation, model))
            fold_ids.append(validation)
        for amount, scale in itertools.product((.25, .5, 1., 2., 4.), (.9, .95, 1., 1.05)):
            scores = {key: normalized(features.rows[key]['base'])+amount*normalized(value)
                      for key, value in oof.items()}
            fold_metrics = [region_ranker.grounding_metric(
                data, ids, region_ranker.boxes(features, ids, scores, scale)) for ids in fold_ids]
            queries = sum(row['queries'] for row in fold_metrics)
            score = sum(row['weighted_grounding']*row['queries'] for row in fold_metrics)/queries
            trial = {**config, 'amount': amount, 'scale': scale,
                     'cv_grounding': score,
                     'fold_scores': [row['weighted_grounding'] for row in fold_metrics],
                     'fold_queries': [row['queries'] for row in fold_metrics]}
            trials.append(trial)
            if best_tree is None or score > best_tree['cv_grounding']+1e-9:
                best_tree = trial
            if score > best['cv_grounding']+1e-9:
                best, best_config = trial, trial
        print(f"Tree ranker {index}/{len(configs)}: best grouped CV {best['cv_grounding']:.4f}", flush=True)

    frozen = {'selected': best, 'feature_kind': FEATURE_KIND,
              'best_tree_candidate': best_tree,
              'fit_only_baseline': {'method': 'training_v3 ridge', 'cv_grounding': baseline_score,
                                    'fold_queries': [21, 17, 16]},
              'selection_used': 'three SIFT-scene-group folds within 33 original-train fit images',
              'fit_images': data.split['fit'], 'internal_dev_images': data.split['dev'],
              'official_validation_or_test_used': False,
              'development_score_recomputed': False,
              'limitation': 'Earlier rounds consulted the same internal-dev split.',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': digest(Path(__file__))}
    write_json(OUT/'results/cv_trials.json', trials)
    write_json(OUT/'results/frozen_config.json', frozen)

    if best_config is None:
        retained = json.loads((BASE/'optimization_results/current_best.json').read_text(encoding='utf-8'))
        write_json(OUT/'results/selection.json', {
            'decision': 'retain_current_best', 'fit_only_baseline': baseline_score,
            'best_tree_candidate': best_tree, 'development_score_recomputed': False,
            'retained_method': retained['method'],
            'retained_development_score_out_of_1': retained['score_out_of_1'],
            'retained_score_source': 'optimization_results/current_best.json',
            'official_validation_or_test_used': False})
        print('No tree candidate beat grouped fit-only baseline; retained current best.', flush=True)
        return

    model = fit_model(features, data.fit_ids, best_config)
    checkpoint = OUT/'weights/tree_candidate_ranker.joblib'
    joblib.dump({'model': model, 'config': best_config, 'feature_kind': FEATURE_KIND}, checkpoint)
    scores = score_regions(features, data.dev_ids, model, best_config['amount'])
    boxes = region_ranker.boxes(features, data.dev_ids, scores, best_config['scale'])
    ranker = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
    rankings = retrieval.predict(data, retrieval.features(data), data.dev_ids,
                                 ranker['retrieval_model'], ranker['config']['retrieval'])
    result, details = measure(data.cache, data.split['dev'],
                              {'grounding': boxes, 'rankings': rankings})
    frozen['development_score_recomputed'] = True
    frozen['checkpoint_sha256'] = digest(checkpoint)
    write_json(OUT/'results/frozen_config.json', frozen)
    write_json(OUT/'results/internal_dev_predictions.json', details)
    write_json(OUT/'results/internal_dev_score.json', {
        'combined_with_v2_retrieval': result, 'target_score': .6,
        'target_met': result['score_out_of_1'] >= .6,
        'official_validation_or_test_used': False,
        'checkpoint_sha256': digest(checkpoint), 'config': frozen,
        'scored_at_utc': datetime.now(timezone.utc).isoformat(),
        'previous_best_same_internal_dev_score': json.loads(
            (BASE/'optimization_results/current_best.json').read_text(encoding='utf-8'))['score_out_of_1']})
    print('Tree ranker with retained v2 retrieval:', result, flush=True)


if __name__ == '__main__':
    main()
