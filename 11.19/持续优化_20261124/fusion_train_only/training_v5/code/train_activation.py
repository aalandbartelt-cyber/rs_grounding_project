"""Fit an activation ablation while retaining the same data and scoring protocol."""
import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v4/code'))
import train_fusion as fusion
from inputs import digest, write_json
from train import grounding_metric, retrieval_metric, chosen_boxes
from lora_visual import combined_scores
from score_round import measure
import retrieval

OUT = BASE/'training_v5'


def prepare():
    features = fusion.ComplementaryFeatures()
    quick = torch.load(OUT/'features/GeoRSCLIP_QuickGELU.pt', weights_only=True)
    assert quick['signature']['visual_and_text_activation_matched']
    for key, row in features.rows.items():
        views = quick['grounding'][key]
        values = [views[:, j]@quick['text'][mode][row['query']]
                  for mode in ('literal', 'concise', 'attribute', 'ensemble', 'centered') for j in (0, 1)]
        values = torch.stack(values, 1)
        row['complementary'] = torch.cat([row['complementary'], values, values-values.mean(0)], 1)
    inputs = fusion.retrieval_features(features.data)
    for q in range(len(features.data.texts)):
        for i, row in enumerate(features.data.records):
            image = quick['gallery'][i]
            n = len(row['boxes'])
            values = []
            for mode in ('literal', 'concise', 'attribute', 'ensemble', 'centered'):
                text = quick['text'][mode][q]
                whole = image['whole']@text
                values.append(whole)
                for j in (0, 1):
                    local = image['regions'][:, j]@text
                    if n:
                        prior = local+.1*row['scores'].clamp_min(1e-6).log()
                        values.extend([local.max(), local.mean(), local.topk(min(2, n)).values.mean(), prior.max()])
                    else:
                        values.extend([whole]*4)
            inputs[f'{q}:{i}'] = torch.cat([inputs[f'{q}:{i}'], torch.stack(values)])
    return features, inputs


def main():
    torch.set_num_threads(4)
    for name in ('results', 'weights'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    features, values = prepare()
    data = features.data
    previous = torch.load(BASE/'training_v1/weights/trained_fusion_final.pt', weights_only=True)
    old_config = previous['config']['grounding']
    folds = data.folds()
    old_models = [fusion.train_old(data, [i for i in data.fit_ids if i not in val], old_config) for val in folds]
    oof = {}
    for val, model in zip(folds, old_models):
        oof.update(fusion.candidate_scores(features, val, model, old_config))
    best = {'policy': 'original', 'amount': 0., 'scale': old_config['scale']}
    best['cv_grounding'] = grounding_metric(data, data.fit_ids, fusion.select_boxes(features, data.fit_ids, oof, best))['weighted_grounding']
    trials = []
    for policy, target, alpha in itertools.product(('original', 'expanded'), ('iou', 'utility'), (.01, .1, 1., 10.)):
        config = {'policy': policy, 'target': target, 'alpha': alpha}
        oof = {}
        for val, old_model in zip(folds, old_models):
            model = fusion.train_candidate(features, [i for i in data.fit_ids if i not in val], config)
            oof.update(fusion.candidate_scores(features, val, old_model, old_config, model))
        for amount, scale in itertools.product((.03, .1, .3, 1.), (.9, .95, 1., 1.05)):
            trial = {**config, 'amount': amount, 'scale': scale}
            trial['cv_grounding'] = grounding_metric(data, data.fit_ids, fusion.select_boxes(features, data.fit_ids, oof, trial))['weighted_grounding']
            trials.append(trial)
            if trial['cv_grounding'] > best['cv_grounding']+1e-9:
                best = trial
        print(f'QuickGELU candidate {config}: best fit-side grounding {best["cv_grounding"]:.4f}', flush=True)
    r_best, r_trials = None, []
    for mode in ('exact', 'semantic'):
        configs = [{'family': 'ridge', 'alpha': a} for a in (.01, .1, 1., 10.)]
        configs += [{'family': 'extra', 'depth': d} for d in (2, 4)]
        for spec in configs:
            config = {**spec, 'target': mode}
            models = [fusion.train_retrieval(data, values, [i for i in data.fit_ids if i not in val], config) for val in folds]
            for amount in (.03, .1, .3, 1., 3.):
                trial = {**config, 'amount': amount}
                metrics = [retrieval_metric(data, val, retrieval.predict(data, values, val, model, trial)) for val, model in zip(folds, models)]
                trial['cv_retrieval'] = sum(r['weighted_retrieval']*r['queries'] for r in metrics)/sum(r['queries'] for r in metrics)
                r_trials.append(trial)
                if r_best is None or trial['cv_retrieval'] > r_best['cv_retrieval']+1e-9:
                    r_best = trial
        print(f'QuickGELU retrieval/{mode}: best fit-side recall {r_best["cv_retrieval"]:.4f}', flush=True)
    write_json(OUT/'results/cv_trials.json', {'grounding': trials, 'retrieval': r_trials})
    frozen = {'grounding': best, 'retrieval': r_best, 'old_grounding': old_config,
              'selection_used': 'scene-group three-fold CV within 33 fit images',
              'fit_images': data.split['fit'], 'internal_dev_images': data.split['dev'],
              'official_validation_or_test_used': False,
              'limitation': 'Model development has consulted internal-dev scores.',
              'activation_feature_sha256': digest(OUT/'features/GeoRSCLIP_QuickGELU.pt'),
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': {str(p.relative_to(BASE)): digest(p) for p in (Path(__file__), Path(fusion.__file__))}}
    write_json(OUT/'results/frozen_config.json', frozen)
    model = fusion.train_candidate(features, data.fit_ids, best) if best['amount'] else None
    old_model = fusion.train_old(data, data.fit_ids, old_config)
    r_model = fusion.train_retrieval(data, values, data.fit_ids, r_best)
    checkpoint = OUT/'weights/activation_fusion.joblib'
    joblib.dump({'grounding': model, 'old_grounding': old_model, 'retrieval': r_model, 'config': frozen}, checkpoint)
    predictions = {'grounding': fusion.select_boxes(features, data.dev_ids, fusion.candidate_scores(features, data.dev_ids, old_model, old_config, model), best),
                   'rankings': retrieval.predict(data, values, data.dev_ids, r_model, r_best)}
    result, details = measure(data.cache, data.split['dev'], predictions)
    write_json(OUT/'results/internal_dev_predictions.json', details)
    visual = torch.load(BASE/'training_v1/features/RemoteCLIP_lora_dev_regions.pt', weights_only=True)
    predictions['grounding'] = chosen_boxes(data, data.dev_ids, combined_scores(data, data.dev_ids, visual), .95)
    retained_local, local_details = measure(data.cache, data.split['dev'], predictions)
    write_json(OUT/'results/retrieval_only_predictions.json', local_details)
    write_json(OUT/'results/internal_dev_score.json', {
        'combined': result, 'new_retrieval_only': retained_local,
        'target_score': .6, 'official_validation_or_test_used': False,
        'checkpoint_sha256': digest(checkpoint), 'config': frozen,
        'scored_at_utc': datetime.now(timezone.utc).isoformat()})
    print('QuickGELU combined internal dev:', result, flush=True)
    print('QuickGELU retrieval with retained LoRA:', retained_local, flush=True)


if __name__ == '__main__':
    main()
