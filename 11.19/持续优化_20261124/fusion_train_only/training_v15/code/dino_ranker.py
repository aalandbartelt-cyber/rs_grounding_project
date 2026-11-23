"""Select DINOv2 and RemoteCLIP candidate scoring on grouped fit-only folds."""
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v11/code'))
import cv_lora
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, TrainArchive, digest, iou, write_json
sys.path.insert(0, str(BASE/'training_v1/code'))
import lora_visual
from train import chosen_boxes, fit_ridge, grounding_metric, ridge
from score_round import measure
import retrieval

OUT = BASE/'training_v15'
FEATURE_PATH = OUT/'features/dinov2_region_features.pt'


def feature_for(data, dino, key, kind):
    visual = dino[key]
    scalar = data.grounding[key]['scalar']
    if kind == 'dino_cls_scalar':
        visual = visual[:, :1536]
    elif kind == 'dino_patch_scalar':
        visual = visual[:, 1536:]
    return torch.cat([visual, scalar], dim=1)


def targets(data, ids, mode):
    assert set(ids) <= set(data.fit_ids), 'DINO ranking supervision outside fit split'
    values = {}
    for key in data.keys(ids):
        row = data.grounding[key]
        truth = data.truths(row['image'], row['query'], ids)
        overlap = torch.stack([iou(box, truth) for box in row['boxes']])
        if mode == 'iou':
            values[key] = overlap.mean(1)
        else:
            values[key] = (.6*(overlap >= .5).float()+.1*(overlap >= .75).float()).mean(1)
    return values


def fit_model(data, dino, ids, config):
    target_rows = targets(data, ids, config['target'])
    xs, ys = [], []
    for key, target in target_rows.items():
        x = feature_for(data, dino, key, config['feature_kind'])
        xs.append(x-x.mean(0))
        ys.append(target-target.mean())
    return fit_ridge(torch.cat(xs), torch.cat(ys), config['alpha'])


def predict_regions(data, dino, ids, model, config):
    predictions = {}
    for key in data.keys(ids):
        x = feature_for(data, dino, key, config['feature_kind'])
        predictions[key] = ridge(model, x-x.mean(0))
    return predictions


def normalize(values):
    return (values-values.mean())/values.std(unbiased=False).clamp_min(1e-6)


def combine_scores(data, ids, lora_scores, dino_scores, amount):
    return {key: normalize(lora_scores[key])+amount*normalize(dino_scores[key])
            for key in data.keys(ids)}


def lora_oof(data, folds, config):
    result, audits = {}, []
    for fold_index, validation in enumerate(folds):
        training = [image for image in data.fit_ids if image not in validation]
        archive = TrainArchive()
        model, transform, state, history = cv_lora.train(data, training, config, archive)
        visual = lora_visual.encode_grounding(model, transform, data, archive, validation)
        result[fold_index] = lora_visual.combined_scores(data, validation, visual)
        audit_path = OUT/'results'/f'lora_fold_{fold_index}_data_access.json'
        archive.audit(audit_path)
        audits.append(str(audit_path.relative_to(BASE)))
        archive.close()
        del model, visual, state, history
        torch.cuda.empty_cache()
        print(f'RemoteCLIP fit-fold baseline {fold_index+1}/{len(folds)} ready', flush=True)
    torch.save({'fold_scores': result, 'adapter_config': config,
                'fold_image_ids': folds,
                'code_sha256': {'cv_lora.py': digest(Path(cv_lora.__file__)),
                                'lora_visual.py': digest(Path(lora_visual.__file__))}},
               OUT/'features/remoteclip_lora_oof_scores.pt')
    return result, audits


def main():
    torch.set_num_threads(4)
    (OUT/'results').mkdir(parents=True, exist_ok=True)
    (OUT/'weights').mkdir(parents=True, exist_ok=True)
    data = Data()
    bundle = torch.load(FEATURE_PATH, weights_only=True)
    dino = bundle['features']
    folds = data.folds()
    adapter_config = json.loads((BASE/'training_v11/results/frozen_config.json').read_text(encoding='utf-8'))['selected']
    lora_scores, audits = lora_oof(data, folds, adapter_config)

    # The promotion threshold is the strongest previously reported fit-only localization CV.
    v3 = json.loads((BASE/'training_v3/results/frozen_config.json').read_text(encoding='utf-8'))
    baseline = v3['grounding']['cv_grounding']
    configs = [dict(feature_kind=kind, target=target, alpha=alpha)
               for kind, target, alpha in itertools.product(
                   ('dino_cls_scalar', 'dino_patch_scalar', 'dino_all_scalar'),
                   ('iou', 'utility'), (.01, .1, 1., 10., 100.))]
    trials, best, best_config = [], {'method': 'existing_v3_ridge', 'cv_grounding': baseline}, None
    for index, config in enumerate(configs, 1):
        fold_predictions = {}
        for fold_index, validation in enumerate(folds):
            training = [image for image in data.fit_ids if image not in validation]
            model = fit_model(data, dino, training, config)
            fold_predictions[fold_index] = predict_regions(data, dino, validation, model, config)
        for amount, scale in itertools.product((.25, .5, 1., 2., 4.), (.9, .95, 1., 1.05)):
            per_fold = []
            for fold_index, validation in enumerate(folds):
                scores = combine_scores(data, validation, lora_scores[fold_index],
                                        fold_predictions[fold_index], amount)
                boxes = chosen_boxes(data, validation, scores, scale)
                per_fold.append(grounding_metric(data, validation, boxes))
            queries = sum(row['queries'] for row in per_fold)
            score = sum(row['weighted_grounding']*row['queries'] for row in per_fold)/queries
            trial = {**config, 'amount': amount, 'scale': scale,
                     'cv_grounding': score,
                     'fold_scores': [row['weighted_grounding'] for row in per_fold],
                     'fold_queries': [row['queries'] for row in per_fold]}
            trials.append(trial)
            if score > best['cv_grounding']+1e-9:
                best, best_config = trial, trial
        print(f"DINOv2+RemoteCLIP ranking {index}/{len(configs)}: "
              f"best grouped CV {best['cv_grounding']:.4f}", flush=True)

    frozen = {'selected': best, 'remoteclip_adapter': adapter_config,
              'fit_only_promotion_baseline': {'method': 'training_v3 ridge', 'cv_grounding': baseline,
                                              'fold_queries': [21, 17, 16]},
              'fit_images': data.split['fit'], 'internal_dev_images': data.split['dev'],
              'selection_used': 'three SIFT-scene-group folds within original-train fit images',
              'official_validation_or_test_used': False,
              'dino_feature_manifest': 'training_v15/results/dino_features_manifest.json',
              'lora_fold_data_access_audits': audits,
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': {'dino_ranker.py': digest(Path(__file__)),
                              'encode_dino_regions.py': digest(OUT/'code/encode_dino_regions.py')}}
    write_json(OUT/'results/cv_trials.json', trials)
    write_json(OUT/'results/frozen_config.json', frozen)

    if best_config is None:
        retained = json.loads((BASE/'optimization_results/current_best.json').read_text(encoding='utf-8'))
        write_json(OUT/'results/selection.json', {
            'decision': 'retain_current_best', 'fit_only_baseline': baseline,
            'best_dino_candidate': max(trials, key=lambda row: row['cv_grounding']),
            'development_score_recomputed': False,
            'retained_method': retained['method'],
            'retained_development_score_out_of_1': retained['score_out_of_1'],
            'retained_score_source': 'optimization_results/current_best.json',
            'official_validation_or_test_used': False})
        print('No DINOv2 candidate beat grouped fit-only baseline; retained current best.', flush=True)
        return

    archive = TrainArchive()
    model, transform, state, history = cv_lora.train(data, data.fit_ids, adapter_config, archive)
    dino_model = fit_model(data, dino, data.fit_ids, best_config)
    checkpoint = OUT/'weights/DINOv2_RemoteCLIP_ranker.pt'
    torch.save({'dino_ranker': dino_model, 'remoteclip_lora_state': state,
                'config': frozen}, checkpoint)
    visual = lora_visual.encode_grounding(model, transform, data, archive, data.dev_ids)
    torch.save(visual, OUT/'features/dino_ranker_dev_lora_regions.pt')
    archive.audit(OUT/'results/full_fit_and_dev_data_access.json')
    archive.close()
    dino_scores = predict_regions(data, dino, data.dev_ids, dino_model, best_config)
    scores = combine_scores(data, data.dev_ids,
                            lora_visual.combined_scores(data, data.dev_ids, visual),
                            dino_scores, best_config['amount'])
    boxes = chosen_boxes(data, data.dev_ids, scores, best_config['scale'])
    ranker = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
    rankings = retrieval.predict(data, retrieval.features(data), data.dev_ids,
                                 ranker['retrieval_model'], ranker['config']['retrieval'])
    result, details = measure(data.cache, data.split['dev'], {'grounding': boxes, 'rankings': rankings})
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
    print('DINOv2 + RemoteCLIP with retained v2 retrieval:', result, flush=True)


if __name__ == '__main__':
    main()
