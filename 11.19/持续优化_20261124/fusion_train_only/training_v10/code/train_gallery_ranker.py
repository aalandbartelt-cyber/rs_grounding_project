"""Score unbiased Qwen gallery comparisons on fit-only grouped folds."""
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, digest, write_json
import retrieval
sys.path.insert(0, str(BASE/'training_v1/code'))
from train import retrieval_metric
from score_round import measure
from lora_visual import combined_scores
from train import chosen_boxes

OUT = BASE/'training_v10'


def pair_scores(data, record, q, gallery, method):
    ids = record['jobs']['selections'][f'{gallery}:{q}']
    positions = {image: index for index, image in enumerate(ids)}
    wins, degree = torch.zeros(len(ids)), torch.zeros(len(ids))
    laplacian = torch.eye(len(ids))*.01
    rhs = torch.zeros(len(ids))
    for a, b in itertools.combinations(ids, 2):
        key = f'{q}:{min(a,b)}:{max(a,b)}'
        row = record['preferences'][key]
        preference = row['symmetric_preference']
        first, second = positions[min(a,b)], positions[max(a,b)]
        signed = float(torch.tensor(preference/5).tanh())
        wins[first] += signed
        wins[second] -= signed
        degree[first] += 1
        degree[second] += 1
        laplacian[first, first] += 1
        laplacian[second, second] += 1
        laplacian[first, second] -= 1
        laplacian[second, first] -= 1
        rhs[first] += preference
        rhs[second] -= preference
    vote = wins/degree.clamp_min(1)
    if method == 'wins':
        score = vote
    else:
        rating = torch.linalg.solve(laplacian, rhs).div(5).tanh()
        if method == 'rating':
            score = rating
        elif method == 'consensus':
            score = (vote+rating)/2
        else:
            raise ValueError(method)
    return {image: float(score[index]) for image, index in positions.items()}


def predict(data, raw, ids, model, config, record, gallery):
    rankings = []
    for q in range(len(data.texts)):
        x = torch.stack([raw[f'{q}:{image}'] for image in ids])
        local = torch.from_numpy(model.predict(x.numpy())).float()
        base = torch.tensor([data.retrieval[f'{q}:{image}']['base_score'] for image in ids])
        scores = base+config['retrieval']['amount']*local
        if config['method'] != 'baseline' and f'{gallery}:{q}' in record['jobs']['selections']:
            evidence = pair_scores(data, record, q, gallery, config['method'])
            candidate_ids = record['jobs']['selections'][f'{gallery}:{q}']
            for image in candidate_ids:
                scores[ids.index(image)] += config['amount']*evidence[image]
        order = torch.argsort(scores, descending=True, stable=True)[:5].tolist()
        rankings.append([ids[index] for index in order])
    return rankings


def main():
    torch.set_num_threads(4)
    for name in ('results', 'weights'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    data = Data()
    raw = retrieval.features(data)
    record = json.loads((OUT/'features/qwen_gallery_pairs.json').read_text(encoding='utf-8'))
    if len(record['preferences']) != record['jobs']['pair_count']:
        raise ValueError('Gallery comparisons are incomplete')
    selected = json.loads((BASE/'training_v2/results/frozen_config.json').read_text(encoding='utf-8'))['retrieval']
    folds = data.folds()
    models, gallery = [], []
    for fold, val in enumerate(folds):
        train_ids = [image for image in data.fit_ids if image not in val]
        models.append(retrieval.fit(data, raw, train_ids, selected))
        gallery.append(f'cv{fold}')
    models.append(retrieval.fit(data, raw, data.fit_ids, selected))
    configs = [{'method': 'baseline', 'amount': 0., 'retrieval': selected}]
    configs.extend({'method': method, 'amount': amount, 'retrieval': selected}
                   for method in ('wins', 'rating', 'consensus')
                   for amount in (.03, .1, .3, 1., 3.))
    trials, best = [], None
    for config in configs:
        if config['method'] == 'baseline':
            pass
        metrics = [retrieval_metric(data, val, predict(data, raw, val, model,
                   config, record, name)) for val, model, name in zip(folds, models, gallery)]
        score = sum(row['weighted_retrieval']*row['queries'] for row in metrics)/sum(row['queries'] for row in metrics)
        trial = {**config, 'cv_retrieval': score,
                 'fold_weighted_retrieval': [row['weighted_retrieval'] for row in metrics],
                 'fold_queries': [row['queries'] for row in metrics]}
        trials.append(trial)
        if best is None or score > best['cv_retrieval']+1e-9:
            best = trial
        print(f'Qwen gallery rank {config["method"]}/{config["amount"]}: {score:.4f}; best {best["cv_retrieval"]:.4f}', flush=True)
    write_json(OUT/'results/cv_trials.json', trials)
    frozen = {'selected': best, 'fit_images': data.split['fit'],
              'internal_dev_images': data.split['dev'],
              'selection_used': 'three scene-group folds within 33 fit images only',
              'official_validation_or_test_used': False,
              'qwen_pair_preferences': 'Qwen3-VL compares only top-8 predicted gallery images; candidate and answer order are swapped, then scores symmetrized',
              'limitation': 'Repeated internal-development rounds have informed the research direction.',
              'pair_feature_sha256': digest(OUT/'features/qwen_gallery_pairs.json'),
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': {str(p.relative_to(BASE)): digest(p) for p in
                  (Path(__file__), OUT/'code/encode_gallery_pairs.py')}}
    write_json(OUT/'results/frozen_config.json', frozen)
    checkpoint = OUT/'weights/qwen_gallery_ranker.joblib'
    joblib.dump({'retrieval_model': models[-1], 'config': best, 'protocol': frozen}, checkpoint)
    rankings = predict(data, raw, data.dev_ids, models[-1], best, record, 'dev')
    metric = retrieval_metric(data, data.dev_ids, rankings)
    previous = torch.load(BASE/'training_v1/weights/trained_fusion_final.pt', weights_only=True)
    predictions = __import__('evaluate').predict(data, data.dev_ids, previous)
    predictions['rankings'] = rankings
    visual = torch.load(BASE/'training_v1/features/RemoteCLIP_lora_dev_regions.pt', weights_only=True)
    predictions['grounding'] = chosen_boxes(data, data.dev_ids, combined_scores(data, data.dev_ids, visual), .95)
    combined, details = measure(data.cache, data.split['dev'], predictions)
    write_json(OUT/'results/internal_dev_predictions.json', details)
    report = {'retrieval': metric, 'combined_with_retained_lora_localization': combined,
              'target_score': .6, 'target_met': combined['score_out_of_1'] >= .6,
              'official_validation_or_test_used': False,
              'checkpoint_sha256': digest(checkpoint), 'config': frozen,
              'scored_at_utc': datetime.now(timezone.utc).isoformat(),
              'previous_best_same_internal_dev_score': .5259259259259259}
    write_json(OUT/'results/internal_dev_score.json', report)
    print('Qwen gallery preference retrieval:', metric, flush=True)
    print('Combined with retained LoRA:', combined, flush=True)


if __name__ == '__main__':
    main()
