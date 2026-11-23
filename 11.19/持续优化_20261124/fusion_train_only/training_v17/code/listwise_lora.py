"""Train query-local candidate ranking longer using original-train fit supervision."""
import gc
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch
from PIL import ImageEnhance, ImageOps
from torch.nn import functional as F

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v13/code'))
from pairwise_lora import load_model
from inputs import Data, TrainArchive, crop, digest, iou, write_json
import lora_visual
from train import chosen_boxes, grounding_metric
from score_round import measure
import retrieval

OUT = BASE/'training_v17'
STAGES = (6, 12)


def training_examples(data, ids):
    assert set(ids) <= set(data.fit_ids), 'Listwise LoRA supervision outside fit split'
    rows = []
    for key in data.keys(ids):
        row = data.grounding[key]
        truth = data.truths(row['image'], row['query'], ids)
        overlaps = torch.stack([iou(box, truth) for box in row['boxes']])
        utility = (.6*(overlaps >= .5).float()+.1*(overlaps >= .75).float()).mean(1)
        rows.append({'key': key, 'image': row['image'], 'query': row['query'],
                     'boxes': row['boxes'], 'utility': utility})
    return rows


def train(data, ids, learning_rate, epochs, archive, evaluate_ids=None, callback=None):
    # Validate boundaries before creating models or opening image payloads.
    examples = training_examples(data, ids)
    torch.manual_seed(20261124)
    torch.cuda.manual_seed_all(20261124)
    model, transform = load_model()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    rng = random.Random(20261124)
    images = {image: archive.image(data.records[image]['label']) for image in ids}
    texts = data.text_features['RemoteCLIP'].cuda()
    history, evaluations = [], {}
    for epoch in range(1, epochs+1):
        rng.shuffle(examples)
        model.train()
        total = 0.
        for row in examples:
            views = []
            for box in row['boxes']:
                view = crop(images[row['image']], box, rng.choice((1., 1.25)))
                if rng.random() < .5:
                    view = ImageOps.mirror(view)
                view = view.rotate(rng.choice((0, 90, 180, 270)), expand=True)
                view = ImageEnhance.Brightness(view).enhance(rng.uniform(.9, 1.1))
                views.append(transform(view))
            optimizer.zero_grad(set_to_none=True)
            chunks = []
            for start in range(0, len(views), 4):
                with torch.autocast('cuda', dtype=torch.float16):
                    visual = model.encode_image(torch.stack(views[start:start+4]).cuda()).float()
                    chunks.append(F.normalize(visual, dim=-1))
            similarity = torch.cat(chunks)@texts[row['query']]
            logits = 20*similarity
            utility = row['utility'].cuda()
            distribution = (utility/.1).softmax(0)
            ranking_loss = -(distribution*logits.log_softmax(0)).sum()
            alignment_loss = F.binary_cross_entropy_with_logits(logits-5, (utility/.7).clamp(0, 1))
            loss = ranking_loss+.2*alignment_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        entry = {'epoch': epoch, 'mean_group_loss': total/len(examples)}
        history.append(entry)
        print(f'Listwise LoRA lr={learning_rate} epoch {epoch}/{epochs} loss={entry["mean_group_loss"]:.5f}', flush=True)
        if evaluate_ids is not None and epoch in STAGES:
            vectors = lora_visual.encode_grounding(model, transform, data, archive, evaluate_ids)
            scores = lora_visual.combined_scores(data, evaluate_ids, vectors)
            evaluations[epoch] = {}
            for scale in (.9, .95, 1.):
                boxes = chosen_boxes(data, evaluate_ids, scores, scale)
                evaluations[epoch][scale] = grounding_metric(data, evaluate_ids, boxes)
            if callback is not None:
                callback(epoch, evaluations[epoch])
    state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
             if parameter.requires_grad}
    return model, transform, state, history, evaluations


def main():
    torch.set_num_threads(4)
    for folder in ('results', 'features', 'weights'):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    data = Data()
    folds = data.folds()
    baseline = json.loads((BASE/'training_v3/results/frozen_config.json').read_text(encoding='utf-8'))['grounding']['cv_grounding']
    trials, best = [], {'method': 'existing_v3_ridge', 'cv_grounding': baseline}
    for learning_rate in (.0005, .002):
        by_stage = {epoch: {scale: [] for scale in (.9, .95, 1.)} for epoch in STAGES}
        for fold_index, validation in enumerate(folds):
            training = [image for image in data.fit_ids if image not in validation]
            archive = TrainArchive()
            model, transform, state, history, evaluations = train(
                data, training, learning_rate, max(STAGES), archive, validation)
            for epoch in STAGES:
                for scale in by_stage[epoch]:
                    by_stage[epoch][scale].append(evaluations[epoch][scale])
            write_json(OUT/'results'/f'lr{learning_rate}_fold{fold_index}_history.json', history)
            archive.audit(OUT/'results'/f'lr{learning_rate}_fold{fold_index}_data_access.json')
            archive.close()
            del model, transform, state
            gc.collect()
            torch.cuda.empty_cache()
        for epoch in STAGES:
            for scale, metrics in by_stage[epoch].items():
                score = sum(row['weighted_grounding']*row['queries'] for row in metrics)/sum(row['queries'] for row in metrics)
                trial = {'learning_rate': learning_rate, 'epochs': epoch, 'scale': scale,
                         'cv_grounding': score,
                         'fold_scores': [row['weighted_grounding'] for row in metrics],
                         'fold_queries': [row['queries'] for row in metrics]}
                trials.append(trial)
                if score > best['cv_grounding']+1e-9:
                    best = trial
        print(f'Completed listwise lr={learning_rate}: best grouped CV {best["cv_grounding"]:.4f}', flush=True)
    frozen = {'selected': best, 'fit_only_baseline': baseline,
              'best_candidate': max(trials, key=lambda row: row['cv_grounding']),
              'fit_images': data.split['fit'], 'internal_dev_images': data.split['dev'],
              'selection_used': 'three SIFT-scene-group folds within original-train fit images',
              'loss': 'query-local utility-softmax listwise ranking + 0.2 binary image-text alignment',
              'official_validation_or_test_used': False,
              'limitation': 'Earlier architecture rounds consulted the same internal-dev split.',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': digest(Path(__file__))}
    write_json(OUT/'results/cv_trials.json', trials)
    write_json(OUT/'results/frozen_config.json', frozen)
    retained = json.loads((BASE/'optimization_results/current_best.json').read_text(encoding='utf-8'))
    if best.get('method') == 'existing_v3_ridge':
        write_json(OUT/'results/selection.json', {
            'decision': 'retain_current_best', 'fit_only_baseline': baseline,
            'best_candidate': frozen['best_candidate'], 'development_score_recomputed': False,
            'retained_development_score_out_of_1': retained['score_out_of_1'],
            'official_validation_or_test_used': False})
        print('No longer listwise LoRA candidate beat grouped fit-only baseline.', flush=True)
        return
    archive = TrainArchive()
    model, transform, state, history, _ = train(data, data.fit_ids, best['learning_rate'],
                                               best['epochs'], archive)
    checkpoint = OUT/'weights/RemoteCLIP_listwise_lora.pt'
    torch.save({'state': state, 'config': frozen}, checkpoint)
    visual = lora_visual.encode_grounding(model, transform, data, archive, data.dev_ids)
    torch.save(visual, OUT/'features/listwise_dev_regions.pt')
    write_json(OUT/'results/full_fit_training_history.json', history)
    archive.audit(OUT/'results/full_fit_and_dev_data_access.json')
    archive.close()
    boxes = chosen_boxes(data, data.dev_ids, lora_visual.combined_scores(data, data.dev_ids, visual), best['scale'])
    ranker = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
    rankings = retrieval.predict(data, retrieval.features(data), data.dev_ids,
                                 ranker['retrieval_model'], ranker['config']['retrieval'])
    result, details = measure(data.cache, data.split['dev'], {'grounding': boxes, 'rankings': rankings})
    write_json(OUT/'results/internal_dev_predictions.json', details)
    write_json(OUT/'results/internal_dev_score.json', {
        'combined_with_v2_retrieval': result, 'target_score': .6,
        'target_met': result['score_out_of_1'] >= .6,
        'official_validation_or_test_used': False, 'config': frozen,
        'checkpoint_sha256': digest(checkpoint), 'scored_at_utc': datetime.now(timezone.utc).isoformat(),
        'previous_best_same_internal_dev_score': retained['score_out_of_1']})
    print('Listwise LoRA with v2 retrieval:', result, flush=True)


if __name__ == '__main__':
    main()
