"""Cross-validate query-derived translation prompts for visual LoRA adaptation."""
import gc
import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, TrainArchive, digest, write_json
sys.path.insert(0, str(BASE/'training_v1/code'))
import lora_visual
from train import chosen_boxes, grounding_metric
from score_round import measure
import retrieval
sys.path.insert(0, str(BASE/'training_v11/code'))
import cv_lora

OUT = BASE/'training_v12'
CONFIG = {'rank': 8, 'blocks': 2, 'epochs': 3}


def apply_prompt(data, mode, prompts):
    data.text_features['RemoteCLIP'] = prompts['RemoteCLIP'][mode]


def cv_score(data, validation, scale, visual):
    scores = lora_visual.combined_scores(data, validation, visual)
    boxes = chosen_boxes(data, validation, scores, scale)
    return grounding_metric(data, validation, boxes)


def main():
    torch.set_num_threads(4)
    for name in ('results', 'weights', 'features'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    prompts_path = BASE/'training_v2/features/query_features.pt'
    prompts = torch.load(prompts_path, weights_only=True)
    data = Data()
    folds = data.folds()
    modes = ('literal', 'concise', 'attribute', 'ensemble', 'centered')
    trials = []
    best = None
    for mode in modes:
        fold_metrics = {scale: [] for scale in (.9, .95, 1.)}
        for fold_index, validation in enumerate(folds):
            data = Data()
            apply_prompt(data, mode, prompts)
            training = [image for image in data.fit_ids if image not in validation]
            archive = TrainArchive()
            model, transform, _, _ = cv_lora.train(data, training, CONFIG, archive)
            visual = lora_visual.encode_grounding(model, transform, data, archive, validation)
            for scale in fold_metrics:
                fold_metrics[scale].append(cv_score(data, validation, scale, visual))
            archive.audit(OUT/'results'/f'{mode}_fold_{fold_index}_data_access.json')
            archive.close()
            del model, visual
            gc.collect()
            torch.cuda.empty_cache()
            print(f'Prompt LoRA CV {mode} fold {fold_index+1}/3: '
                  f'{[fold_metrics[s][-1]["weighted_grounding"] for s in fold_metrics]}', flush=True)
        for scale, rows in fold_metrics.items():
            value = sum(row['weighted_grounding']*row['queries'] for row in rows)/sum(row['queries'] for row in rows)
            trial = {'prompt_mode': mode, **CONFIG, 'scale': scale, 'cv_grounding': value,
                     'fold_weighted_grounding': [row['weighted_grounding'] for row in rows],
                     'fold_queries': [row['queries'] for row in rows]}
            trials.append(trial)
            if best is None or value > best['cv_grounding']+1e-9:
                best = trial
        print(f'Prompt LoRA mode {mode}: best grouped-CV grounding {best["cv_grounding"]:.4f}', flush=True)
    write_json(OUT/'results/cv_trials.json', trials)
    frozen = {'selected': best, 'selection_used': 'three SIFT-scene-group folds within 33 original-train fit images',
              'fit_images': data.split['fit'], 'internal_dev_images': data.split['dev'],
              'official_validation_or_test_used': False,
              'prompts_derived_from_query_text_only': True,
              'prompt_features_sha256': digest(prompts_path),
              'limitation': 'Previous experiments consulted the fixed internal-dev split; prompt mode selection uses fit folds only.',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': {str(p.relative_to(BASE)): digest(p) for p in
                    (Path(__file__), BASE/'training_v11/code/cv_lora.py', BASE/'training_v1/code/lora_visual.py')}}
    write_json(OUT/'results/frozen_config.json', frozen)
    data = Data()
    apply_prompt(data, best['prompt_mode'], prompts)
    archive = TrainArchive()
    model, transform, state, history = cv_lora.train(data, data.fit_ids, CONFIG, archive)
    checkpoint = OUT/'weights/RemoteCLIP_visual_lora_prompt_cv.pt'
    torch.save({'state': state, 'config': frozen}, checkpoint)
    write_json(OUT/'results/full_fit_training_history.json', history)
    visual = lora_visual.encode_grounding(model, transform, data, archive, data.dev_ids)
    torch.save(visual, OUT/'features/lora_prompt_cv_dev_regions.pt')
    archive.audit(OUT/'results/full_fit_and_dev_data_access.json')
    archive.close()
    predictions = __import__('evaluate').predict(data, data.dev_ids,
        torch.load(BASE/'training_v1/weights/trained_fusion_final.pt', weights_only=True))
    predictions['grounding'] = chosen_boxes(data, data.dev_ids,
        lora_visual.combined_scores(data, data.dev_ids, visual), best['scale'])
    ranker = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
    inputs = retrieval.features(data)
    predictions['rankings'] = retrieval.predict(data, inputs, data.dev_ids,
        ranker['retrieval_model'], ranker['config']['retrieval'])
    result, details = measure(data.cache, data.split['dev'], predictions)
    write_json(OUT/'results/internal_dev_predictions.json', details)
    report = {'combined_with_v2_retrieval': result, 'target_score': .6,
              'target_met': result['score_out_of_1'] >= .6,
              'official_validation_or_test_used': False,
              'checkpoint_sha256': digest(checkpoint), 'config': frozen,
              'scored_at_utc': datetime.now(timezone.utc).isoformat(),
              'previous_best_same_internal_dev_score': .5259259259259259}
    write_json(OUT/'results/internal_dev_score.json', report)
    print('Query-prompt-selected LoRA with v2 retrieval:', result, flush=True)


if __name__ == '__main__':
    main()
