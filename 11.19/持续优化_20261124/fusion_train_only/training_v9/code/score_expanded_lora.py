"""Apply the previously frozen LoRA scoring formula to expanded predicted boxes."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, digest, write_json
from fusion_features import Features, box_predictions
from score_round import measure
from train import chosen_boxes
from lora_visual import combined_scores
import retrieval

OUT = BASE/'training_v9'


def main():
    torch.set_num_threads(4)
    (OUT/'results').mkdir(parents=True, exist_ok=True)
    features = Features()
    data = features.data
    checkpoint = BASE/'training_v2/features/candidates.pt'
    scores = {key: row['adapted_base'] for key, row in features.rows.items()}
    scale = .95
    predicted = box_predictions(features, data.dev_ids, scores, scale)
    previous = torch.load(BASE/'training_v1/weights/trained_fusion_final.pt', weights_only=True)
    predictions = __import__('evaluate').predict(data, data.dev_ids, previous)
    predictions['grounding'] = predicted
    v2 = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
    predictions['rankings'] = retrieval.predict(data, retrieval.features(data), data.dev_ids,
        v2['retrieval_model'], v2['config']['retrieval'])
    result, details = measure(data.cache, data.split['dev'], predictions)
    write_json(OUT/'results/expanded_lora_predictions.json', details)
    config = {'method': 'training_v2 RemoteCLIP LoRA candidate similarity plus GeoRSCLIP',
              'weights': 'unchanged combined_scores formula from training_v1/code/lora_visual.py',
              'box_scale': scale, 'fit_or_dev_labels_used_to_change_scoring_weights': False,
              'training_images': data.split['fit'], 'internal_dev_images': data.split['dev'],
              'candidate_feature_sha256': digest(checkpoint),
              'lora_weight_sha256': digest(BASE/'training_v1/weights/RemoteCLIP_visual_lora.pt'),
              'official_validation_or_test_used': False,
              'limitation': 'Repeated development has consulted this same internal-dev split.',
              'scored_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(OUT/'results/frozen_config.json', config)
    write_json(OUT/'results/internal_dev_score.json', {
        **result, 'target_score': .6, 'target_met': result['score_out_of_1'] >= .6,
        'official_validation_or_test_used': False, 'config': config,
        'checkpoint_sha256': digest(checkpoint)})
    print('Expanded candidates with frozen LoRA scoring:', result, flush=True)


if __name__ == '__main__':
    main()
