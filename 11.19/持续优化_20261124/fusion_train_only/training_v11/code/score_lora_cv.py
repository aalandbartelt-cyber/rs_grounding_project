"""Complete scoring from the already frozen fold-selected adapter and cached dev features."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
from inputs import Data, digest, write_json
from score_round import measure
from train import chosen_boxes
from lora_visual import combined_scores
import retrieval

OUT = BASE/'training_v11'


def main():
    torch.set_num_threads(4)
    data = Data()
    config = json.loads((OUT/'results/frozen_config.json').read_text(encoding='utf-8'))
    checkpoint = OUT/'weights/RemoteCLIP_visual_lora_cvselected.pt'
    summary = torch.load(checkpoint, weights_only=True)
    assert summary['config']['selected'] == config['selected']
    visual = torch.load(OUT/'features/lora_cvselected_dev_regions.pt', weights_only=True)
    previous = torch.load(BASE/'training_v1/weights/trained_fusion_final.pt', weights_only=True)
    predictions = __import__('evaluate').predict(data, data.dev_ids, previous)
    predictions['grounding'] = chosen_boxes(data, data.dev_ids,
        combined_scores(data, data.dev_ids, visual), config['selected']['scale'])
    ranker = joblib.load(BASE/'training_v2/weights/multiview_candidate_ranker.joblib')
    inputs = retrieval.features(data)
    predictions['rankings'] = retrieval.predict(data, inputs, data.dev_ids,
        ranker['retrieval_model'], ranker['config']['retrieval'])
    result, details = measure(data.cache, data.split['dev'], predictions)
    write_json(OUT/'results/internal_dev_predictions.json', details)
    report = {'combined_with_v2_retrieval': result, 'target_score': .6,
              'target_met': result['score_out_of_1'] >= .6,
              'official_validation_or_test_used': False,
              'checkpoint_sha256': digest(checkpoint), 'config': config,
              'scored_at_utc': datetime.now(timezone.utc).isoformat(),
              'previous_best_same_internal_dev_score': .5259259259259259}
    write_json(OUT/'results/internal_dev_score.json', report)
    print('CV-selected LoRA with v2 retrieval:', result, flush=True)


if __name__ == '__main__':
    main()
