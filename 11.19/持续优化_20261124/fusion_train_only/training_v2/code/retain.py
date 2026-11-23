"""Retain complete, reproducible combinations without replacing earlier rounds."""
from datetime import datetime, timezone

import joblib
import torch

from inputs import BASE, OUT, Data, digest, write_json
from evaluate import predict
from lora_visual import combined_scores
from train import chosen_boxes
from score_round import measure
import retrieval


def main():
    torch.set_num_threads(4)
    data = Data()
    previous = torch.load(BASE/'training_v1/weights/trained_fusion_final.pt', weights_only=True)
    bundle = joblib.load(OUT/'weights/multiview_candidate_ranker.joblib')
    predictions = predict(data, data.dev_ids, previous)
    predictions['rankings'] = retrieval.predict(data, retrieval.features(data), data.dev_ids,
                                               bundle['retrieval_model'], bundle['config']['retrieval'])
    old, old_details = measure(data.cache, data.split['dev'], predictions)
    visual = torch.load(BASE/'training_v1/features/RemoteCLIP_lora_dev_regions.pt', weights_only=True)
    predictions['grounding'] = chosen_boxes(data, data.dev_ids,
                                           combined_scores(data, data.dev_ids, visual), .95)
    result, details = measure(data.cache, data.split['dev'], predictions)
    write_json(OUT/'results/retained_predictions.json', details)
    write_json(OUT/'results/retained_result.json', {
        **result, 'method': 'v2 attribute retrieval with v1 RemoteCLIP LoRA localization',
        'score_formula': '0.1*R@1 + 0.2*R@5 + 0.6*Acc@0.5 + 0.1*Acc@0.75',
        'target_score': .6, 'target_met': result['score_out_of_1'] >= .6,
        'official_validation_or_test_used': False,
        'selection_consulted_internal_dev': True,
        'scope': 'iterative development on the same 16 original-train images; 27 queries per task',
        'previous_best_same_internal_dev_score': .5037037037037037,
        'scored_at_utc': datetime.now(timezone.utc).isoformat(),
        'required_checkpoints': {
            str(path.relative_to(BASE)): digest(path) for path in (
                BASE/'training_v1/weights/RemoteCLIP_visual_lora.pt',
                OUT/'weights/multiview_candidate_ranker.joblib')},
        'ablation_without_visual_lora': old})
    print('Retained combination:', result, flush=True)


if __name__ == '__main__':
    main()
