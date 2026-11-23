"""Score the frozen checkpoint once on the original 16-image internal dev split."""
from datetime import datetime, timezone
import json

import torch

from data import BASE,Data
from common import digest,write_json
from train import chosen_boxes,ridge,retrieval_predictions
from score_round import measure
import neural

OUT=BASE/"training_v1"


def predict(data,ids,bundle):
    config=bundle["config"]
    grounding=config["grounding"]
    if grounding["family"]=="neural":
        scores=neural.grounding_scores(data,ids,bundle["grounding"])
    else:
        scores={}
        for key in data.keys(ids):
            row=data.grounding[key]
            score=row["base_score"]
            if grounding["kind"]!="base":
                score=score+grounding["fraction"]*ridge(bundle["grounding"],row[grounding["kind"]])
            scores[key]=score
    boxes=chosen_boxes(data,ids,scores,grounding["scale"],bundle.get("regression"),config["regression"]["fraction"])
    retrieval=config["retrieval"]
    if retrieval["family"]=="neural":
        rankings=neural.retrieval_rankings(data,ids,bundle["retrieval"])
    else:
        rankings=retrieval_predictions(data,ids,bundle.get("retrieval"),retrieval["kind"],retrieval["fraction"])
    return {"grounding":boxes,"rankings":rankings}


def main():
    torch.set_num_threads(4)
    destination=OUT/"results/internal_dev_score.json"
    if destination.exists():
        raise FileExistsError("This frozen model has already been scored on internal dev; preserve that round")
    checkpoint=OUT/"weights/trained_fusion_final.pt"
    bundle=torch.load(checkpoint,weights_only=True)
    summary=json.loads((OUT/"results/final_training_summary.json").read_text(encoding="utf-8"))
    assert digest(checkpoint)==summary["checkpoint_sha256"]
    for name,expected in summary["code_sha256"].items():
        assert digest(OUT/"code"/name)==expected,name
    data=Data()
    predictions=predict(data,data.dev_ids,bundle)
    result,details=measure(data.cache,data.split["dev"],predictions)
    report={"scope":"16 original-training internal-dev images; 27 queries per task",
            "score_formula":"0.1*R@1 + 0.2*R@5 + 0.6*Acc@0.5 + 0.1*Acc@0.75",
            "official_validation_or_test_used":False,
            "training_images":33,"internal_dev_images":16,
            "fit_cross_validation_score_out_of_1":summary["fit_cross_validation_score_out_of_1"],
            "checkpoint_sha256":digest(checkpoint),
            "frozen_config_sha256":digest(OUT/"results/final_frozen_config.json"),
            "scored_at_utc":datetime.now(timezone.utc).isoformat(),
            "previous_best_same_dev_score":.4666666666666667,
            "target_score":.6,"target_met":result["score_out_of_1"]>=.6,
            **result}
    write_json(destination,report)
    write_json(OUT/"results/internal_dev_predictions.json",details)
    print(f"Frozen internal dev: {result['score_out_of_1']:.4f}/1 {result['counts']}",flush=True)


if __name__=="__main__":
    main()
