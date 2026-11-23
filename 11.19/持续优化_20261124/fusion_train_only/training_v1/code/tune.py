"""Compare ridge and gradient-trained rankers in the fit-side grouped folds."""
import itertools
import json
from datetime import datetime, timezone

import torch

from data import BASE,Data
from common import digest,write_json
from train import (chosen_boxes,fit_ridge,grounding_metric,retrieval_metric,regression_supervision,
                   retrieval_predictions,retrieval_supervision,ridge,supervision,target)
import neural

OUT=BASE/"training_v1"


def main():
    torch.set_num_threads(4)
    data=Data()
    folds=data.folds()
    previous=json.loads((OUT/"results/frozen_config.json").read_text(encoding="utf-8"))
    bundle=torch.load(OUT/"weights/trained_fusion_heads.pt",weights_only=True)
    grounding_config=previous["grounding"]
    grounding_config={"family":"ridge",**grounding_config}
    retrieval_config={"family":"ridge",**previous["retrieval"]}
    best_grounding=grounding_config["cv_grounding"]
    best_retrieval=retrieval_config["cv_retrieval"]
    trials=[]
    oof_best=None
    for task in ("grounding","retrieval"):
        for kind,hidden,decay,epochs in itertools.product(("scalar","conditioned"),(0,16),(.01,.3),(60,180)):
            models=[]
            for fold in folds:
                ids=[i for i in data.fit_ids if i not in fold]
                models.append(neural.train(data,ids,task,kind,hidden,decay,epochs))
            config={"family":"neural","kind":kind,"hidden":hidden,"decay":decay,"epochs":epochs}
            if task == "grounding":
                scores={}
                for fold,model in zip(folds,models):
                    scores.update(neural.grounding_scores(data,fold,model))
                for scale in (.85,.95,1.,1.05,1.15):
                    metric=grounding_metric(data,data.fit_ids,chosen_boxes(data,data.fit_ids,scores,scale))["weighted_grounding"]
                    row={"task":task,**config,"scale":scale,"cv_score":metric}
                    trials.append(row)
                    if metric>best_grounding+1e-9:
                        best_grounding=metric
                        grounding_config={**config,"scale":scale,"cv_grounding":metric}
                        oof_best=scores
            else:
                metrics=[retrieval_metric(data,fold,neural.retrieval_rankings(data,fold,model)) for fold,model in zip(folds,models)]
                metric=sum(r["weighted_retrieval"]*r["queries"] for r in metrics)/sum(r["queries"] for r in metrics)
                trials.append({"task":task,**config,"cv_score":metric})
                if metric>best_retrieval+1e-9:
                    best_retrieval=metric
                    retrieval_config={**config,"cv_retrieval":metric}
            print(f"{task} {kind} hidden={hidden} decay={decay} epochs={epochs}: selected CV {best_grounding+best_retrieval:.4f}/1",flush=True)
    # Retrain the selected models using all 33 fit images. Dev remains untouched.
    for task,config in (("grounding",grounding_config),("retrieval",retrieval_config)):
        if config["family"]=="neural":
            bundle[task]=neural.train(data,data.fit_ids,task,config["kind"],config["hidden"],config["decay"],config["epochs"])
    config={"grounding":grounding_config,"retrieval":retrieval_config,
            "regression":previous["regression"],
            "selection_used":"three group folds within 33 fit images only",
            "internal_dev_used_for_fitting_or_selection":False,
            "frozen_at_utc":datetime.now(timezone.utc).isoformat()}
    bundle["config"]=config
    write_json(OUT/"results/neural_cv_trials.json",trials)
    write_json(OUT/"results/final_frozen_config.json",config)
    torch.save(bundle,OUT/"weights/trained_fusion_final.pt")
    write_json(OUT/"results/final_training_summary.json",{
        "fit_cross_validation_score_out_of_1":best_grounding+best_retrieval,
        "fit_image_count":len(data.fit_ids),"network_backbones_frozen":True,
        "trained_components":["supervised candidate ranker","multi-positive image retrieval ranker"],
        "official_validation_or_test_used":False,
        "checkpoint_sha256":digest(OUT/"weights/trained_fusion_final.pt"),
        "code_sha256":{p.name:digest(p) for p in (OUT/"code").glob("*.py")},
        "qwen_semantic_sha256":digest(OUT/"features/qwen_semantic.json"),"config":config})
    print("Final frozen fit CV",best_grounding+best_retrieval,config,flush=True)


if __name__=="__main__":
    main()
