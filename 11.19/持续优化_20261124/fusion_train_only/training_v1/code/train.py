"""Train shared rankers and box regression; select only by grouped fit folds."""
import json
from datetime import datetime, timezone

import torch

from data import BASE, Data, decode_deltas, encode_deltas, iou
from common import digest, write_json

OUT = BASE / "training_v1"


def fit_ridge(x, y, alpha):
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(.01)
    x = (x - mean) / std / x.shape[1] ** .5
    target_mean = y.mean(0)
    solution = torch.linalg.solve(x @ x.T + alpha * torch.eye(len(x)), y - target_mean)
    return {"mean": mean, "std": std, "weight": x.T @ solution, "bias": target_mean}


def ridge(model, x):
    return ((x - model["mean"]) / model["std"] / x.shape[1] ** .5) @ model["weight"] + model["bias"]


def supervision(data, ids, kind):
    xs, overlaps = [], []
    for key in data.keys(ids):
        row = data.grounding[key]
        truth = data.truths(row["image"], row["query"], ids)
        overlap = torch.stack([iou(box, truth) for box in row["boxes"]]).max(1).values
        xs.append(row[kind])
        overlaps.append(overlap)
    return torch.cat(xs), torch.cat(overlaps)


def regression_supervision(data, ids):
    xs, ys = [], []
    seen = set()
    for key in data.keys(ids):
        row = data.grounding[key]
        truth = data.truths(row["image"], row["query"], ids)
        overlaps = torch.stack([iou(box, truth) for box in row["boxes"]])
        maximum, matched = overlaps.max(1)
        for j in range(len(maximum)):
            signature = (row["image"], tuple(row["boxes"][j].tolist()))
            if maximum[j] < .4 or signature in seen:
                continue
            seen.add(signature)
            xs.append(row["bbox_features"][j])
            ys.append(encode_deltas(row["boxes"][j:j+1], truth[matched[j]:matched[j]+1])[0])
    return torch.stack(xs), torch.stack(ys)


def target(overlap, mode):
    if mode == "iou":
        return overlap
    return .6 * (overlap >= .5).float() + .1 * (overlap >= .75).float()


def chosen_boxes(data, ids, scores, scale=1., regression=None, regression_fraction=0.):
    result = {}
    for key in data.keys(ids):
        row = data.grounding[key]
        image = data.records[row["image"]]
        boxes = row["boxes"]
        if regression is not None and regression_fraction:
            boxes = decode_deltas(boxes, regression_fraction * ridge(regression, row["bbox_features"]),
                                  image["width"], image["height"])
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        half = (boxes[:, 2:] - boxes[:, :2]) * scale / 2
        boxes = torch.cat([center-half, center+half], 1)
        boxes[:, 0::2].clamp_(0, image["width"])
        boxes[:, 1::2].clamp_(0, image["height"])
        result[key] = boxes[int(scores[key].argmax())].tolist()
    return result


def grounding_metric(data, ids, boxes):
    counts, n = [0, 0], 0
    for image in ids:
        record = data.records[image]
        for j, text in enumerate(record["texts"]):
            key = f"{data.texts.index(text)}:{image}"
            overlap = float(iou(torch.tensor(boxes[key]), record["truth"][j:j+1])[0])
            counts[0] += int(overlap >= .5)
            counts[1] += int(overlap >= .75)
            n += 1
    return {"queries": n, "Acc@0.5": counts[0]/n, "Acc@0.75": counts[1]/n,
            "weighted_grounding": (.6*counts[0]+.1*counts[1])/n}


def retrieval_supervision(data, ids, kind):
    xs, targets = [], []
    queries = sorted(set(data.texts.index(t) for i in ids for t in data.records[i]["texts"]))
    for q in queries:
        for i in ids:
            xs.append(data.retrieval[f"{q}:{i}"][kind])
            targets.append(float(data.texts[q] in data.records[i]["texts"]))
    return torch.stack(xs), torch.tensor(targets)


def retrieval_predictions(data, ids, model=None, kind="scalar", fraction=0.):
    result = []
    for q in range(len(data.texts)):
        rows = [data.retrieval[f"{q}:{i}"] for i in ids]
        base = torch.tensor([r["base_score"] for r in rows])
        score = base if model is None else base + fraction * ridge(model, torch.stack([r[kind] for r in rows]))
        order = score.argsort(descending=True, stable=True)[:5]
        result.append([ids[j] for j in order.tolist()])
    return result


def retrieval_metric(data, ids, rankings):
    hits1 = hits5 = n = 0
    for i in ids:
        for text in data.records[i]["texts"]:
            rank = rankings[data.texts.index(text)]
            n += 1
            hits1 += int(rank[0] == i)
            hits5 += int(i in rank)
    return {"queries": n, "R@1": hits1/n, "R@5": hits5/n,
            "weighted_retrieval": (.1*hits1+.2*hits5)/n}


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261123)
    data = Data()
    folds = data.folds()
    write_json(OUT / "results/protocol.json", {
        "target_score": .6, "original_train_fit_images": len(data.fit_ids),
        "original_train_internal_dev_images": len(data.dev_ids),
        "groups_preserved": True, "fold_image_ids": folds,
        "official_validation_or_test_used": False,
        "hyperparameter_selection": "three grouped folds within the original 33 fit images",
        "known_limitation": "The old 16-image internal-dev score has already informed the framework choice.",
        "starting_frameworks": ["RemoteCLIP", "GeoRSCLIP", "GroundingDINO", "GeoGround", "Qwen3-VL"]})
    baseline_scores = {k: r["base_score"] for k,r in data.grounding.items()}
    baseline_boxes = chosen_boxes(data, data.fit_ids, baseline_scores)
    print("Starting fit grounding", grounding_metric(data, data.fit_ids, baseline_boxes), flush=True)
    oof_cache, regression_oof = {}, {}
    trials = []
    base_cv = grounding_metric(data, data.fit_ids, baseline_boxes)["weighted_grounding"]
    best = {"kind": "base", "alpha": 0., "target": "iou", "fraction": 0., "scale": 1.,
            "cv_grounding": base_cv}
    best_scores = baseline_scores
    for kind in ("scalar", "vision", "conditioned"):
        for mode in ("iou", "threshold"):
            for alpha in (.01, .1, 1., 10., 100.):
                predictions = {}
                for validation in folds:
                    training = [i for i in data.fit_ids if i not in validation]
                    x, overlaps = supervision(data, training, kind)
                    model = fit_ridge(x, target(overlaps, mode), alpha)
                    for key in data.keys(validation):
                        predictions[key] = ridge(model, data.grounding[key][kind])
                oof_cache[(kind, mode, alpha)] = predictions
                for fraction in (.1, .25, .5, 1., 2.):
                    scores = {k: baseline_scores[k] + fraction * v for k,v in predictions.items()}
                    for scale in (.85, .95, 1., 1.05, 1.15):
                        boxes = chosen_boxes(data, data.fit_ids, scores, scale)
                        metric = grounding_metric(data, data.fit_ids, boxes)["weighted_grounding"]
                        trial = {"kind": kind, "alpha": alpha, "target": mode,
                                 "fraction": fraction, "scale": scale, "cv_grounding": metric}
                        trials.append(trial)
                        if metric > best["cv_grounding"] + 1e-9:
                            best, best_scores = trial, scores
            print(f"Grouped CV {kind}/{mode}: best grounding {best['cv_grounding']:.4f}", flush=True)
    # Box regression is trained on matching predicted regions of training folds only.
    best_regression = {"alpha": 1., "fraction": 0., "cv_grounding": best["cv_grounding"]}
    for alpha in (.1, 1., 10., 100.):
        models = []
        for validation in folds:
            training = [i for i in data.fit_ids if i not in validation]
            x,y = regression_supervision(data, training)
            models.append(fit_ridge(x,y,alpha))
        for fraction in (.25, .5, 1.):
            boxes = {}
            for validation, model in zip(folds, models):
                boxes.update(chosen_boxes(data, validation, best_scores, best["scale"], model, fraction))
            metric = grounding_metric(data, data.fit_ids, boxes)["weighted_grounding"]
            if metric > best_regression["cv_grounding"] + 1e-9:
                best_regression = {"alpha": alpha, "fraction": fraction, "cv_grounding": metric}
    print("Selected grounding", best, "box regression", best_regression, flush=True)
    # Retrieval uses complete galleries and a multi-positive training objective.
    retrieval_trials = []
    initial = retrieval_metric(data, data.fit_ids, retrieval_predictions(data, data.fit_ids))
    best_retrieval = {"kind": "base", "alpha": 0., "fraction": 0., "cv_retrieval": 0.}
    # Each fold is scored in its own gallery, matching the gallery setup of validation.
    base_hits = []
    for validation in folds:
        base_hits.append(retrieval_metric(data, validation, retrieval_predictions(data,validation)))
    best_retrieval["cv_retrieval"] = sum(r["weighted_retrieval"]*r["queries"] for r in base_hits)/sum(r["queries"] for r in base_hits)
    for kind in ("scalar", "vision", "conditioned"):
        for alpha in (.01, .1, 1., 10., 100.):
            models = []
            for validation in folds:
                training = [i for i in data.fit_ids if i not in validation]
                x,y = retrieval_supervision(data,training,kind)
                models.append(fit_ridge(x,y,alpha))
            for fraction in (.03, .1, .3, 1., 3.):
                metrics = [retrieval_metric(data, validation,
                    retrieval_predictions(data,validation,model,kind,fraction))
                    for validation,model in zip(folds,models)]
                score = sum(r["weighted_retrieval"]*r["queries"] for r in metrics)/sum(r["queries"] for r in metrics)
                trial = {"kind":kind,"alpha":alpha,"fraction":fraction,"cv_retrieval":score}
                retrieval_trials.append(trial)
                if score > best_retrieval["cv_retrieval"] + 1e-9:
                    best_retrieval = trial
        print(f"Grouped retrieval CV {kind}: best weighted recall {best_retrieval['cv_retrieval']:.4f}",flush=True)
    config = {"grounding": best, "regression": best_regression, "retrieval":best_retrieval,
              "selection_used": "grouped cross-validation within 33 fit images only",
              "internal_dev_used_for_fitting_or_selection": False,
              "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(OUT / "results/cv_trials.json", {"grounding":trials,"retrieval":retrieval_trials})
    write_json(OUT / "results/frozen_config.json", config)
    bundle = {"config":config}
    if best["kind"] != "base":
        x,overlap = supervision(data,data.fit_ids,best["kind"])
        bundle["grounding"] = fit_ridge(x,target(overlap,best["target"]),best["alpha"])
    if best_regression["fraction"]:
        x,y=regression_supervision(data,data.fit_ids)
        bundle["regression"]=fit_ridge(x,y,best_regression["alpha"])
    if best_retrieval["kind"] != "base":
        x,y=retrieval_supervision(data,data.fit_ids,best_retrieval["kind"])
        bundle["retrieval"]=fit_ridge(x,y,best_retrieval["alpha"])
    (OUT / "weights").mkdir(exist_ok=True,parents=True)
    torch.save(bundle, OUT / "weights/trained_fusion_heads.pt")
    write_json(OUT / "results/training_summary.json",{
        "fitted_on_image_labels": [data.records[i]["label"] for i in data.fit_ids],
        "trained_parameters": sum(t.numel() for k,v in bundle.items() if k != "config" for t in v.values()),
        "checkpoint_sha256":digest(OUT / "weights/trained_fusion_heads.pt"),
        "fit_cross_validation_score_out_of_1": best_regression["cv_grounding"]+best_retrieval["cv_retrieval"],
        "config":config})
    print("Frozen checkpoint", config, flush=True)


if __name__ == "__main__":
    main()
