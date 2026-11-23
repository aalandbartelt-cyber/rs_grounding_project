"""Fit fusion coefficients on fit images, then score one original-train dev round."""
import itertools
import json
from datetime import datetime, timezone

import torch

from box_parser import iou, parse
from common import ROOT, digest, split_labels, write_json
from features import score_features, base_scores


def chinese_clip_baseline(cache, labels):
    ids = [i for i, row in enumerate(cache["records"]) if row["label"] in labels]
    visual = torch.nn.functional.normalize(cache["initial"], dim=-1)
    scores, boxes = [], {}
    for i in ids:
        row = cache["records"][i]
        n, start = len(row["boxes"]), row["start"]
        scores.append(cache["text"] @ visual[start])
        if n:
            local = cache["text"] @ visual[start + 1 + n:start + 1 + 2 * n].T
            local += 0.1 * row["scores"].clamp_min(1e-6).log()
            selected = local.argmax(1)
            for query, choice in enumerate(selected.tolist()):
                boxes[f"{query}:{i}"] = expanded_box(row["boxes"][choice], 0.9,
                                                    row["width"], row["height"]).tolist()
        else:
            for query in range(len(cache["texts"])):
                boxes[f"{query}:{i}"] = [0.] * 4
    order = torch.stack(scores, 1).argsort(dim=1, descending=True, stable=True)[:, :5]
    return {"rankings": [[ids[i] for i in row] for row in order.tolist()], "grounding": boxes}


def expanded_box(box, scale, width, height):
    center = (box[:2] + box[2:]) / 2
    radius = (box[2:] - box[:2]) * scale / 2
    result = torch.cat((center - radius, center + radius))
    result[0::2].clamp_(0, width)
    result[1::2].clamp_(0, height)
    return result


def prepare():
    cache, features = score_features()
    base, _ = base_scores(cache, features)
    jobs = json.loads((ROOT / "features/jobs.json").read_text(encoding="utf-8"))
    retrieval = json.loads((ROOT / "features/qwen_retrieval.json").read_text(encoding="utf-8"))
    grounding = json.loads((ROOT / "features/qwen_grounding.json").read_text(encoding="utf-8"))
    geo = json.loads((ROOT / "features/geoground_answers.json").read_text(encoding="utf-8"))
    geo_scores = torch.load(ROOT / "features/geoground_region_scores.pt", weights_only=True)
    units = json.loads((ROOT / "results/geoground_coordinate_units.json").read_text())["units"]
    return cache, features, base, jobs, retrieval, grounding, geo, geo_scores, units


def predict(data, labels, config):
    """Consumes predicted regions, query text and visual scores, without reading GT."""
    cache, features, base, jobs, retrieval, grounding, geo, geo_scores, units = data
    indices = {row["label"]: i for i, row in enumerate(cache["records"])}
    ids = [indices[label] for label in labels]
    mix = config["remote_fraction"]
    whole = mix * features["RemoteCLIP"][0] + (1 - mix) * features["GeoRSCLIP"][0]
    image_scores = []
    for i in ids:
        row = cache["records"][i]
        regional = mix * features["RemoteCLIP"][1][i] + (1 - mix) * features["GeoRSCLIP"][1][i]
        if len(row["boxes"]):
            regional = regional + config["detector"] * row["scores"].clamp_min(1e-6).log()
            best = regional.max(1).values
        else:
            best = whole[:, i]
        fraction = config["region_fraction"]
        image_scores.append((1 - fraction) * whole[:, i] + fraction * best)
    image_scores = torch.stack(image_scores, 1)
    rankings = []
    for query in range(len(cache["texts"])):
        # The shortlist is fixed before seeing the Qwen answers or fitting coefficients.
        if config.get("fixed_shortlist", True):
            shortlist = base[query, ids].argsort(dim=0, descending=True, stable=True)[:5]
        else:
            shortlist = image_scores[query].argsort(dim=0, descending=True, stable=True)[:5]
        local = image_scores[query, shortlist].clone()
        for col, index in enumerate(shortlist.tolist()):
            key = f"{query}:{ids[index]}"
            if key in retrieval:
                local[col] += config["qwen_retrieval"] * torch.tanh(
                    torch.tensor(retrieval[key]["yes_minus_no_logit"]) / 5)
        order = local.argsort(dim=0, descending=True, stable=True)
        rankings.append([ids[index] for index in shortlist[order].tolist()])
    predictions = {}
    for job in jobs["grounding"]:
        if job["label"] not in labels:
            continue
        query, image = job["query"], job["image"]
        key = f"{query}:{image}"
        row = cache["records"][image]
        boxes = [expanded_box(box, 0.9, row["width"], row["height"]) for box in row["boxes"]]
        regional = mix * features["RemoteCLIP"][1][image][query] + (1 - mix) * features["GeoRSCLIP"][1][image][query]
        local = regional + config["detector"] * row["scores"].clamp_min(1e-6).log()
        geo_box = (parse(geo[key]["answer"], row["width"], row["height"], units)
                   if config.get("enable_geoground", True) else None)
        if geo_box is not None:
            boxes.append(geo_box)
            extra = mix * geo_scores["RemoteCLIP"][key] + (1 - mix) * geo_scores["GeoRSCLIP"][key]
            confidence_prior = (float(row["scores"].max().clamp_min(1e-6).log())
                                if len(row["scores"]) else 0)
            local = torch.cat((local, torch.tensor([extra + config["detector"] * confidence_prior])))
        if boxes:
            boxes = torch.stack(boxes)
            if geo_box is not None:
                local += config["geoground_vote"] * iou(geo_box, boxes)
            selected = grounding[key]["chosen"]
            if 0 <= selected < len(boxes):
                local[selected] += config["qwen_grounding"]
            box = boxes[int(local.argmax())]
        else:
            box = torch.zeros(4)
        predictions[key] = box.tolist()
    return {"rankings": rankings, "grounding": predictions}


def measure(cache, labels, predictions):
    """The only scoring function that reads ground-truth boxes and source IDs."""
    selected = set(labels)
    text_ids = {t: q for q, t in enumerate(cache["texts"])}
    counts = {"R@1": 0, "R@5": 0, "Acc@0.5": 0, "Acc@0.75": 0}
    details = []
    for image, row in enumerate(cache["records"]):
        if row["label"] not in selected:
            continue
        for annotation, text in enumerate(row["texts"]):
            query = text_ids[text]
            key = f"{query}:{image}"
            ranking = predictions["rankings"][query]
            box = torch.tensor(predictions["grounding"][key])
            overlap = float(iou(box, row["truth"][annotation:annotation + 1])[0])
            counts["R@1"] += bool(ranking) and ranking[0] == image
            counts["R@5"] += image in ranking
            counts["Acc@0.5"] += overlap >= 0.5
            counts["Acc@0.75"] += overlap >= 0.75
            details.append({"label": row["label"], "annotation": annotation, "text": text,
                            "retrieval_rank": ranking.index(image) + 1 if image in ranking else None,
                            "iou": overlap, "bbox_xyxy": box.tolist()})
    n = len(details)
    result = {"images": len(selected), "queries_per_task": n, "counts": counts,
              **{k: v / n for k, v in counts.items()}}
    result["score_out_of_1"] = (counts["R@1"] + 2 * counts["R@5"]
                               + 6 * counts["Acc@0.5"] + counts["Acc@0.75"]) / (10 * n)
    return result, details


def configs():
    for mix, region, detector, qr, qg, gv in itertools.product(
            (0.0, 0.5, 1.0), (0.0, 0.5, 1.0), (0.05, 0.1),
            (0.0, 0.03, 0.1), (0.0, 0.05, 0.2), (0.0, 0.05, 0.2)):
        yield {"remote_fraction": mix, "region_fraction": region, "detector": detector,
               "qwen_retrieval": qr, "qwen_grounding": qg, "geoground_vote": gv}


def main():
    torch.set_num_threads(4)
    data = prepare()
    cache, _, _, _, _, _, _, _, _ = data
    split = split_labels()
    trials = []
    for number, config in enumerate(configs(), 1):
        result, _ = measure(cache, split["fit"], predict(data, split["fit"], config))
        trials.append({"config": config, "fit_score_out_of_1": result["score_out_of_1"]})
        if number % 100 == 0:
            print(f"Calibrated {number} settings on fit images only", flush=True)
    selected = max(trials, key=lambda row: row["fit_score_out_of_1"])
    write_json(ROOT / "results/fit_search.json", trials)
    frozen = {"selected": selected, "split_sha256": digest(ROOT / "results/split.json"),
              "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "selection_used": "fit images only", "dev_scores_used_for_selection": False,
              "retrieval_shortlist_size": 5,
              "code_sha256": {p.name: digest(p) for p in (ROOT / "code").glob("*.py")}}
    write_json(ROOT / "results/frozen_config.json", frozen)
    write_json(ROOT / "weights/fusion_coefficients.json", {
        "fusion_kind": "score-level ensemble of frozen pretrained networks",
        "network_weights_finetuned": False, "selected_config": selected["config"],
        "selection_used": "33 original-training fit images only",
        "frozen_config_sha256": digest(ROOT / "results/frozen_config.json")})
    baselines = {
        "RemoteCLIP_DINO": {"remote_fraction": 1.0, "region_fraction": 0.5, "detector": 0.1,
                            "qwen_retrieval": 0, "qwen_grounding": 0, "geoground_vote": 0,
                            "fixed_shortlist": False, "enable_geoground": False},
        "GeoRSCLIP_DINO": {"remote_fraction": 0.0, "region_fraction": 0.5, "detector": 0.1,
                           "qwen_retrieval": 0, "qwen_grounding": 0, "geoground_vote": 0,
                           "fixed_shortlist": False, "enable_geoground": False},
        "equal_remote_fusion": {"remote_fraction": 0.5, "region_fraction": 0.5, "detector": 0.1,
                                "qwen_retrieval": 0, "qwen_grounding": 0, "geoground_vote": 0,
                                "enable_geoground": False},
        "four_frameworks_equal": {"remote_fraction": 0.5, "region_fraction": 0.5, "detector": 0.1,
                                  "qwen_retrieval": 0.03, "qwen_grounding": 0.05, "geoground_vote": 0.05},
        "fit_selected_fusion": selected["config"]}
    baseline, details = measure(cache, split["dev"], chinese_clip_baseline(cache, split["dev"]))
    results = {"ChineseCLIP_DINO_frozen": baseline}
    write_json(ROOT / "results/ChineseCLIP_DINO_frozen_dev_predictions.json", details)
    print(f"ChineseCLIP_DINO_frozen: {baseline['score_out_of_1']:.4f}/1 {baseline['counts']}", flush=True)
    for name, config in baselines.items():
        result, details = measure(cache, split["dev"], predict(data, split["dev"], config))
        results[name] = {"config": config, **result}
        write_json(ROOT / f"results/{name}_dev_predictions.json", details)
        print(f"{name}: {result['score_out_of_1']:.4f}/1 {result['counts']}", flush=True)
    report = {"scope": "single grouped internal-development round from 49 original public-training images",
              "fit_images": len(split["fit"]), "dev_images": len(split["dev"]),
              "official_validation_or_test_images_used": False,
              "previously_fitted_adapters_used": False, "results": results,
              "model_inference_network_access": "blocked by Python audit hook",
              "quantization": "Qwen3-VL and GeoGround NF4 4-bit; visual encoders unquantized",
              "scored_at_utc": datetime.now(timezone.utc).isoformat(),
              "retrieval_reranking_limitation": "Qwen reranks a fixed top-five shortlist",
              "frozen_config_sha256": digest(ROOT / "results/frozen_config.json")}
    write_json(ROOT / "results/score_round.json", report)


if __name__ == "__main__":
    main()
