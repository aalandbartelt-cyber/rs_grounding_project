"""Compare full-image retrieval with text-conditioned civilian region ranking.

Tune only on image folds inside the reviewed training split. Freeze config.json
before running the separate evaluate command on the reviewed holdout.
"""
import argparse
import contextlib
import io
import itertools
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import torch
from PIL import Image
from torch.nn.functional import normalize
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          ChineseCLIPModel, ChineseCLIPProcessor)

from check_civilian_grounding import box_ious, read_truth, summary
from prepare_check import WORKSPACE, check_models, digest, image_name, read_reviewed_archive
from train_civilian_retrieval import train_adapter

SEED = 32
PROMPT = "civilian passenger airliner."
CACHE_VERSION = 1


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def scaled_boxes(boxes, scale, width, height):
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    radius = (boxes[:, 2:] - boxes[:, :2]) * scale / 2
    result = torch.cat((center - radius, center + radius), dim=1)
    result[:, 0::2].clamp_(0, width)
    result[:, 1::2].clamp_(0, height)
    return result


def crop(image, box, scale):
    box = scaled_boxes(torch.tensor([box]), scale, image.width, image.height)[0]
    x1, y1, x2, y2 = box.tolist()
    return image.crop((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)))


def extract(args, split, manifest):
    path = args.output / f"features_{split}.pt"
    signature = {"version": CACHE_VERSION, "archive": digest(args.archive),
                 "labels": manifest["splits"][split], "prompt": PROMPT,
                 "threshold": 0.15, "max_candidates": 20, "crop_scales": [1.0, 1.5],
                 "models": json.loads((Path(__file__).parent / "model_files.json").read_text())}
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached["signature"] != signature:
            raise ValueError(f"Stale feature cache: {path}; use a new output directory")
        print(f"Reusing {path.name}", flush=True)
        return cached
    check_models(args.weights_dir, ("clip", "dino"))
    clip_path = args.weights_dir / "chinese-clip-vit-base-patch16"
    dino_path = args.weights_dir / "grounding-dino-tiny"
    clip = ChineseCLIPModel.from_pretrained(clip_path, local_files_only=True,
                                            use_safetensors=True).to(args.device).eval()
    cp = ChineseCLIPProcessor.from_pretrained(clip_path, local_files_only=True)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(
        dino_path, local_files_only=True, use_safetensors=True,
        disable_custom_kernels=True).to(args.device).eval()
    dp = AutoProcessor.from_pretrained(dino_path, local_files_only=True, use_fast=False)
    records, pooled, initial = [], [], []
    started = time.perf_counter()
    with ZipFile(args.archive) as data, torch.inference_mode():
        for number, label in enumerate(manifest["splits"][split], 1):
            row = json.loads(data.read(label))
            src = image_name(label, row)
            with Image.open(io.BytesIO(data.read(src))) as opened:
                image = opened.convert("RGB")
            batch = dp(images=image, text=PROMPT, return_tensors="pt").to(args.device)
            prediction = dp.post_process_grounded_object_detection(
                dino(**batch), batch.input_ids, threshold=0.15, text_threshold=0.25,
                target_sizes=[(image.height, image.width)])[0]
            boxes, scores = prediction["boxes"].float().cpu(), prediction["scores"].float().cpu()
            boxes[:, 0::2].clamp_(0, image.width)
            boxes[:, 1::2].clamp_(0, image.height)
            keep = boxes.isfinite().all(1) & scores.isfinite() & (boxes[:, 2:] > boxes[:, :2]).all(1)
            boxes, scores = boxes[keep], scores[keep]
            order = scores.argsort(descending=True, stable=True)[:20]
            boxes, scores = boxes[order], scores[order]
            gt = read_truth(row)
            crops = [image]
            # GT crops supervise training only; holdout features use predictions only.
            for scale in (1.0, 1.5):
                crops.extend(crop(image, box, scale) for box in boxes.tolist())
            if split == "train":
                for scale in (1.0, 1.5):
                    crops.extend(crop(image, box, scale) for box in gt.tolist())
            start = sum(x.shape[0] for x in pooled)
            for offset in range(0, len(crops), 8):
                pixels = cp(images=crops[offset:offset + 8], return_tensors="pt")["pixel_values"].to(args.device)
                vision = clip.vision_model(pixel_values=pixels, return_dict=True).pooler_output
                pooled.append(vision.cpu())
                initial.append(clip.visual_projection(vision).cpu())
            records.append({"label": label, "image": src, "width": image.width, "height": image.height,
                            "boxes": boxes, "scores": scores, "truth": gt, "start": start,
                            "texts": [s["description"].strip() for s in row["shapes"]]})
            print(f"{split} {number}/{len(manifest['splits'][split])}: {len(boxes)} candidates", flush=True)
        texts = list(dict.fromkeys(text for r in records for text in r["texts"]))
        text_features = []
        for offset in range(0, len(texts), 16):
            batch = cp(text=texts[offset:offset + 16], padding=True, truncation=True,
                       max_length=128, return_tensors="pt").to(args.device)
            text_features.append(normalize(clip.get_text_features(**batch), dim=-1).cpu())
    result = {"signature": signature, "records": records, "pooled": torch.cat(pooled),
              "initial": torch.cat(initial), "texts": texts, "text": torch.cat(text_features),
              "device": args.device, "extraction_seconds": round(time.perf_counter() - started, 2)}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return result


def fit(cache, image_ids, mode):
    rows, captions = [], []
    for i in image_ids:
        r = cache["records"][i]
        if mode == "full":
            rows.append(r["start"])
            captions.append(r["texts"])
        else:
            n, m = len(r["boxes"]), len(r["truth"])
            start = r["start"] + 1 + 2 * n + (m if mode == "context" else 0)
            rows.extend(range(start, start + m))
            captions.extend([[text] for text in r["texts"]])
    unique = list(dict.fromkeys(t for group in captions for t in group))
    text_ids = {text: i for i, text in enumerate(unique)}
    all_text_ids = {text: i for i, text in enumerate(cache["texts"])}
    pairs = [(i, text_ids[t]) for i, group in enumerate(captions) for t in group]
    torch.manual_seed(SEED)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = train_adapter(cache["pooled"][rows], cache["initial"][rows],
                                 cache["text"][[all_text_ids[t] for t in unique]], pairs,
                                 rank=16, epochs=80, lr=0.002,
                                 drift_weight=0.25 if mode == "full" else 1.0)
    return model


def visual_features(cache, adapters):
    base = normalize(cache["initial"], dim=-1)
    with torch.no_grad():
        result = {"zero": base}
        for mode, adapter in adapters.items():
            result[mode] = adapter(cache["pooled"], cache["initial"])
    return result


def configuration_grid():
    # This small, fixed grid is evaluated on training folds only.
    yield {"mode": "full", "adapted": True, "threshold": 0.25,
           "detector_weight": 0.0, "region_weight": 0.0, "box_scale": 1.0}
    for mode, adapted, threshold, dw, rw, scale in itertools.product(
            ("tight", "context"), (False, True), (0.15, 0.25),
            (0.0, 0.025, 0.1), (0.5, 1.0), (0.9, 1.0, 1.1)):
        yield {"mode": mode, "adapted": adapted, "threshold": threshold,
               "detector_weight": dw, "region_weight": rw, "box_scale": scale}


def evaluate(cache, ids, features, config, details=False):
    queries = [(i, j, text) for i in ids for j, text in enumerate(cache["records"][i]["texts"])]
    text_ids = {text: i for i, text in enumerate(cache["texts"])}
    unique = list(dict.fromkeys(text for _, _, text in queries))
    # The scoring block reads image features, detector outputs, and text only.
    text = cache["text"][[text_ids[t] for t in unique]]
    lookup = {t: j for j, t in enumerate(unique)}
    scores_by_image, boxes_by_image, coverage_by_image = {}, {}, {}
    for i in ids:
        r = cache["records"][i]
        full = text @ features["full"][r["start"]]
        keep = r["scores"] >= config["threshold"]
        n = len(r["boxes"])
        boxes = scaled_boxes(r["boxes"][keep], config["box_scale"], r["width"], r["height"])
        coverage_by_image[i] = boxes
        if not keep.any():
            scores_by_image[i] = full
            boxes_by_image[i] = torch.zeros((len(unique), 4))
            continue
        if config["mode"] == "full":
            selected = torch.zeros(len(unique), dtype=torch.long)
            scores_by_image[i] = full
        else:
            start = r["start"] + 1 + (n if config["mode"] == "context" else 0)
            kind = config["mode"] if config["adapted"] else "zero"
            local = text @ features[kind][start:start + n][keep].T
            local += config["detector_weight"] * r["scores"][keep].clamp_min(1e-6).log()[None]
            best, selected = local.max(dim=1)
            w = config["region_weight"]
            scores_by_image[i] = (1 - w) * full + w * best
        boxes_by_image[i] = boxes[selected]
    rankings = torch.stack([scores_by_image[i] for i in ids], dim=1).argsort(
        dim=1, descending=True, stable=True)
    local_image = {i: k for k, i in enumerate(ids)}
    ranks, ious, coverage, predictions = [], [], [], []
    for i, j, query in queries:
        q = lookup[query]
        ranks.append(int((rankings[q] == local_image[i]).nonzero()[0, 0]) + 1)
        truth = cache["records"][i]["truth"][j:j + 1]
        box = boxes_by_image[i][q:q + 1]
        iou = float(box_ious(truth, box)[0, 0])
        ious.append(iou)
        candidates = coverage_by_image[i]
        coverage.append(float(box_ious(truth, candidates).max()) if len(candidates) else 0.0)
        if details:
            predictions.append({"image": cache["records"][i]["image"], "annotation_index": j,
                                "text": query, "bbox_xyxy": box[0].tolist(), "iou": iou,
                                "retrieval_rank": ranks[-1]})
    n = len(queries)
    retrieval = {"R@1": sum(r == 1 for r in ranks) / n, "R@5": sum(r <= 5 for r in ranks) / n,
                 "hits_at_1": sum(r == 1 for r in ranks), "hits_at_5": sum(r <= 5 for r in ranks)}
    grounding = summary(ious)
    return {"images": len(ids), "queries": n, "retrieval": retrieval, "grounding": grounding,
            "candidate_coverage": summary(coverage), "predictions": predictions}


def objective(result, task):
    if task == "retrieval":
        return 0.1 * result["retrieval"]["R@1"] + 0.2 * result["retrieval"]["R@5"]
    return (0.6 * result["grounding"]["IoU@0.5"]["rate"]
            + 0.1 * result["grounding"]["IoU@0.75"]["rate"])


def annotation_audit(cache):
    by_text = defaultdict(Counter)
    repeated_image_queries = 0
    for i, record in enumerate(cache["records"]):
        counts = Counter(record["texts"])
        repeated_image_queries += sum(n - 1 for n in counts.values())
        for text, n in counts.items():
            by_text[text][i] += n
    n = sum(sum(counts.values()) for counts in by_text.values())
    return {"unique_texts": len(by_text),
            "texts_shared_by_images": sum(len(counts) > 1 for counts in by_text.values()),
            "repeated_image_text_queries": repeated_image_queries,
            "deterministic_text_only_retrieval_ceiling": {
                f"R@{k}": sum(sum(c for _, c in counts.most_common(k))
                              for counts in by_text.values()) / n for k in (1, 5)}}


def tune(cache, output):
    order = list(range(len(cache["records"])))
    random.Random(SEED).shuffle(order)
    grid = list(configuration_grid())
    totals = [[0.0, 0.0] for _ in grid]
    folds = []
    for fold in range(3):
        val = order[fold::3]
        train = [i for i in order if i not in val]
        adapters = {mode: fit(cache, train, mode) for mode in ("full", "tight", "context")}
        features = visual_features(cache, adapters)
        folds.append({"train": [cache["records"][i]["label"] for i in train],
                      "val": [cache["records"][i]["label"] for i in val]})
        for k, config in enumerate(grid):
            result = evaluate(cache, val, features, config)
            for task_id, task in enumerate(("retrieval", "grounding")):
                totals[k][task_id] += objective(result, task) * result["queries"]
        print(f"Training-fold selection {fold + 1}/3 complete", flush=True)
    chosen = {}
    n = sum(len(r["texts"]) for r in cache["records"])
    for task_id, task in enumerate(("retrieval", "grounding")):
        best = max(range(len(grid)), key=lambda k: totals[k][task_id])
        chosen[task] = grid[best]
        chosen[f"dev_{task}_weighted_contribution"] = totals[best][task_id] / n
    chosen.update({"seed": SEED, "folds": folds, "training_images": len(order),
                   "train_archive_sha256": cache["signature"]["archive"],
                   "grid_size": len(grid), "feature_cache_version": CACHE_VERSION})
    write_json(output / "config.json", chosen)
    write_json(output / "dev_grid.json", [{"config": c, "retrieval_contribution": t[0] / n,
                                           "grounding_contribution": t[1] / n}
                                          for c, t in zip(grid, totals)])
    adapters = {mode: fit(cache, order, mode) for mode in ("full", "tight", "context")}
    for mode, model in adapters.items():
        torch.save(model.state_dict(), output / f"adapter_{mode}.pt")
    print(json.dumps({k: v for k, v in chosen.items() if k != "folds"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("tune", "evaluate"))
    parser.add_argument("--archive", type=Path, default=WORKSPACE / "data/rs_civilian_reviewed.zip")
    parser.add_argument("--weights-dir", type=Path, default=WORKSPACE / "weights")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "experiments/regions_v2")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    args.device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(4)
    manifest = read_reviewed_archive(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    train = extract(args, "train", manifest)
    if args.command == "tune":
        if (args.output / "holdout_results.json").exists():
            parser.error("Holdout was already evaluated; preserve the frozen experiment directory")
        tune(train, args.output)
        return
    config_path = args.output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["train_archive_sha256"] != digest(args.archive):
        raise ValueError("Frozen configuration belongs to a different archive")
    config_sha = digest(config_path)
    val = extract(args, "val", manifest)
    from train_civilian_retrieval import VisualProjectionAdapter
    adapters = {}
    for mode in ("full", "tight", "context"):
        model = VisualProjectionAdapter(train["pooled"].shape[1], train["initial"].shape[1], 16)
        model.load_state_dict(torch.load(args.output / f"adapter_{mode}.pt", weights_only=True))
        adapters[mode] = model.eval()
    features = visual_features(val, adapters)
    ids = list(range(len(val["records"])))
    baseline = evaluate(val, ids, features, next(configuration_grid()))
    retrieval = evaluate(val, ids, features, config["retrieval"], details=True)
    grounding = evaluate(val, ids, features, config["grounding"], details=True)
    result = {"scope": "reviewed civilian passenger-aircraft holdout only",
              "device": args.device, "images": len(ids), "queries": retrieval["queries"],
              "config_sha256": config_sha,
              "annotation_audit": annotation_audit(val),
              "baseline": {k: v for k, v in baseline.items() if k != "predictions"},
              "optimized": {"retrieval": retrieval["retrieval"], "grounding": grounding["grounding"],
                            "candidate_coverage": grounding["candidate_coverage"]},
              "baseline_weighted_score": objective(baseline, "retrieval") + objective(baseline, "grounding"),
              "optimized_weighted_score": objective(retrieval, "retrieval") + objective(grounding, "grounding"),
              "extraction_seconds": val["extraction_seconds"]}
    write_json(args.output / "holdout_results.json", result)
    write_json(args.output / "holdout_predictions.json", {"retrieval": retrieval["predictions"],
                                                          "grounding": grounding["predictions"]})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
