import argparse
import io
import json
import time
from pathlib import Path
from zipfile import ZipFile

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from prepare_check import WORKSPACE, check_models, experiment_splits, image_name, read_reviewed_archive


def box_ious(truth, boxes):
    top_left = torch.maximum(truth[:, None, :2], boxes[None, :, :2])
    bottom_right = torch.minimum(truth[:, None, 2:], boxes[None, :, 2:])
    overlap = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    truth_area = (truth[:, 2:] - truth[:, :2]).prod(dim=-1)
    box_area = (boxes[:, 2:] - boxes[:, :2]).prod(dim=-1)
    return overlap / (truth_area[:, None] + box_area[None, :] - overlap).clamp(min=1e-6)


def read_truth(row):
    boxes = []
    for shape in row["shapes"]:
        xs, ys = zip(*shape["points"])
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def summary(values):
    return {
        f"IoU@{threshold}": {
            "hits": sum(value >= threshold for value in values),
            "total": len(values),
            "rate": sum(value >= threshold for value in values) / len(values),
        }
        for threshold in (0.5, 0.75)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate candidate boxes on the reviewed civilian subset.")
    parser.add_argument("--archive", type=Path, default=WORKSPACE / "data" / "rs_civilian_reviewed.zip")
    parser.add_argument("--weights-dir", type=Path, default=WORKSPACE / "weights")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "experiments" / "M_dino_pilot")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dev-manifest", type=Path)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    args = parser.parse_args()
    if args.limit < 1 or args.max_candidates < 1:
        parser.error("limit and max-candidates must be positive")
    if not (0 <= args.box_threshold <= 1 and 0 <= args.text_threshold <= 1):
        parser.error("thresholds must be in [0, 1]")

    manifest = read_reviewed_archive(args.archive)
    splits, split_name = experiment_splits(manifest, args.dev_manifest)
    check_models(args.weights_dir, ("dino",))
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(4)

    weights = args.weights_dir / "grounding-dino-tiny"
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(weights, local_files_only=True, use_fast=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        weights, local_files_only=True, use_safetensors=True, disable_custom_kernels=True,
    ).to(device).eval()
    print(f"Loaded GroundingDINO on {device}.", flush=True)

    files = splits["val"][:args.limit]
    top_box, coverage, candidates = [], [], []
    empty_images = 0
    with ZipFile(args.archive) as data, torch.inference_mode():
        for number, path in enumerate(files, 1):
            row = json.loads(data.read(path))
            src = image_name(path, row)
            with Image.open(io.BytesIO(data.read(src))) as opened:
                image = opened.convert("RGB")

            batch = processor(images=image, text="civilian passenger airliner.", return_tensors="pt").to(device)
            out = processor.post_process_grounded_object_detection(
                model(**batch), batch.input_ids, threshold=args.box_threshold,
                text_threshold=args.text_threshold, target_sizes=[(image.height, image.width)],
            )[0]
            boxes = out["boxes"].float().cpu()
            boxes[:, 0::2].clamp_(0, image.width)
            boxes[:, 1::2].clamp_(0, image.height)
            keep = (boxes[:, 2:] > boxes[:, :2]).all(dim=1) & boxes.isfinite().all(dim=1)
            boxes, scores = boxes[keep], out["scores"].float().cpu()[keep]
            ranking = scores.argsort(descending=True, stable=True)[:args.max_candidates]
            boxes, scores = boxes[ranking], scores[ranking]
            gt = read_truth(row)
            if len(boxes):
                ious = box_ious(gt, boxes)
                top_box.extend(ious[:, 0].tolist())
                coverage.extend(ious.amax(dim=1).tolist())
            else:
                empty_images += 1
                top_box.extend([0.0] * len(gt))
                coverage.extend([0.0] * len(gt))
            candidates.append({"image": src, "boxes_xyxy": boxes.tolist(), "scores": scores.tolist()})
            print(f"{number}/{len(files)}: {len(boxes)} candidate boxes, {len(gt)} annotations", flush=True)

    if not top_box:
        raise ValueError("Selected images have no annotations to evaluate.")
    result = {
        "data_split": split_name,
        "images": len(files),
        "annotations": len(top_box),
        "empty_images": empty_images,
        "device": device,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "max_candidates": args.max_candidates,
        "top_box": summary(top_box),
        "candidate_coverage": summary(coverage),
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "candidates.json").write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
    report = json.dumps(result, indent=2)
    (args.output / "metrics.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
