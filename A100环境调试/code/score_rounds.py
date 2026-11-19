"""Print competition-weighted totals and export the measured rounds."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
from zipfile import ZIP_DEFLATED, ZipFile


FORMULA = "0.1×R@1 + 0.2×R@5 + 0.6×Acc@0.5 + 0.1×Acc@0.75"
WEIGHTS = {"R@1": 0.1, "R@5": 0.2, "Acc@0.5": 0.6, "Acc@0.75": 0.1}
CLIP_PATH = "experiments/S_clip_a100/results.json"
DINO_PATH = "experiments/M_dino_val/metrics.json"
DINO_CANDIDATES = "experiments/M_dino_val/candidates.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def score_row(name, device, retrieval, grounding, n_retrieval, n_grounding):
    metrics = {"R@1": retrieval["R@1"], "R@5": retrieval["R@5"],
               "Acc@0.5": grounding["IoU@0.5"]["rate"],
               "Acc@0.75": grounding["IoU@0.75"]["rate"]}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in metrics.values()):
        raise ValueError(f"Expected all four rates to be fractions in [0, 1]: {name}")
    return {"round": name, "device": device, "retrieval_queries": n_retrieval,
            "grounding_queries": n_grounding, **metrics,
            "total_score": math.fsum(metrics[key] * weight for key, weight in WEIGHTS.items())}


def baseline_rows(results, metrics):
    if (results["data_split"] != metrics["data_split"] or
            results["validation_images"] != metrics["images"] or
            results["validation_queries"] != metrics["annotations"]):
        raise ValueError("First-round retrieval and grounding must cover the same validation split")
    device = "A100" if results["device"] == metrics["device"] == "cuda" else (
        f"{results['device']} / {metrics['device']}")
    return [score_row(name, device, results[key], metrics["top_box"],
                      results["validation_queries"], metrics["annotations"])
            for key, name in (("zero_shot", "第 1 轮 · 零样本检索 + 基线定位"),
                              ("adapted", "第 1 轮 · 微调检索 + 基线定位"))]


def region_rows(run):
    report = read_json(run / "holdout_results.json")
    device = "A100" if report["device"] == "cuda" else report["device"].upper()
    rows = [score_row(name, device, report[key]["retrieval"], report[key]["grounding"],
                      report["queries"], report["queries"])
            for key, name in (("baseline", "第 2 轮 · 同轮复训基线"),
                              ("optimized", "第 2 轮 · 区域检索 + 改进定位"))]
    preserved = read_json(run / "retained_retrieval_results.json")
    rows.append(score_row("第 2 轮 · 首轮检索 + 改进定位",
                          device, preserved["recommended"]["retrieval"],
                          preserved["recommended"]["grounding"], preserved["queries"], report["queries"]))
    return rows


def format_summary(rows):
    lines = ["遥感文搜图定位 · 民航留出集", f"总分 = {FORMULA}（满分 1）", ""]
    for row in rows:
        counts = f"检索 {row['retrieval_queries']} 条，定位 {row['grounding_queries']} 条"
        lines.append(f"{row['round']}  |  {row['device']}  |  {counts}")
        lines.append("  " + "  ".join(f"{key} {row[key]:.2%}" for key in WEIGHTS)
                     + f"  总分 {row['total_score']:.4f}/1")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--baseline-tar", type=Path, help="Read the first A100 results directly from its tar.gz")
    parser.add_argument("--round-dir", type=Path, help="Include second-round holdout results from this directory")
    parser.add_argument("--zip", type=Path, required=True, help="Output ZIP, with the score summary and raw results")
    args = parser.parse_args()
    rows, files = [], {}
    if args.baseline_tar:
        with tarfile.open(args.baseline_tar) as old:
            for entry in (CLIP_PATH, DINO_PATH, DINO_CANDIDATES,
                          "experiments/S_clip_a100/adapter.safetensors"):
                with old.extractfile(entry) as source:
                    files[entry] = source.read()
        rows.extend(baseline_rows(json.loads(files[CLIP_PATH]), json.loads(files[DINO_PATH])))
    elif (args.root / CLIP_PATH).is_file() and (args.root / DINO_PATH).is_file():
        for entry in (CLIP_PATH, DINO_PATH, DINO_CANDIDATES,
                      "experiments/S_clip_a100/adapter.safetensors"):
            if (args.root / entry).is_file():
                files[entry] = (args.root / entry).read_bytes()
        rows.extend(baseline_rows(json.loads(files[CLIP_PATH]), json.loads(files[DINO_PATH])))
    if args.round_dir:
        rows.extend(region_rows(args.round_dir))
        for name in ("config.json", "dev_grid.json", "holdout_results.json", "holdout_predictions.json",
                     "recommended_config.json", "retained_retrieval_results.json", "environment.json",
                     "adapter_full.pt", "adapter_tight.pt", "adapter_context.pt"):
            path = args.round_dir / name
            if path.is_file():
                files[f"experiments/{args.round_dir.name}/{name}"] = path.read_bytes()
        retained = read_json(args.round_dir / "recommended_config.json")["retrieval"]
        adapter = (args.root / retained["adapter_path"]).resolve()
        if adapter.is_file() and adapter.is_relative_to(args.root.resolve()):
            data = adapter.read_bytes()
            if hashlib.sha256(data).hexdigest() != retained["adapter_sha256"]:
                raise ValueError("The retained retrieval checkpoint changed after evaluation")
            files[adapter.relative_to(args.root.resolve()).as_posix()] = data
    if not rows:
        parser.error("No complete round found. First-round totals require both retrieval and grounding metrics.")
    summary = format_summary(rows)
    scores = {"scope": "reviewed civilian holdout", "max_score": 1,
              "formula": FORMULA, "weights": WEIGHTS, "rounds": rows}
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.zip, "w", compression=ZIP_DEFLATED) as out:
        out.writestr("score_summary.txt", summary)
        out.writestr("score_summary.json", json.dumps(scores, ensure_ascii=False, indent=2) + "\n")
        for name, data in files.items():
            out.writestr(name, data)
    print(summary, end="")
    print(f"结果压缩包：{args.zip.resolve()}")


if __name__ == "__main__":
    main()
