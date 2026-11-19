"""Measure the unchanged first-round retrieval checkpoint with v2 grounding."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from benchmark_regions import objective, write_json
from prepare_check import WORKSPACE, check_models, digest, read_reviewed_archive
from train_civilian_retrieval import VisualProjectionAdapter, encode_split, measure_retrieval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=WORKSPACE / "data/rs_civilian_reviewed.zip")
    parser.add_argument("--weights-dir", type=Path, default=WORKSPACE / "weights")
    parser.add_argument("--reference-adapter", type=Path,
                        default=WORKSPACE / "experiments/original_reference/adapter.safetensors")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "experiments/regions_v2")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    manifest = read_reviewed_archive(args.archive)
    check_models(args.weights_dir, ("clip",))
    frozen = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
    paired = json.loads((args.output / "holdout_results.json").read_text(encoding="utf-8"))
    if paired["config_sha256"] != digest(args.output / "config.json"):
        raise ValueError("Frozen config changed after holdout evaluation")
    torch.set_num_threads(6)
    weights = args.weights_dir / "chinese-clip-vit-base-patch16"
    model = ChineseCLIPModel.from_pretrained(weights, local_files_only=True,
                                            use_safetensors=True).to(args.device).eval()
    processor = ChineseCLIPProcessor.from_pretrained(weights, local_files_only=True)
    state = load_file(str(args.reference_adapter))
    adapter = VisualProjectionAdapter(state["down.weight"].shape[1],
                                      state["up.weight"].shape[0], state["up.weight"].shape[1])
    adapter.load_state_dict(state)
    adapter = adapter.to(args.device).eval()
    with torch.inference_mode():
        pooled, initial, text, pairs = encode_split(args.archive, manifest["splits"]["val"],
                                                   model, processor, torch.device(args.device))
        retrieval = measure_retrieval(adapter(pooled, initial), text, pairs)
    # A retained checkpoint is evaluated as-is. No fit or parameter search here.
    old = {"retrieval": retrieval, "grounding": paired["baseline"]["grounding"]}
    retained = {"retrieval": retrieval, "grounding": paired["optimized"]["grounding"]}
    adapter_path = args.reference_adapter.resolve()
    adapter_sha = digest(adapter_path)
    config = {"retrieval": {"mode": "unchanged_first_round_whole_image",
                             "adapter_sha256": adapter_sha,
                             "adapter_path": str(adapter_path.relative_to(WORKSPACE)
                                                 if adapter_path.is_relative_to(WORKSPACE) else adapter_path)},
              "grounding": frozen["grounding"], "frozen_region_config_sha256": paired["config_sha256"]}
    result = {"scope": paired["scope"], "images": paired["images"], "queries": len(pairs),
              "device": args.device, "reference_adapter_sha256": adapter_sha,
              "baseline": old, "recommended": retained,
              "baseline_weighted_score": objective(old, "retrieval") + objective(old, "grounding"),
              "recommended_weighted_score": objective(retained, "retrieval") + objective(retained, "grounding"),
              "regional_retrieval": paired["optimized"]["retrieval"],
              "decision": "Retain the existing retrieval checkpoint after a failed regional retrieval experiment; adopt v2 grounding.",
              "note": "This is an engineering rollback after holdout observation, not a new independent holdout result."}
    write_json(args.output / "recommended_config.json", config)
    write_json(args.output / "retained_retrieval_results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
