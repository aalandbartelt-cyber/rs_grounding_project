"""Download pinned public baseline weights; never execute repository Python code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os

SPECS = {
    "clip": {"repo_id": "OFA-Sys/chinese-clip-vit-base-patch16", "revision": "36e679e65c2a2fead755ae21162091293ad37834",
             "files": ["config.json", "preprocessor_config.json", "vocab.txt", "model.safetensors"]},
    "dino": {"repo_id": "IDEA-Research/grounding-dino-tiny", "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
             "files": ["config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json", "added_tokens.json", "vocab.txt", "model.safetensors"]},
}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", choices=("clip", "dino", "both"), default="both")
    args = parser.parse_args()
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import HfApi, snapshot_download
    report_path = args.out / "provenance.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    for name, spec in SPECS.items():
        if args.only not in (name, "both"):
            continue
        folder = args.out / name
        info = HfApi().model_info(spec["repo_id"], revision=spec["revision"], token=False, files_metadata=True)
        if info.sha != spec["revision"]:
            raise ValueError("Resolved model revision differs from pinned commit")
        available = {item.rfilename: item for item in info.siblings}
        required = [f for f in spec["files"] if f in available]
        if "model.safetensors" not in required:
            raise ValueError("Pinned safetensors weights not found")
        snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"], local_dir=folder,
                          allow_patterns=required, token=False, max_workers=2)
        files = {}
        for filename in required:
            path = folder / filename
            actual = digest(path)
            item = available[filename]
            expected = item.lfs.sha256 if item.lfs else None
            if expected and actual != expected:
                raise ValueError(f"LFS digest mismatch: {name}/{filename}")
            files[filename] = {"sha256": actual, "bytes": path.stat().st_size, "lfs_sha256": expected}
        report[name] = {**spec, "resolved_revision": info.sha, "verified_files": files, "origin": "public_original_weights"}
        args.out.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"VERIFIED {name}: {info.sha}", flush=True)


if __name__ == "__main__":
    main()
