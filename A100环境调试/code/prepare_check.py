import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
REVIEWED_SPLIT = "c197d2ee3c6cb30bf1cb29fc27c016e0f2bbed3a9b1b03bec64ce61cd1b2592e"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_reviewed_archive(archive):
    manifest = json.loads((ROOT / "civilian_retrieval_split.json").read_text(encoding="utf-8"))
    serial = json.dumps(manifest["splits"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(serial).hexdigest() != REVIEWED_SPLIT:
        raise ValueError("The image lists must match the reviewed civilian split.")
    if digest(archive) != manifest["archive_sha256"]:
        raise ValueError(f"Archive SHA-256 mismatch: {archive}")
    return manifest


def experiment_splits(manifest, dev_path):
    if dev_path is None:
        return manifest["splits"], "holdout"

    selected = json.loads(dev_path.read_text(encoding="utf-8"))
    approved = set(manifest["splits"]["train"])
    if not isinstance(selected, dict) or selected.keys() != {"train", "val"}:
        raise ValueError("Dev manifest must have 'train' and 'val' lists.")
    train, val = selected["train"], selected["val"]
    if not isinstance(train, list) or not isinstance(val, list) or not train or len(val) < 5:
        raise ValueError("Dev split needs a non-empty training set and at least 5 validation images.")
    combined = train + val
    if len(combined) != len(set(combined)) or not set(combined) <= approved:
        raise ValueError("Dev images must be disjoint, unique, and drawn from reviewed training images.")
    return selected, "dev"


def image_name(label, row):
    return label.split("/labels_json/", 1)[0] + "/images/" + row["imagePath"]


def inspect_data(archive, manifest):
    expected = set()
    counts = {}
    with ZipFile(archive) as data:
        names = data.namelist()
        for split, labels in manifest["splits"].items():
            descriptions = 0
            for label in labels:
                row = json.loads(data.read(label))
                image = image_name(label, row)
                if image in expected:
                    raise ValueError(f"Repeated image in reviewed archive: {image}")
                expected.update((label, image))
                descriptions += sum(bool(item["description"].strip()) for item in row["shapes"])
            counts[split] = {"images": len(labels), "descriptions": descriptions}
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("The review archive must contain exactly the listed label and image files.")
        damaged = data.testzip()
        if damaged:
            raise ValueError(f"Invalid ZIP entry: {damaged}")
    return counts


def check_models(weights_dir, model_keys):
    specs = json.loads((ROOT / "model_files.json").read_text(encoding="utf-8"))
    for key in model_keys:
        spec = specs[key]
        folder = weights_dir / spec["directory"]
        for name in spec["files"]:
            if not (folder / name).is_file() or (folder / name).stat().st_size == 0:
                raise FileNotFoundError(f"Model file missing or empty: {folder / name}")
        weight = folder / spec["weight_file"]
        if digest(weight) != spec["weight_sha256"]:
            raise ValueError(f"Model SHA-256 mismatch: {weight}")
        print(f"{key}: verified {spec['revision']}")


def main():
    parser = argparse.ArgumentParser(description="Check the reviewed data, model files, and optionally CUDA.")
    parser.add_argument("--archive", type=Path, default=WORKSPACE / "data" / "rs_civilian_reviewed.zip")
    parser.add_argument("--weights-dir", type=Path, default=WORKSPACE / "weights")
    parser.add_argument("--models", choices=("clip", "dino", "all"))
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--reuse-runtime", action="store_true", help="Use the CUDA PyTorch supplied by the current notebook.")
    args = parser.parse_args()

    manifest = read_reviewed_archive(args.archive)
    print(json.dumps(inspect_data(args.archive, manifest), indent=2))
    if args.models:
        selected = ("clip", "dino") if args.models == "all" else (args.models,)
        check_models(args.weights_dir, selected)
    if args.cuda:
        import torch

        if not args.reuse_runtime:
            if sys.version_info[:2] != (3, 10) or torch.__version__.split("+")[0] != "2.3.1":
                raise SystemExit(f"Activate the project Python 3.10 / Torch 2.3.1 environment. Got {sys.version.split()[0]} / {torch.__version__}")
            if torch.version.cuda != "12.1":
                raise SystemExit(f"The cu121 Torch package is required for this runbook, found CUDA {torch.version.cuda}.")
        if not torch.cuda.is_available():
            raise SystemExit(f"CUDA unavailable with Torch {torch.__version__} and CUDA {torch.version.cuda}")
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise SystemExit("Expected one assigned A100 GPU.")
        sample = torch.ones(64, 64, device="cuda")
        if (sample @ sample)[0, 0].item() != 64.0:
            raise RuntimeError("CUDA matrix multiplication failed")
        print(f"CUDA OK: {torch.cuda.get_device_name(0)} | Torch {torch.__version__} | CUDA {torch.version.cuda}")


if __name__ == "__main__":
    main()
