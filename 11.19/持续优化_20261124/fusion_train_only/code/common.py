"""Train-only civilian data access and shared experiment records."""
import hashlib
import io
import json
import random
import sys
from pathlib import Path
from zipfile import ZipFile

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = Path("E:/A大三/智元-遥感文搜图")
SOURCE_MANIFEST = PREVIOUS / "optimization_v2/code/civilian_retrieval_split.json"
ARCHIVE = PREVIOUS / "optimization_v2/data/rs_civilian_reviewed.zip"
LOCAL_ARCHIVE = ROOT / "data/original_train_only.zip"
LOCAL_MANIFEST = ROOT / "data/manifest.json"


def deny_network(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise OSError("Networking is disabled in train-only model inference")


sys.addaudithook(deny_network)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def training_labels():
    if LOCAL_MANIFEST.exists():
        labels = json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8"))["training_labels"]
    else:
        labels = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))["splits"]["train"]
    if len(labels) != 49 or len(set(labels)) != len(labels):
        raise ValueError("Unexpected reviewed training image list")
    if any("/train/labels_json/" not in label for label in labels):
        raise ValueError("Only original public-training images are permitted")
    return labels


class TrainArchive:
    """Only whitelisted original train images and annotations can be opened."""
    def __init__(self):
        self.labels = training_labels()
        self.allowed = set(self.labels)
        self.data = ZipFile(LOCAL_ARCHIVE if LOCAL_ARCHIVE.exists() else ARCHIVE)
        self.rows = {}
        self.accesses = []
        for label in self.labels:
            row = json.loads(self.read(label))
            image = label.split("/labels_json/")[0] + "/images/" + row["imagePath"]
            if "/train/images/" not in image:
                raise ValueError("Invalid training image path")
            self.allowed.add(image)
            self.rows[label] = row

    def read(self, name):
        if name not in self.allowed:
            raise PermissionError(f"Data access is outside the training whitelist: {name}")
        self.accesses.append(name)
        return self.data.read(name)

    def image(self, label):
        row = self.rows[label]
        name = label.split("/labels_json/")[0] + "/images/" + row["imagePath"]
        with Image.open(io.BytesIO(self.read(name))) as opened:
            return opened.convert("RGB")

    def audit(self, path):
        write_json(path, {"allowed_original_split": "train",
                          "annotation_files": self.labels,
                          "accessed_entries": sorted(set(self.accesses)),
                          "all_accesses_within_training_whitelist": set(self.accesses) <= self.allowed,
                          "holdout_or_test_entries_accessed": []})

    def close(self):
        self.data.close()


def load_training_cache():
    # Only immutable pretrained features are reused. Previous trained adapters are excluded.
    local = ROOT / "data/pretrained_train_features.pt"
    source = local if local.exists() else PREVIOUS / "optimization_v2/experiments/regions_v2/features_train.pt"
    cache = torch.load(source, weights_only=True, map_location="cpu")
    if [row["label"] for row in cache["records"]] != training_labels():
        raise ValueError("Feature cache is not the reviewed original training split")
    return cache


def split_labels():
    path = ROOT / "results/split.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        combined = result["fit"] + result["dev"]
        if set(combined) != set(training_labels()) or len(combined) != len(set(combined)):
            raise ValueError("Internal split no longer matches the training whitelist")
        return result
    order = training_labels()
    random.Random(20261123).shuffle(order)
    result = {"seed": 20261123, "source_split": "original train only",
              "fit": order[15:], "dev": order[:15],
              "description": "One fixed internal development round, no official validation or test images"}
    write_json(path, result)
    return result


def crop(image, box, scale):
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    return image.crop((max(0, int(cx - rx)), max(0, int(cy - ry)),
                       min(image.width, int(cx + rx + 0.999)), min(image.height, int(cy + ry + 0.999))))
