"""Parse model-produced coordinates and select their global units on fit images."""
import json
import re

import torch

from common import ROOT, load_training_cache, split_labels, write_json


def numbers(answer):
    sequences = re.findall(r"[\[({]\s*([-+\d.eE,\s]+)[\])}]", answer)
    for sequence in sequences:
        values = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", sequence)
        if len(values) == 4:
            return [float(v) for v in values]
    values = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", answer)
    return [float(v) for v in values[:4]] if len(values) >= 4 else None


def parse(answer, width, height, units):
    values = numbers(answer)
    if values is None or not all(torch.isfinite(torch.tensor(values))):
        return None
    box = torch.tensor(values, dtype=torch.float32)
    if max(values) <= 1.01 and min(values) >= 0:
        box *= torch.tensor([width, height, width, height])
    elif units != "pixels":
        box *= torch.tensor([width, height, width, height]) / float(units)
    box[0::2].clamp_(0, width)
    box[1::2].clamp_(0, height)
    if not (box[2:] > box[:2]).all():
        return None
    return box


def iou(box, targets):
    if box is None:
        return torch.zeros(len(targets))
    low = torch.maximum(box[:2], targets[:, :2])
    high = torch.minimum(box[2:], targets[:, 2:])
    intersection = (high - low).clamp_min(0).prod(-1)
    area = (box[2:] - box[:2]).prod()
    other = (targets[:, 2:] - targets[:, :2]).prod(-1)
    return intersection / (area + other - intersection).clamp_min(1e-6)


def select_units():
    cache = load_training_cache()
    fitting = set(split_labels()["fit"])
    answers = json.loads((ROOT / "features/geoground_answers.json").read_text(encoding="utf-8"))
    candidates = ("100", "1000", "336", "pixels")
    values = {}
    for units in candidates:
        overlaps = []
        for image_id, row in enumerate(cache["records"]):
            if row["label"] not in fitting:
                continue
            for annotation, text in enumerate(row["texts"]):
                query = cache["texts"].index(text)
                answer = answers[f"{query}:{image_id}"]["answer"]
                box = parse(answer, row["width"], row["height"], units)
                overlaps.append(float(iou(box, row["truth"][annotation:annotation + 1])[0]))
        values[units] = sum(overlaps) / len(overlaps)
    chosen = max(values, key=values.get)
    write_json(ROOT / "results/geoground_coordinate_units.json",
               {"units": chosen, "fit_mean_iou_by_units": values, "selected_using": "fit images only"})
    print(f"GeoGround coordinate units selected from fit images: {chosen}", flush=True)


if __name__ == "__main__":
    select_units()
