"""Prediction inputs and separately accessed fit-only supervision."""
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn.functional import normalize

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "code"))
from box_parser import iou, parse
from common import load_training_cache, split_labels


class Data:
    def __init__(self):
        self.cache = load_training_cache()
        self.split = split_labels()
        self.records = self.cache["records"]
        self.texts = self.cache["texts"]
        self.fit_ids = [i for i, r in enumerate(self.records) if r["label"] in self.split["fit"]]
        self.dev_ids = [i for i, r in enumerate(self.records) if r["label"] in self.split["dev"]]
        self.images = {}
        self.models = ("RemoteCLIP", "GeoRSCLIP")
        self.text_features, self.visual = {}, {}
        for name in self.models:
            self.visual[name] = torch.load(BASE / f"features/{name}.pt", weights_only=True)
            self.text_features[name] = torch.load(BASE / f"features/{name}_text.pt", weights_only=True)["text"]
        geo_visual = torch.load(BASE / "training_v1/features/geoground_visual.pt", weights_only=True)
        geo = self.read("features/geoground_answers.json")
        qwen = self.read("features/qwen_grounding.json")
        units = self.read("results/geoground_coordinate_units.json")["units"]
        jobs = self.read("features/jobs.json")["grounding"]
        self.jobs = {}
        for job in jobs:
            key = f"{job['query']}:{job['image']}"
            self.jobs[key] = {k: job[k] for k in ("query", "image")}
        for i, row in enumerate(self.records):
            n = len(row["boxes"])
            streams, whole = [], []
            for name in self.models:
                visual = self.visual[name]
                metadata = visual["records"][i]
                assert row["label"] == metadata["label"] and n == metadata["candidates"]
                start = metadata["start"]
                whole.append(visual["visual"][start])
                streams.append((visual["visual"][start + 1:start + 1 + n] +
                                visual["visual"][start + 1 + n:start + 1 + 2 * n]) / 2)
            self.images[i] = {"whole": whole, "regions": streams}
        self.grounding = {}
        for key, job in self.jobs.items():
            q, i = job["query"], job["image"]
            row = self.records[i]
            boxes = row["boxes"].clone()
            center = (boxes[:, :2] + boxes[:, 2:]) / 2
            half = (boxes[:, 2:] - boxes[:, :2]) * .45
            boxes = torch.cat([center - half, center + half], dim=1)
            scores = row["scores"].clone()
            streams = [v.clone() for v in self.images[i]["regions"]]
            region = parse(geo[key]["answer"], row["width"], row["height"], units)
            is_geo = torch.zeros(len(boxes))
            if region is not None:
                boxes = torch.cat([boxes, region[None]])
                scores = torch.cat([scores, scores.max()[None] if len(scores) else torch.tensor([.1])])
                streams = [torch.cat([v, geo_visual[name][key][None]]) for name, v in zip(self.models, streams)]
                is_geo = torch.cat([is_geo, torch.ones(1)])
            chosen = torch.zeros(len(boxes))
            if 0 <= qwen[key]["chosen"] < len(boxes):
                chosen[qwen[key]["chosen"]] = 1
            agreement = iou(region, boxes)
            whole = self.images[i]["whole"]
            base = self.features(q, i, boxes, scores, streams, whole, is_geo, chosen, agreement)
            base.update({"boxes": boxes, "image": i, "query": q})
            self.grounding[key] = base
        self.retrieval = self.retrieval_inputs()
        semantic_path = BASE / "training_v1/features/qwen_semantic.json"
        if semantic_path.exists():
            semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
            if set(semantic["grounding"]) != set(self.grounding):
                raise ValueError("Qwen regional evidence is incomplete")
            for key, row in self.grounding.items():
                logits = torch.tensor(semantic["grounding"][key]).tanh()
                assert len(logits) == len(row["boxes"])
                extra = torch.stack([logits,logits-logits.max()],1)
                for kind in ("scalar","vision","conditioned"):
                    row[kind] = torch.cat([row[kind],extra],1)
            for key,row in self.retrieval.items():
                # Queries not present in a gallery do not need model inference.
                raw = semantic["retrieval"].get(key)
                value = torch.tensor([raw / 5 if raw is not None else 0.]).tanh()
                for kind in ("scalar","vision","conditioned"):
                    row[kind] = torch.cat([row[kind],value])

    @staticmethod
    def read(relative):
        return json.loads((BASE / relative).read_text(encoding="utf-8"))

    def features(self, query, image_id, boxes, confidence, streams, whole, is_geo, chosen, agreement):
        row = self.records[image_id]
        n = len(boxes)
        text = [self.text_features[name][query] for name in self.models]
        similarities = [v @ t for v, t in zip(streams, text)]
        whole_similarity = [torch.full((n,), float(v @ t)) for v, t in zip(whole, text)]
        sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1)
        normalized = boxes / torch.tensor([row["width"], row["height"], row["width"], row["height"]])
        centers = (normalized[:, :2] + normalized[:, 2:]) / 2
        dimensions = normalized[:, 2:] - normalized[:, :2]
        scalar = torch.cat([torch.stack(similarities + whole_similarity, 1),
                            confidence.clamp_min(1e-6).log()[:, None], centers, dimensions,
                            dimensions.prod(1)[:, None], (sizes[:, 0] / sizes[:, 1]).log()[:, None],
                            is_geo[:, None], chosen[:, None], agreement[:, None],
                            (similarities[0] - similarities[0].max())[:, None],
                            (similarities[1] - similarities[1].max())[:, None]], 1)
        visual = torch.cat(streams, 1)
        interaction = torch.cat([v * t for v, t in zip(streams, text)], 1)
        return {"scalar": scalar, "vision": torch.cat([visual, scalar], 1),
                "conditioned": torch.cat([visual, interaction, scalar], 1),
                "bbox_features": torch.cat([visual, scalar[:, 5:11], is_geo[:, None]], 1),
                "base_score": .5 * (similarities[0] + similarities[1]) + .1 * confidence.clamp_min(1e-6).log()
                              + .05 * chosen + .05 * agreement}

    def retrieval_inputs(self):
        results = {}
        for q in range(len(self.texts)):
            for i, row in enumerate(self.records):
                whole, streams = self.images[i]["whole"], self.images[i]["regions"]
                txt = [self.text_features[name][q] for name in self.models]
                global_sim = torch.tensor([float(v @ t) for v, t in zip(whole, txt)])
                if len(row["boxes"]):
                    local = torch.stack([v @ t for v, t in zip(streams, txt)], 1)
                    adjusted = local.mean(1) + .1 * row["scores"].clamp_min(1e-6).log()
                    best = int(adjusted.argmax())
                    region = [v[best] for v in streams]
                    regional_sim = local[best]
                    confidence = row["scores"][best].clamp_min(1e-6).log()
                else:
                    region, regional_sim, confidence = whole, global_sim, torch.tensor(0.)
                scalar = torch.cat([global_sim, regional_sim, confidence[None]])
                visual = torch.cat(whole + region)
                interaction = torch.cat([v * t for v, t in zip(whole + region, txt + txt)])
                results[f"{q}:{i}"] = {"scalar": scalar,
                    "vision": torch.cat([visual, scalar]),
                    "conditioned": torch.cat([visual, interaction, scalar]),
                    "base_score": float(.5 * global_sim.mean() + .5 * (regional_sim.mean() + .1 * confidence))}
        return results

    def truths(self, image, query, training_ids=None):
        if training_ids is not None:
            assert set(training_ids) <= set(self.fit_ids), "Training attempted to use internal-dev labels"
            assert image in training_ids, "Image is outside this training fold"
        row = self.records[image]
        ids = [j for j, t in enumerate(row["texts"]) if t == self.texts[query]]
        return row["truth"][ids]

    def keys(self, ids):
        return [key for key, job in self.jobs.items() if job["image"] in ids]

    def folds(self):
        lookup = {r["label"]: i for i, r in enumerate(self.records)}
        groups = [[lookup[label] for label in group] for group in self.split["overlap_groups"]
                  if set(group) <= set(self.split["fit"])]
        groups.sort(key=lambda g: -sum(len(self.records[i]["texts"]) for i in g))
        folds = [[], [], []]
        for group in groups:
            target = min(range(3), key=lambda j: sum(len(self.records[i]["texts"]) for i in folds[j]))
            folds[target].extend(group)
        return folds


def encode_deltas(boxes, targets):
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1)
    center = (boxes[:, 2:] + boxes[:, :2]) / 2
    target_wh = (targets[:, 2:] - targets[:, :2]).clamp_min(1)
    target_center = (targets[:, 2:] + targets[:, :2]) / 2
    return torch.cat([(target_center - center) / wh, (target_wh / wh).log()], 1)


def decode_deltas(boxes, deltas, width, height):
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1)
    center = (boxes[:, 2:] + boxes[:, :2]) / 2
    center = center + deltas[:, :2].clamp(-.2, .2) * wh
    wh = wh * deltas[:, 2:].clamp(-.4, .4).exp()
    result = torch.cat([center - wh / 2, center + wh / 2], 1)
    result[:, 0::2].clamp_(0, width)
    result[:, 1::2].clamp_(0, height)
    return result
