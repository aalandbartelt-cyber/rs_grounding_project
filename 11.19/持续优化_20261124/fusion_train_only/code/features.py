"""Prediction features and input jobs for the fixed original-training split."""
import json

import torch

from common import ROOT, load_training_cache, split_labels, write_json


def model_scores(name, cache):
    features = torch.load(ROOT / f"features/{name}.pt", weights_only=True, map_location="cpu")
    text = torch.load(ROOT / f"features/{name}_text.pt", weights_only=True, map_location="cpu")
    if text["texts"] != cache["texts"]:
        raise ValueError("Query feature order differs from original-training text")
    whole, regional = [], []
    for row, record in zip(cache["records"], features["records"]):
        if row["label"] != record["label"]:
            raise ValueError("Image feature order differs from original-training images")
        start, n = record["start"], record["candidates"]
        whole.append(text["text"] @ features["visual"][start])
        tight = text["text"] @ features["visual"][start + 1:start + 1 + n].T
        context = text["text"] @ features["visual"][start + 1 + n:start + 1 + 2 * n].T
        regional.append(0.5 * (tight + context))
    return torch.stack(whole, dim=1), regional


def score_features():
    cache = load_training_cache()
    scores = {name: model_scores(name, cache) for name in ("RemoteCLIP", "GeoRSCLIP")}
    return cache, scores


def base_scores(cache, scores):
    whole = 0.5 * (scores["RemoteCLIP"][0] + scores["GeoRSCLIP"][0])
    by_image, selected = [], []
    for i, row in enumerate(cache["records"]):
        regional = 0.5 * (scores["RemoteCLIP"][1][i] + scores["GeoRSCLIP"][1][i])
        if len(row["boxes"]):
            local = regional + 0.1 * row["scores"].clamp_min(1e-6).log()
            best, index = local.max(1)
            by_image.append(0.5 * whole[:, i] + 0.5 * best)
            selected.append(index)
        else:
            by_image.append(whole[:, i])
            selected.append(torch.zeros(len(cache["texts"]), dtype=torch.long))
    return torch.stack(by_image, 1), selected


def make_jobs():
    cache, scores = score_features()
    lookup = {row["label"]: i for i, row in enumerate(cache["records"])}
    text_ids = {t: j for j, t in enumerate(cache["texts"])}
    base, selected = base_scores(cache, scores)
    split = split_labels()
    retrieval_jobs, grounding_jobs = [], []
    for role in ("fit", "dev"):
        ids = [lookup[label] for label in split[role]]
        texts = list(dict.fromkeys(t for i in ids for t in cache["records"][i]["texts"]))
        for text in texts:
            query = text_ids[text]
            ranking = base[query, ids].argsort(dim=0, descending=True, stable=True)[:5]
            for image in ranking.tolist():
                image_id = ids[image]
                retrieval_jobs.append({"role": role, "query": query, "image": image_id,
                                       "text": text, "label": cache["records"][image_id]["label"],
                                       "candidate": int(selected[image_id][query])})
        for image_id in ids:
            # Grounding queries explicitly supply their input image, as required by the task.
            for text in dict.fromkeys(cache["records"][image_id]["texts"]):
                grounding_jobs.append({"role": role, "query": text_ids[text], "image": image_id,
                                       "text": text, "label": cache["records"][image_id]["label"]})
    write_json(ROOT / "features/jobs.json", {"retrieval": retrieval_jobs, "grounding": grounding_jobs,
                                            "retrieval_shortlist_size": 5,
                                            "no_grounding_assignments_used_for_retrieval": True})
    print(f"Prepared {len(retrieval_jobs)} retrieval reranks and {len(grounding_jobs)} grounding inputs", flush=True)


if __name__ == "__main__":
    make_jobs()
