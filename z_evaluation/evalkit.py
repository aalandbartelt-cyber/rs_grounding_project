"""Reproducible local retrieval/grounding evaluation; Python 3.10+, stdlib only."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import shutil
import struct
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "rs_grounding_train_v1"
VERSION = "2.0.0"
SEED = "rs-grounding-dev4-v2-20261121"
QUOTAS = {1: 9, 2: 9, 3: 3, 4: 9}
WEIGHTS = {"R1": 0.1, "R5": 0.2, "Acc05": 0.6, "Acc75": 0.1}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized(text):
    # Diagnostic grouping only. Queries retain the original annotation text.
    return " ".join(unicodedata.normalize("NFKC", text).split())


def number(v):
    return type(v) in (int, float) and math.isfinite(v)


def integer(v):
    return type(v) is int


def jpeg_size(path):
    data = Path(path).read_bytes()
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"Not a JPEG: {path}")
    i = 2
    while i < len(data):
        if data[i] != 255:
            raise ValueError(f"Invalid JPEG marker: {path}")
        while data[i] == 255:
            i += 1
        marker = data[i]
        i += 1
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if marker in (0xD9, 0xDA):
            break
        length = struct.unpack(">H", data[i:i + 2])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            height, width = struct.unpack(">HH", data[i + 3:i + 7])
            return width, height
        i += length
    raise ValueError(f"JPEG dimensions not found: {path}")


def bbox_from_points(points):
    if not isinstance(points, list) or len(points) not in (2, 4):
        raise ValueError("Rectangle requires two or four points")
    if any(not isinstance(p, list) or len(p) != 2 or not all(number(v) for v in p) for p in points):
        raise ValueError("Invalid rectangle coordinates")
    xs, ys = zip(*points)
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Non-positive rectangle area")
    if len(points) == 4 and set(map(tuple, points)) != {(x1, y1), (x1, y2), (x2, y1), (x2, y2)}:
        raise ValueError("Four points are not an axis-aligned rectangle")
    return [x1, y1, x2 - x1, y2 - y1]


def source_inventory(data):
    return {str(f.relative_to(data)).replace("\\", "/"): sha(f)
            for f in sorted(data.rglob("*")) if f.is_file()}


def load_data(data):
    mapping = read(data / "label_map.json")
    if sorted(mapping.values()) != [1, 2, 3, 4]:
        raise ValueError("Expected category IDs 1..4")
    records, warnings = [], []
    for split in ("train", "val"):
        images = {p.stem: p for p in (data / split / "images").glob("*.jpg")}
        labels = {p.stem: p for p in (data / split / "labels_json").glob("*.json")}
        xmls = {p.stem: p for p in (data / split / "labels_xml").glob("*.xml")}
        if not images or set(images) != set(labels) or set(images) != set(xmls):
            raise ValueError(f"Image/JSON/XML filenames disagree in {split}")
        for stem in sorted(images):
            obj = read(labels[stem])
            width, height = jpeg_size(images[stem])
            if (obj["imageWidth"], obj["imageHeight"]) != (width, height):
                raise ValueError(f"JPEG/JSON dimensions disagree: {split}/{stem}")
            rec = {"key": f"{split}/{stem}", "split": split, "stem": stem,
                   "width": width, "height": height, "image_sha256": sha(images[stem]), "objects": []}
            for index, shape in enumerate(obj["shapes"]):
                if shape["shape_type"] != "rectangle":
                    raise ValueError(f"Unsupported shape: {rec['key']}/{index}")
                bbox = bbox_from_points(shape["points"])
                text = shape.get("description", "")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Empty description: {rec['key']}/{index}")
                cat = mapping[shape["label"]]
                rec["objects"].append({"shape_index": index, "category_id": cat, "text": text, "bbox": bbox})
                x, y, w, h = bbox
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    warnings.append({"kind": "out_of_bounds_json", "key": rec["key"], "shape_index": index, "bbox": bbox})
            xml = ET.parse(xmls[stem]).getroot()
            if (int(xml.findtext("size/width")), int(xml.findtext("size/height"))) != (width, height):
                warnings.append({"kind": "xml_size_difference", "key": rec["key"]})
            xml_objects = xml.findall("object")
            # Compare as multisets: object order is irrelevant.
            xml_boxes = []
            for o in xml_objects:
                x1, y1, x2, y2 = [float(o.findtext("bndbox/" + k)) for k in ("xmin", "ymin", "xmax", "ymax")]
                xml_boxes.append((mapping[o.findtext("name")], x1, y1, x2, y2))
            json_boxes = []
            for o in rec["objects"]:
                x, y, w, h = o["bbox"]
                json_boxes.append((o["category_id"], x, y, x + w, y + h))
            if collections.Counter(t[0] for t in xml_boxes) != collections.Counter(t[0] for t in json_boxes):
                raise ValueError(f"JSON/XML object counts or categories disagree: {rec['key']}")
            if sorted(xml_boxes) != sorted(json_boxes):
                max_delta = max((abs(a - b) for xb, jb in zip(sorted(xml_boxes), sorted(json_boxes))
                                 for a, b in zip(xb[1:], jb[1:])), default=0)
                warnings.append({"kind": "json_xml_coordinate_difference", "key": rec["key"],
                                 "max_sorted_pair_delta": max_delta})
            records.append(rec)
    return records, warnings, mapping


def audit(data, records, warnings, mapping):
    descriptions, image_hashes = collections.defaultdict(list), collections.defaultdict(list)
    summary = {}
    for split in ("train", "val"):
        rows = [r for r in records if r["split"] == split]
        summary[split] = {
            "images": len(rows), "objects": sum(len(r["objects"]) for r in rows),
            "objects_by_category": dict(sorted(collections.Counter(o["category_id"] for r in rows for o in r["objects"]).items())),
            "images_by_category": dict(sorted(collections.Counter(c for r in rows for c in {o["category_id"] for o in r["objects"]}).items())),
            "dimensions": dict(collections.Counter(f"{r['width']}x{r['height']}" for r in rows)),
            "empty_annotation_images": [r["key"] for r in rows if not r["objects"]],
        }
    for r in records:
        image_hashes[r["image_sha256"]].append(r["key"])
        for o in r["objects"]:
            descriptions[normalized(o["text"])].append({"key": r["key"], "shape_index": o["shape_index"], "category_id": o["category_id"]})
    repeated = [{"text": k, "occurrences": v} for k, v in sorted(descriptions.items()) if len(v) > 1]
    across_images = [g for g in repeated if len({r["key"] for r in g["occurrences"]}) > 1]
    across_split = [g for g in across_images if len({r["key"].split("/")[0] for r in g["occurrences"]}) > 1]
    return {"version": VERSION, "category_map": mapping, "summary": summary,
            "warnings": warnings,
            "exact_duplicate_images": [v for v in image_hashes.values() if len(v) > 1],
            "unique_normalized_descriptions": len(descriptions),
            "repeated_descriptions": repeated,
            "description_groups_across_images": len(across_images),
            "description_groups_across_train_val": len(across_split),
            "scope": "SHA256 finds identical JPEG bytes only; near-duplicate/spatial overlap is not ruled out."}


def choose_dev(records):
    buckets = collections.defaultdict(list)
    for r in records:
        if r["split"] != "train" or not r["objects"]:
            continue
        counts = collections.Counter(o["category_id"] for o in r["objects"])
        dominant = min(counts, key=lambda c: (-counts[c], c))
        buckets[dominant].append(r)
    chosen = []
    for cat, quota in QUOTAS.items():
        bucket = sorted(buckets[cat], key=lambda r: hashlib.sha256((SEED + "/" + r["key"]).encode()).hexdigest())
        if len(bucket) < quota:
            raise ValueError(f"Insufficient images for category {cat}")
        chosen.extend(r["key"] for r in bucket[:quota])
    return sorted(chosen)


def make_protocol(name, query_records, gallery, out):
    input_dir = out / "protocols" / name / "input_path"
    info = [{"image_id": r["image_id"], "file_name": r["file_name"], "width": r["width"], "height": r["height"]} for r in gallery]
    alternatives = collections.defaultdict(list)
    for r in gallery:
        for o in r["objects"]:
            alternatives[(normalized(o["text"]), o["category_id"])].append((r["image_id"], o["bbox"]))
    queries, answers = [], []
    for r in query_records:
        for o in r["objects"]:
            key = (normalized(o["text"]), o["category_id"])
            alts = alternatives[key]
            for kind in ("retrieval", "grounding"):
                qid = len(queries) + 1
                q = {"query_id": qid, "type": kind, "text": o["text"]}
                if kind == "grounding":
                    q["image_id"] = r["image_id"]
                queries.append(q)
                a = {"query_id": qid, "type": kind, "image_id": r["image_id"], "category_id": o["category_id"],
                     "bbox": o["bbox"], "source_key": r["key"], "shape_index": o["shape_index"],
                     "alternative_image_ids": sorted({i for i, b in alts}),
                     "alternative_bboxes": [b for i, b in alts if i == r["image_id"]]}
                answers.append(a)
    write(input_dir / "info.json", info)
    write(input_dir / "queries.json", queries)
    write(out / "private" / f"{name}_gt.json", {
        "schema_version": VERSION, "protocol": name, "gallery": info, "answers": answers,
        "metric_policy": {"weights": WEIGHTS, "bbox": "continuous_xywh_no_plus_one_no_clipping",
                          "missing_prediction": "zero", "category_match_required_for_primary": False,
                          "primary_target": "one_original_annotation", "multi_target": "diagnostic_only"}})
    text_targets = collections.defaultdict(collections.Counter)
    repeated_in_image = collections.Counter()
    for r in query_records:
        for o in r["objects"]:
            text_targets[o["text"]][r["image_id"]] += 1
            repeated_in_image[(r["image_id"], o["text"])] += 1
    n = sum(sum(c.values()) for c in text_targets.values())
    return {"images": len(info), "query_images": len(query_records), "queries": len(queries),
            "objects_by_category": dict(collections.Counter(o["category_id"] for r in query_records for o in r["objects"])),
            "duplicate_text_diagnostic": {
                "repeated_text_groups_across_query_images": sum(len(c) > 1 for c in text_targets.values()),
                "image_text_groups_with_multiple_target_objects": sum(v > 1 for v in repeated_in_image.values()),
                "text_comparison": "exact original string",
                "text_only_single_answer_R1_ceiling": sum(max(c.values()) for c in text_targets.values()) / n if n else None,
                "text_only_single_answer_R5_ceiling": sum(sum(sorted(c.values(), reverse=True)[:5]) for c in text_targets.values()) / n if n else None,
                "ceiling_scope": "For a deterministic retriever using text only; query_id and annotation order must not be used as model signals."}}


def build(data, out):
    if data == out or data in out.parents:
        raise ValueError("Output must be outside the read-only training package")
    before = source_inventory(data)
    records, warnings, mapping = load_data(data)
    report = audit(data, records, warnings, mapping)
    if report["exact_duplicate_images"]:
        raise ValueError("Duplicate JPEG files found; resolve split policy before building")
    dev_keys = choose_dev(records)
    train_keys = sorted(r["key"] for r in records if r["split"] == "train" and r["key"] not in dev_keys)
    manifest = {"version": VERSION, "seed": SEED, "selection": "dominant-category quota then SHA256(seed/key) order",
                "dominant_category_quotas": QUOTAS, "dev_keys": dev_keys, "train_keys": train_keys,
                "official_val_keys": sorted(r["key"] for r in records if r["split"] == "val"),
                "source_sha256": before}
    manifest_path = out / "splits" / "manifest.json"
    if manifest_path.exists() and read(manifest_path) != json.loads(json.dumps(manifest)):
        raise ValueError("Existing frozen manifest differs. Use a new output directory for a new protocol.")
    gallery = [dict(r) for r in records if r["key"] in dev_keys or r["split"] == "val"]
    for i, r in enumerate(gallery, 1):
        r["image_id"], r["file_name"] = i, f"{i:05d}.jpg"
    protocols = {"dev4_81": [r for r in gallery if r["key"] in dev_keys],
                 "official_val51_81": [r for r in gallery if r["split"] == "val"]}
    results = {}
    for name, query_records in protocols.items():
        results[name] = make_protocol(name, query_records, gallery, out)
        image_dir = out / "protocols" / name / "input_path" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for r in gallery:
            src = data / r["split"] / "images" / (r["stem"] + ".jpg")
            dst = image_dir / r["file_name"]
            if not dst.exists() or sha(dst) != r["image_sha256"]:
                shutil.copyfile(src, dst)
    heldout_text = {normalized(o["text"]) for r in gallery for o in r["objects"]}
    train_text = {normalized(o["text"]) for r in records if r["key"] in train_keys for o in r["objects"]}
    report["frozen_split"] = {"train_images": len(train_keys), "dev_images": len(dev_keys),
        "official_val_images": len(manifest["official_val_keys"]), "gallery_images": len(gallery),
        "shared_normalized_train_heldout_descriptions": len(train_text & heldout_text),
        "exact_image_hash_overlap": sorted({r["image_sha256"] for r in records if r["key"] in train_keys}
                                           & {r["image_sha256"] for r in gallery})}
    if source_inventory(data) != before:
        raise ValueError("Source files changed during build")
    write(manifest_path, manifest)
    for filename, keys in (("train_images.json", train_keys), ("dev_images.json", dev_keys),
                           ("official_val_images.json", manifest["official_val_keys"])):
        keyset = set(keys)
        write(out / "splits" / filename, [
            {"source_key": r["key"], "image_path": f"{r['split']}/images/{r['stem']}.jpg",
             "label_json_path": f"{r['split']}/labels_json/{r['stem']}.json",
             "width": r["width"], "height": r["height"], "object_count": len(r["objects"])}
            for r in records if r["key"] in keyset])
    for source_name in ("README.md", "classes.txt", "label_map.json"):
        destination = out / "docs" / ("DATASET_README.md" if source_name == "README.md" else source_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data / source_name, destination)
    write(out / "audit" / "data_audit.json", report)
    write(out / "audit" / "build_summary.json", {"version": VERSION, "protocols": results,
          "train_images": len(train_keys), "source_files_unchanged": True})
    print(json.dumps({"write_verified": True, "source_unchanged": True, "protocols": results, "train_images": len(train_keys)}, ensure_ascii=False))


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def validate_predictions(gt, predictions):
    if not isinstance(predictions, list):
        raise ValueError("pred.json must be a JSON array")
    answers = {a["query_id"]: a for a in gt["answers"]}
    gallery_ids = {r["image_id"] for r in gt["gallery"]}
    seen = {}
    for index, p in enumerate(predictions):
        if not isinstance(p, dict):
            raise ValueError(f"Prediction {index} must be an object")
        qid = p.get("query_id")
        if not integer(qid) or qid not in answers or qid in seen:
            raise ValueError(f"Unknown/duplicate/non-integer query_id at prediction {index}")
        a = answers[qid]
        if p.get("type") != a["type"]:
            raise ValueError(f"Wrong query type: {qid}")
        if a["type"] == "retrieval":
            ranking = p.get("ranking")
            if not isinstance(ranking, list) or len(ranking) < min(5, len(gallery_ids)):
                raise ValueError(f"Retrieval {qid} must provide at least Top-5")
            if any(not integer(i) or i not in gallery_ids for i in ranking) or len(set(ranking)) != len(ranking):
                raise ValueError(f"Retrieval {qid}: invalid or repeated image_id")
        else:
            if not integer(p.get("image_id")) or p["image_id"] != a["image_id"]:
                raise ValueError(f"Grounding {qid}: wrong image_id")
            if not integer(p.get("category_id")) or p["category_id"] not in (1, 2, 3, 4):
                raise ValueError(f"Grounding {qid}: category_id must be 1..4")
            if not number(p.get("score")) or not 0 <= p["score"] <= 1:
                raise ValueError(f"Grounding {qid}: score must be finite and within [0,1]")
            b = p.get("bbox")
            if not isinstance(b, list) or len(b) != 4 or not all(number(v) for v in b) or b[2] <= 0 or b[3] <= 0:
                raise ValueError(f"Grounding {qid}: bbox must be finite xywh with positive width/height")
            if not all(math.isfinite(v) for v in (b[0] + b[2], b[1] + b[3], b[2] * b[3])):
                raise ValueError(f"Grounding {qid}: bbox arithmetic overflow")
        seen[qid] = p
    return seen


def aggregate(rows):
    r = [x for x in rows if x["type"] == "retrieval"]
    g = [x for x in rows if x["type"] == "grounding"]
    def mean(items, key):
        return sum(x[key] for x in items) / len(items) if items else None
    m = {"R1": mean(r, "hit1"), "R5": mean(r, "hit5"), "Acc05": mean(g, "acc05"), "Acc75": mean(g, "acc75")}
    total = sum(WEIGHTS[k] * m[k] for k in WEIGHTS) if all(v is not None for v in m.values()) else None
    return {"retrieval_queries": len(r), "grounding_queries": len(g), **m, "total": total,
            "total_percent": total * 100 if total is not None else None}


def image_bootstrap(rows, samples=1000, seed=20261121):
    """Resample source images, keeping correlated queries within each image together."""
    if samples <= 0:
        return None
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["source_key"]].append(row)
    # Sufficient statistics keep this independent of the number of object queries.
    stats = []
    for key in sorted(grouped):
        r = [x for x in grouped[key] if x["type"] == "retrieval"]
        g = [x for x in grouped[key] if x["type"] == "grounding"]
        stats.append([len(r), len(g), sum(x["hit1"] for x in r), sum(x["hit5"] for x in r),
                      sum(x["acc05"] for x in g), sum(x["acc75"] for x in g)])
    if not stats:
        return None
    rng = random.Random(seed)
    values = {k: [] for k in (*WEIGHTS, "total")}
    for _ in range(samples):
        totals = [0] * 6
        for _ in stats:
            v = rng.choice(stats)
            totals = [a + b for a, b in zip(totals, v)]
        nr, ng, h1, h5, a5, a75 = totals
        if nr == 0 or ng == 0:
            continue
        metrics = {"R1": h1 / nr, "R5": h5 / nr, "Acc05": a5 / ng, "Acc75": a75 / ng}
        metrics["total"] = sum(WEIGHTS[k] * metrics[k] for k in WEIGHTS)
        for k, v in metrics.items():
            values[k].append(v)
    def percentile(items, p):
        items = sorted(items)
        if not items:
            return None
        index = (len(items) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return items[lo] + (items[hi] - items[lo]) * (index - lo)
    return {"method": "source-image-cluster percentile bootstrap", "confidence": 0.95,
            "samples": samples, "valid_samples": len(values["total"]), "seed": seed,
            "source_images": len(stats),
            "intervals": {k: [percentile(v, 0.025), percentile(v, 0.975)] for k, v in values.items()},
            "scope": "Local held-out sampling uncertainty only; does not estimate leaderboard distribution shift or training randomness."}


def evaluate(gt, predictions):
    pred = validate_predictions(gt, predictions)
    details, multi, gated = [], [], []
    for a in gt["answers"]:
        qid = a["query_id"]
        p = pred.get(qid)
        row = {"query_id": qid, "type": a["type"], "category_id": a["category_id"],
               "source_key": a["source_key"], "missing": p is None}
        if a["type"] == "retrieval":
            ranking = p["ranking"] if p else []
            row.update(hit1=int(a["image_id"] in ranking[:1]), hit5=int(a["image_id"] in ranking[:5]))
            alt = dict(row, hit1=int(any(i in a["alternative_image_ids"] for i in ranking[:1])),
                       hit5=int(any(i in a["alternative_image_ids"] for i in ranking[:5])))
            gate = dict(row)
        else:
            overlap = iou(p["bbox"], a["bbox"]) if p else 0.0
            row.update(iou=overlap, acc05=int(overlap >= 0.5), acc75=int(overlap >= 0.75),
                       category_correct=bool(p and p["category_id"] == a["category_id"]))
            max_iou = max((iou(p["bbox"], b) for b in a["alternative_bboxes"]), default=0.0) if p else 0.0
            alt = dict(row, iou=max_iou, acc05=int(max_iou >= 0.5), acc75=int(max_iou >= 0.75))
            gate = dict(row, acc05=int(row["acc05"] and row["category_correct"]),
                        acc75=int(row["acc75"] and row["category_correct"]))
        details.append(row)
        multi.append(alt)
        gated.append(gate)
    by_class = {str(cat): aggregate([r for r in details if r["category_id"] == cat]) for cat in (1, 2, 3, 4)}
    cls_totals = [m["total"] for m in by_class.values() if m["total"] is not None]
    return {"version": VERSION, "protocol": gt["protocol"], "primary": aggregate(details),
            "missing_queries": sum(r["missing"] for r in details),
            "by_category": by_class, "macro_category_total_diagnostic": sum(cls_totals) / len(cls_totals) if cls_totals else None,
            "multi_target_diagnostic": aggregate(multi), "category_gated_diagnostic": aggregate(gated),
            "details": details}


def baseline(gt, mode, seed=20261121):
    rng = random.Random(seed)
    ids = [r["image_id"] for r in gt["gallery"]]
    images = {r["image_id"]: r for r in gt["gallery"]}
    predictions = []
    for a in gt["answers"]:
        p = {"query_id": a["query_id"], "type": a["type"]}
        if a["type"] == "retrieval":
            ranking = list(ids)
            if mode == "oracle":
                ranking.remove(a["image_id"])
                ranking.insert(0, a["image_id"])
            elif mode == "zero":
                ranking.remove(a["image_id"])
                ranking.append(a["image_id"])
            elif mode == "random":
                rng.shuffle(ranking)
            p["ranking"] = ranking
        else:
            image = images[a["image_id"]]
            bbox = a["bbox"]
            cat = a["category_id"]
            if mode in ("random", "full_image"):
                cat = rng.randint(1, 4) if mode == "random" else 1
                w, h = image["width"], image["height"]
                bbox = [0, 0, w, h]
                if mode == "random":
                    x, y = rng.uniform(0, w - 1), rng.uniform(0, h - 1)
                    bbox = [x, y, rng.uniform(1, w - x), rng.uniform(1, h - y)]
            elif mode == "zero":
                bbox = [image["width"] + 100, image["height"] + 100, 1, 1]
            p.update(image_id=a["image_id"], category_id=cat, score=1.0, bbox=bbox)
        predictions.append(p)
    return predictions


def strict_json(path):
    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj:
                raise ValueError(f"Duplicate JSON key: {k}")
            obj[k] = v
        return obj
    def bad(value):
        raise ValueError(f"Non-standard JSON number: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=pairs, parse_constant=bad)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", type=Path, default=DEFAULT_DATA)
    b.add_argument("--out", type=Path, default=ROOT)
    s = sub.add_parser("score")
    s.add_argument("--gt", type=Path, required=True)
    s.add_argument("--pred", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--require-complete", action="store_true")
    s.add_argument("--bootstrap", type=int, default=1000, help="image-cluster bootstrap samples; 0 disables")
    c = sub.add_parser("baselines")
    c.add_argument("--root", type=Path, default=ROOT)
    v = sub.add_parser("verify-source")
    v.add_argument("--data", type=Path, default=DEFAULT_DATA)
    v.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    if args.command == "build":
        build(args.data.resolve(), args.out.resolve())
    elif args.command == "score":
        gt = read(args.gt)
        predictions = strict_json(args.pred)
        result = evaluate(gt, predictions)
        if args.require_complete and result["missing_queries"]:
            raise ValueError(f"Missing {result['missing_queries']} queries")
        result["image_bootstrap_95pct"] = image_bootstrap(result["details"], args.bootstrap)
        result["provenance"] = {"gt_sha256": sha(args.gt), "pred_sha256": sha(args.pred), "evaluator_sha256": sha(Path(__file__))}
        write(args.out, result)
        print(json.dumps({"primary": result["primary"], "missing_queries": result["missing_queries"]}))
    elif args.command == "baselines":
        summary = {}
        for name in ("dev4_81", "official_val51_81"):
            gt = read(args.root / "private" / f"{name}_gt.json")
            summary[name] = {}
            for mode in ("oracle", "zero", "random", "full_image"):
                predictions = baseline(gt, mode)
                result = evaluate(gt, predictions)
                folder = args.root / "baselines" / name
                write(folder / f"{mode}_pred.json", predictions)
                write(folder / f"{mode}_metrics.json", result)
                summary[name][mode] = result["primary"]
                if mode in ("oracle", "zero"):
                    expected = 1.0 if mode == "oracle" else 0.0
                    if any(result["primary"][k] != expected for k in WEIGHTS):
                        raise ValueError(f"{mode} sanity check failed")
        write(args.root / "audit" / "baseline_summary.json", summary)
        print(json.dumps(summary))
    elif args.command == "verify-source":
        expected = read(args.root / "splits" / "manifest.json")["source_sha256"]
        actual = source_inventory(args.data.resolve())
        if expected != actual:
            raise ValueError("Training-package files differ from the frozen source inventory")
        print(json.dumps({"source_files_unchanged": True, "verified_files": len(actual)}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ET.ParseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
