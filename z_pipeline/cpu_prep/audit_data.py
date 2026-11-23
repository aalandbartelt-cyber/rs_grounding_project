"""Audit frozen source annotations and suggest augmentation review flags, stdlib only."""
from __future__ import annotations

import argparse
import collections
from pathlib import Path
import re

from common import REPO, KEYWORDS, Resources, fresh_output, read, write, sha, side_bin
from evalkit import load_data, source_inventory, normalized


def audit(data, output, budget_mb=256):
    resources = Resources(budget_mb)
    data = data.resolve()
    if output.resolve() == data or data in output.resolve().parents:
        raise ValueError("Do not write into the source dataset")
    manifest_path = REPO / "z_evaluation" / "splits" / "manifest.json"
    manifest = read(manifest_path)
    if source_inventory(data) != manifest["source_sha256"]:
        raise ValueError("Frozen source inventory mismatch")
    output = fresh_output(output)
    groups = {"train": set(manifest["train_keys"]), "dev": set(manifest["dev_keys"]),
              "official_val": set(manifest["official_val_keys"])}
    all_keys = [key for keys in groups.values() for key in keys]
    if len(set(all_keys)) != len(all_keys):
        raise ValueError("Image overlap between train and held-out splits")
    records, warnings, mapping = load_data(data)
    summary = {}
    training_reviews = []
    for split, keys in groups.items():
        rows = [r for r in records if r["key"] in keys]
        if {r["key"] for r in rows} != keys:
            raise ValueError("Frozen image missing in source")
        objects = [(r, o) for r in rows for o in r["objects"]]
        category = {}
        for cid in range(1, 5):
            selected = [(r, o) for r, o in objects if o["category_id"] == cid]
            sides = sorted(min(o["bbox"][2:]) for r, o in selected)
            median = (sides[(len(sides)-1)//2] + sides[len(sides)//2]) / 2 if sides else None
            category[str(cid)] = {"objects": len(selected), "source_images": len({r["key"] for r, o in selected}),
                "median_short_side_px": median,
                "short_side_bins": dict(collections.Counter(side_bin(o["bbox"]) for r, o in selected))}
        text_groups = collections.defaultdict(list)
        for r, o in objects:
            text_groups[normalized(o["text"])].append((r["key"], o["shape_index"]))
            if split == "train":
                flags = {name: bool(re.search(pattern, o["text"])) for name, pattern in KEYWORDS.items()}
                training_reviews.append({"source_key": r["key"], "shape_index": o["shape_index"],
                    "category_id": o["category_id"], "bbox": o["bbox"], "text": o["text"], **flags,
                    "joint_geometric_augmentation": "disabled_pending_semantic_review",
                    "photometric_review": "required_for_color_attributes" if flags["color_brightness_flag"] else "required",
                    "semantic_review_status": "not_human_reviewed"})
        summary[split] = {"images": len(rows), "objects": len(objects), "categories": category,
            "objects_per_image_max": max(len(r["objects"]) for r in rows),
            "empty_images": sum(not r["objects"] for r in rows),
            "keyword_flags": {name: sum(bool(re.search(pattern, o["text"])) for r, o in objects)
                              for name, pattern in KEYWORDS.items()},
            "unique_normalized_texts": len(text_groups),
            "same_text_multiple_images": sum(len({key for key, _ in values}) > 1 for values in text_groups.values()),
            "same_image_text_multiple_objects": sum(n > 1 for n in collections.Counter(
                (r["key"], normalized(o["text"])) for r, o in objects).values())}
        resources.check()
    write(output / "training_review_queue.json", training_reviews)
    report = {"version": "cpu-prep-1", "manifest_sha256": sha(manifest_path),
        "source_files_verified": len(manifest["source_sha256"]), "source_unchanged": source_inventory(data) == manifest["source_sha256"],
        "category_map": mapping, "splits": summary, "annotation_warning_counts": dict(collections.Counter(w["kind"] for w in warnings)),
        "keyword_patterns": KEYWORDS,
        "limitations": ["Keyword flags are review hints, not validated semantic labels",
                         "Augmentation cannot create independent geographic scenes from the same source image",
                         "Small-object bins use original-pixel short side, not COCO AP size bins"],
        "resources": resources.report()}
    write(output / "audit.json", report)
    md = ["# 数据与增强准备审核", "", "仅审核；原始图片和标注未修改。关键词标记不等同于人工语义审核。", "",
          "| 划分 | 图片 | 目标 | 颜色/明暗标记 | 位置/方向标记 |", "|---|---:|---:|---:|---:|"]
    for split, row in summary.items():
        md.append(f"| {split} | {row['images']} | {row['objects']} | {row['keyword_flags']['color_brightness_flag']} | {row['keyword_flags']['position_direction_flag']} |")
    md += ["", "训练样本的图文几何增强默认关闭；旋转/翻转仅作为检测预览候选。",
           "训练样本审核队列见 training_review_queue.json，不包含开发/验证样本。", "",
           "进程资源：" + str(report["resources"])]
    (output / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print({"splits": {k: {"images": v["images"], "objects": v["objects"]} for k, v in summary.items()},
           "resources": report["resources"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget-mb", type=int, default=256)
    args = parser.parse_args()
    audit(args.data, args.out, args.budget_mb)
