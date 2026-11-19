#!/usr/bin/env python3
"""统计官方训练包的数据分布，为技术路线提供依据。

输出：终端 Markdown 表 + JSON（outputs/analysis/dataset_profile.json）。
只读 data/rs_grounding_train_v1，不修改任何原始文件。
"""
import json
import os
import re
from collections import Counter, defaultdict

ROOT = os.path.expanduser("~/rs_grounding_project")
DATA = os.path.join(ROOT, "data/rs_grounding_train_v1")
OUT = os.path.join(ROOT, "outputs/analysis/dataset_profile.json")

COLOR_WORDS = ["白", "灰", "红", "蓝", "绿", "黑", "黄", "银", "橙", "褐"]
POS_WORDS = ["左", "右", "上", "下", "中", "中央", "边", "角", "附近", "旁"]
SIZE_WORDS = ["大", "小", "长", "宽", "高", "巨型", "中型", "小型"]
STATE_WORDS = ["停泊", "静止", "行驶", "飞行", "移动", "排列", "整齐"]


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((q / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def bbox_of(shape):
    xs = [p[0] for p in shape["points"]]
    ys = [p[1] for p in shape["points"]]
    return min(xs), min(ys), max(xs), max(ys)


def analyze_split(split):
    label_dir = os.path.join(DATA, split, "labels_json")
    image_dir = os.path.join(DATA, split, "images")
    files = sorted(f for f in os.listdir(label_dir) if f.endswith(".json"))
    stats = {
        "split": split,
        "images": len(files),
        "boxes": 0,
        "classes": Counter(),
        "objects_per_image": [],
        "area_ratio": [],
        "short_side_px": [],
        "desc_chars": [],
        "descriptions": [],
        "multi_same_class_images": 0,
        "empty_desc": 0,
        "color_hits": 0,
        "pos_hits": 0,
        "size_hits": 0,
        "state_hits": 0,
        "bad_boxes": 0,
    }
    for name in files:
        with open(os.path.join(label_dir, name), encoding="utf-8") as fh:
            row = json.load(fh)
        w = row.get("imageWidth") or 1024
        h = row.get("imageHeight") or 1024
        shapes = row.get("shapes") or []
        stats["objects_per_image"].append(len(shapes))
        per_class = Counter()
        for shape in shapes:
            x1, y1, x2, y2 = bbox_of(shape)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                stats["bad_boxes"] += 1
                continue
            label = shape.get("label") or ""
            desc = shape.get("description") or ""
            stats["boxes"] += 1
            stats["classes"][label] += 1
            per_class[label] += 1
            stats["area_ratio"].append((bw * bh) / float(w * h))
            stats["short_side_px"].append(min(bw, bh))
            stats["desc_chars"].append(len(desc))
            if not desc.strip():
                stats["empty_desc"] += 1
            else:
                stats["descriptions"].append(desc)
                if any(word in desc for word in COLOR_WORDS):
                    stats["color_hits"] += 1
                if any(word in desc for word in POS_WORDS):
                    stats["pos_hits"] += 1
                if any(word in desc for word in SIZE_WORDS):
                    stats["size_hits"] += 1
                if any(word in desc for word in STATE_WORDS):
                    stats["state_hits"] += 1
        if any(count > 1 for count in per_class.values()):
            stats["multi_same_class_images"] += 1
    # 图片实际尺寸抽查
    sizes = Counter()
    for name in sorted(os.listdir(image_dir))[:5]:
        sizes[name] = os.path.getsize(os.path.join(image_dir, name))
    stats["sample_image_bytes"] = sizes
    return stats


def summarize(stats):
    boxes = stats["boxes"]
    desc_unique = len(set(stats["descriptions"]))
    return {
        "split": stats["split"],
        "images": stats["images"],
        "boxes": boxes,
        "classes": dict(stats["classes"]),
        "objects_per_image": {
            "mean": round(sum(stats["objects_per_image"]) / max(1, len(stats["objects_per_image"])), 2),
            "max": max(stats["objects_per_image"] or [0]),
            "p50": pct(stats["objects_per_image"], 50),
            "p90": pct(stats["objects_per_image"], 90),
        },
        "area_ratio_percent": {
            "p10": round(100 * pct(stats["area_ratio"], 10), 3),
            "p50": round(100 * pct(stats["area_ratio"], 50), 3),
            "p90": round(100 * pct(stats["area_ratio"], 90), 3),
            "max": round(100 * max(stats["area_ratio"] or [0]), 3),
        },
        "short_side_px": {
            "p10": pct(stats["short_side_px"], 10),
            "p50": pct(stats["short_side_px"], 50),
            "p90": pct(stats["short_side_px"], 90),
        },
        "tiny_ratio_short_side_lt16": round(
            100 * sum(1 for v in stats["short_side_px"] if v < 16) / max(1, boxes), 1),
        "desc_chars": {
            "p10": pct(stats["desc_chars"], 10),
            "p50": pct(stats["desc_chars"], 50),
            "p90": pct(stats["desc_chars"], 90),
        },
        "descriptions_unique": desc_unique,
        "descriptions_unique_ratio": round(100 * desc_unique / max(1, len(stats["descriptions"])), 1),
        "empty_desc": stats["empty_desc"],
        "multi_same_class_images": stats["multi_same_class_images"],
        "cue_ratio_percent": {
            "color": round(100 * stats["color_hits"] / max(1, boxes), 1),
            "position": round(100 * stats["pos_hits"] / max(1, boxes), 1),
            "size": round(100 * stats["size_hits"] / max(1, boxes), 1),
            "state": round(100 * stats["state_hits"] / max(1, boxes), 1),
        },
        "bad_boxes": stats["bad_boxes"],
    }


def main():
    train = analyze_split("train")
    val = analyze_split("val")
    summary = {"train": summarize(train), "val": summarize(val)}
    # 名称重叠检查
    train_names = {os.path.splitext(n)[0] for n in os.listdir(os.path.join(DATA, "train/labels_json"))}
    val_names = {os.path.splitext(n)[0] for n in os.listdir(os.path.join(DATA, "val/labels_json"))}
    summary["name_overlap_train_val"] = sorted(train_names & val_names)
    # 描述重复（同一文本对应多个框）
    all_desc = Counter(train["descriptions"])
    summary["train_duplicate_desc_top"] = [
        {"text": text[:60], "count": count}
        for text, count in all_desc.most_common(5) if count > 1
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    for split in ("train", "val"):
        s = summary[split]
        print(f"\n## {split}: {s['images']} images / {s['boxes']} boxes")
        print(f"- classes: {s['classes']}")
        print(f"- objects/image mean={s['objects_per_image']['mean']} "
              f"p50={s['objects_per_image']['p50']} p90={s['objects_per_image']['p90']} "
              f"max={s['objects_per_image']['max']}")
        print(f"- box area% p10={s['area_ratio_percent']['p10']} "
              f"p50={s['area_ratio_percent']['p50']} p90={s['area_ratio_percent']['p90']}")
        print(f"- short side px p10={s['short_side_px']['p10']} "
              f"p50={s['short_side_px']['p50']} p90={s['short_side_px']['p90']} "
              f"| <16px {s['tiny_ratio_short_side_lt16']}%")
        print(f"- desc chars p10={s['desc_chars']['p10']} p50={s['desc_chars']['p50']} "
              f"p90={s['desc_chars']['p90']} | unique {s['descriptions_unique']} "
              f"({s['descriptions_unique_ratio']}%)")
        print(f"- multi same-class images: {s['multi_same_class_images']}/{s['images']}")
        print(f"- cue% color={s['cue_ratio_percent']['color']} "
              f"pos={s['cue_ratio_percent']['position']} size={s['cue_ratio_percent']['size']} "
              f"state={s['cue_ratio_percent']['state']}")
    print(f"\nname overlap train/val: {summary['name_overlap_train_val']}")
    print(f"train duplicate descriptions: {summary['train_duplicate_desc_top']}")
    print(f"\nJSON: {OUT}")


if __name__ == "__main__":
    main()
