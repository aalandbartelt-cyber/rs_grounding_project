#!/usr/bin/env python3
"""按类别量化定位难度：目标尺寸、同类多目标比例、图内重复描述导致的文本可分辨上限。

文本可分辨上限 = 唯一(image, description)组数 / 框总数。
含义：若同一图内多个框共享完全相同的描述，任何"只看文本"的选择器在组内只能猜，
平均正确率最多为 1/k，因此整体上限就是该比值。
"""
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.expanduser("~/rs_grounding_project")
DATA = os.path.join(ROOT, "data/rs_grounding_train_v1")


def bbox(shape):
    xs = [p[0] for p in shape["points"]]
    ys = [p[1] for p in shape["points"]]
    return min(xs), min(ys), max(xs), max(ys)


def profile(split):
    label_dir = os.path.join(DATA, split, "labels_json")
    per_class = defaultdict(lambda: {
        "boxes": 0, "images": 0, "dup_boxes": 0, "groups": 0,
        "short": [], "multi_images": 0,
    })
    for name in sorted(os.listdir(label_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(label_dir, name), encoding="utf-8") as fh:
            row = json.load(fh)
        by_class = defaultdict(list)
        for shape in row.get("shapes") or []:
            label = shape.get("label", "")
            by_class[label].append(shape)
        for label, shapes in by_class.items():
            st = per_class[label]
            st["images"] += 1
            st["boxes"] += len(shapes)
            if len(shapes) > 1:
                st["multi_images"] += 1
            texts = Counter((s.get("description") or "").strip() for s in shapes)
            st["groups"] += len(texts)
            st["dup_boxes"] += len(shapes) - len(texts)
            for shape in shapes:
                x1, y1, x2, y2 = bbox(shape)
                st["short"].append(min(x2 - x1, y2 - y1))
    return per_class


def main():
    for split in ("train", "val"):
        per_class = profile(split)
        total_boxes = sum(st["boxes"] for st in per_class.values())
        total_groups = sum(st["groups"] for st in per_class.values())
        print(f"\n=== {split}: 文本可分辨上限 = {total_groups}/{total_boxes} "
              f"= {100 * total_groups / total_boxes:.1f}% ===")
        print(f"{'类别':<6}{'框数':>6}{'图片':>6}{'均框/图':>8}{'同类多目标图':>12}"
              f"{'重复框':>7}{'文本上限':>9}{'短边p50':>9}")
        for label, st in sorted(per_class.items(), key=lambda kv: -kv[1]["boxes"]):
            short = sorted(st["short"])
            p50 = short[len(short) // 2]
            print(f"{label:<6}{st['boxes']:>6}{st['images']:>6}"
                  f"{st['boxes'] / st['images']:>8.2f}{st['multi_images']:>12}"
                  f"{st['dup_boxes']:>7}"
                  f"{100 * st['groups'] / st['boxes']:>8.1f}%{p50:>9.0f}")


if __name__ == "__main__":
    main()
