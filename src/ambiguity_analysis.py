#!/usr/bin/env python3
"""量化两类"不可消除的歧义"，用于设定合理目标：

1. 图内重复描述组内，标注顺序与几何先验（最大/最左/最上）是否一致；
2. 同一描述在多少张图中出现 —— 文本检索的理论上限（完美文本匹配但无法区分图像）。
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


def load(split):
    d = os.path.join(DATA, split, "labels_json")
    out = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            with open(os.path.join(d, name), encoding="utf-8") as fh:
                out.append((name, json.load(fh)))
    return out


def group_prior_analysis(rows):
    groups = size_hist = Counter()
    n_groups = n_dup_groups = 0
    first_is_largest = first_is_leftmost = first_is_topmost = 0
    for name, row in rows:
        by_text = defaultdict(list)
        for shape in row.get("shapes") or []:
            by_text[(shape.get("description") or "").strip()].append(shape)
        for text, shapes in by_text.items():
            n_groups += 1
            if len(shapes) > 1:
                n_dup_groups += 1
                size_hist[len(shapes)] += 1
                boxes = [bbox(s) for s in shapes]
                areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
                if areas[0] == max(areas):
                    first_is_largest += 1
                if boxes[0][0] == min(b[0] for b in boxes):
                    first_is_leftmost += 1
                if boxes[0][1] == min(b[1] for b in boxes):
                    first_is_topmost += 1
    return {
        "groups": n_groups,
        "dup_groups": n_dup_groups,
        "size_hist": dict(sorted(size_hist.items())),
        "first_largest_pct": 100 * first_is_largest / max(1, n_dup_groups),
        "first_leftmost_pct": 100 * first_is_leftmost / max(1, n_dup_groups),
        "first_topmost_pct": 100 * first_is_topmost / max(1, n_dup_groups),
    }


def retrieval_ceiling(rows):
    """把每个框当作一条查询；同一描述出现在 m 张图中时，文本检索的期望上限为：
    R@1 = 1/m（组内均匀猜一张），R@5 = min(5,m)/m。
    """
    text_to_images = defaultdict(set)
    box_texts = []
    for name, row in rows:
        for shape in row.get("shapes") or []:
            text = (shape.get("description") or "").strip()
            text_to_images[text].add(name)
            box_texts.append(text)
    r1 = sum(1.0 / len(text_to_images[t]) for t in box_texts) / len(box_texts)
    r5 = sum(min(5, len(text_to_images[t])) / len(text_to_images[t]) for t in box_texts) / len(box_texts)
    dist = Counter(len(text_to_images[t]) for t in box_texts)
    return {
        "queries": len(box_texts),
        "unique_texts": len(text_to_images),
        "images_per_text_hist": dict(sorted(dist.items())[:12]),
        "text_only_R1_ceiling": round(r1, 3),
        "text_only_R5_ceiling": round(r5, 3),
    }


def main():
    for split in ("train", "val"):
        rows = load(split)
        print(f"\n=== {split} ===")
        g = group_prior_analysis(rows)
        print(f"描述组: {g['groups']}，其中图内重复组 {g['dup_groups']}")
        print(f"重复组规模分布: {g['size_hist']}")
        print(f"组内首个标注 == 面积最大: {g['first_largest_pct']:.1f}% | "
              f"== 最左: {g['first_leftmost_pct']:.1f}% | "
              f"== 最上: {g['first_topmost_pct']:.1f}%")
        r = retrieval_ceiling(rows)
        print(f"文本检索理论上限(把每个框当查询): R@1={r['text_only_R1_ceiling']:.3f} "
              f"R@5={r['text_only_R5_ceiling']:.3f}")
        print(f"同一描述覆盖的图像数分布(前12): {r['images_per_text_hist']}")


if __name__ == "__main__":
    main()
