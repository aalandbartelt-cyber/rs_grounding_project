#!/usr/bin/env python3
"""核对参考材料中的关键数字与假设，输出可引用的证据。"""
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter

ROOT = os.path.expanduser("~/rs_grounding_project")
DATA = os.path.join(ROOT, "data/rs_grounding_train_v1")
CLASS_WORDS = {"油罐": ["油罐", "罐"], "舰船": ["舰船", "舰", "船"],
               "车辆": ["车辆", "车"], "飞机": ["飞机", "机"]}


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


def main():
    total_json = total_xml = 0
    point_counts = Counter()
    in_image_dup = 0
    images_with_dup = 0
    class_word_miss = 0
    per_class_short = {}
    color_per_class = {}
    xml_mismatch = 0
    long_images = []

    for split in ("train", "val"):
        rows = load(split)
        xml_dir = os.path.join(DATA, split, "labels_xml")
        for name, row in rows:
            texts = []
            per_class = Counter()
            for shape in row.get("shapes") or []:
                total_json += 1
                point_counts[len(shape.get("points") or [])] += 1
                label = shape.get("label", "")
                desc = shape.get("description", "")
                texts.append(desc)
                per_class[label] += 1
                if not any(word in desc for word in CLASS_WORDS.get(label, [label])):
                    class_word_miss += 1
                x1, y1, x2, y2 = bbox(shape)
                per_class_short.setdefault(label, []).append(min(x2 - x1, y2 - y1))
                for color in ["白", "灰", "绿", "蓝", "红", "黑", "黄", "橙"]:
                    if color in desc:
                        color_per_class.setdefault(label, Counter())[color] += 1
                        break
            dup = sum(c - 1 for c in Counter(texts).values() if c > 1)
            if dup:
                images_with_dup += 1
                in_image_dup += dup
            if len(texts) >= 50:
                long_images.append((split, name, len(texts)))
            # XML 对账
            xml_path = os.path.join(xml_dir, name.replace(".json", ".xml"))
            if os.path.exists(xml_path):
                root = ET.parse(xml_path).getroot()
                n_xml = len(root.findall("object"))
                total_xml += n_xml
                if n_xml != len(texts):
                    xml_mismatch += 1

    print("== 数量对账 ==")
    print(f"JSON shapes 总数: {total_json}")
    print(f"XML  object 总数: {total_xml}")
    print(f"README 声称的描述-框对: 3095  -> 与实际差 {3095 - total_json}")
    print(f"XML/JSON 框数不一致的图片数: {xml_mismatch}")

    print("\n== 框点格式 ==")
    print(f"每种点数的框数量: {dict(point_counts)}  (材料称'两角点')")

    print("\n== 文本歧义 ==")
    print(f"图内出现重复描述的图片: {images_with_dup}，重复框合计 {in_image_dup}")
    print(f"描述中缺失类别词的框: {class_word_miss}/{total_json} "
          f"({100 * class_word_miss / total_json:.1f}%)")

    print("\n== 各类目标尺寸(短边px)与颜色 ==")
    for label, values in sorted(per_class_short.items()):
        values.sort()
        p10 = values[int(0.1 * (len(values) - 1))]
        p50 = values[len(values) // 2]
        print(f"{label}: n={len(values)} p10={p10} p50={p50} "
              f"| 颜色 top3={color_per_class.get(label, Counter()).most_common(3)}")

    print("\n== 长尾拥挤图 ==")
    print(f"单图>=50个目标的图片: {len(long_images)}，最大: {max(long_images or [(0, 0, 0)], key=lambda x: x[2])}")


if __name__ == "__main__":
    main()
