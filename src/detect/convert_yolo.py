"""把 labelme JSON 转成 Ultralytics YOLO 目录（图片用软链接，标签写 txt）。

输出：
  data/yolo/images/<split>/*.jpg -> 原图
  data/yolo/labels/<split>/*.txt
  data/yolo/data_baseline.yaml   # train_minus_dev4 -> dev4（选模）
  data/yolo/data_full.yaml       # official_train -> official_val（最终报告/重训）
  data/yolo/data_dev4.yaml       # train_minus_dev4 -> dev4（显式）
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.labels import load_label  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "rs_grounding_train_v1")
SPLITS = os.path.join(ROOT, "data", "splits")
DEFAULT_OUT = os.path.join(ROOT, "data", "yolo")


def source_split(name):
    return "val" if name == "official_val" else "train"


def read_manifest(name):
    with open(os.path.join(SPLITS, f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)["images"]


def convert_split(out_root, split_name, image_ids):
    image_dir = os.path.join(out_root, "images", split_name)
    label_dir = os.path.join(out_root, "labels", split_name)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    label_split = source_split(split_name)
    written = skipped = 0
    for iid in image_ids:
        src_image = os.path.join(DATA, label_split, "images", f"{iid}.jpg")
        dst_image = os.path.join(image_dir, f"{iid}.jpg")
        if not os.path.exists(src_image):
            raise FileNotFoundError(src_image)
        if os.path.islink(dst_image) or os.path.exists(dst_image):
            os.remove(dst_image)
        os.symlink(src_image, dst_image)

        label = load_label(os.path.join(DATA, label_split, "labels_json", f"{iid}.json"))
        lines = []
        for obj in label["objects"]:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            width, height = label["width"], label["height"]
            cx = (x1 + x2) / 2.0 / width
            cy = (y1 + y2) / 2.0 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
            if w <= 0 or h <= 0:
                skipped += 1
                continue
            lines.append(f"{obj['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(os.path.join(label_dir, f"{iid}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        written += 1
    print(f"[{split_name}] images={written} skipped_boxes={skipped}")
    return written


def write_yaml(out_root, name, train_split, val_split):
    path = os.path.join(out_root, name)
    text = (
        f"path: {out_root}\n"
        f"train: images/{train_split}\n"
        f"val: images/{val_split}\n"
        "names:\n"
        "  0: oil_tank\n"
        "  1: ship\n"
        "  2: vehicle\n"
        "  3: airplane\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    required = ["official_train", "official_val", "dev4", "train_minus_dev4"]
    manifests = {}
    for name in required:
        path = os.path.join(SPLITS, f"{name}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 不存在，先运行 src/eval/make_splits.py")
        manifests[name] = read_manifest(name)

    # 去重：official_train 已覆盖 train_minus_dev4 / dev4，但仍按 split 写软链接，便于切换 yaml
    convert_split(args.out, "official_train", manifests["official_train"])
    convert_split(args.out, "official_val", manifests["official_val"])
    convert_split(args.out, "dev4", manifests["dev4"])
    convert_split(args.out, "train_minus_dev4", manifests["train_minus_dev4"])

    write_yaml(args.out, "data_baseline.yaml", "train_minus_dev4", "dev4")
    write_yaml(args.out, "data_dev4.yaml", "train_minus_dev4", "dev4")
    write_yaml(args.out, "data_full.yaml", "official_train", "official_val")


if __name__ == "__main__":
    main()
