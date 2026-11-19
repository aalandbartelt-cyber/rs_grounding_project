"""生成数据 split：official_train / official_val / dev4 / dev_vehicle / train_minus_dev4。

- official_*：官方目录原样，只读。
- dev4：从 train 分层抽取，覆盖 4 类（车辆只有 7 图，抽 2 图），只用于选模。
- train_minus_dev4：选模阶段的训练集，避免泄漏。
- 所有 manifest 带 sha256，写入 data/splits/SHA256SUMS。
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.labels import CLASS_NAMES, list_image_ids, load_label  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "rs_grounding_train_v1")
DEFAULT_OUT = os.path.join(ROOT, "data", "splits")
SEED = 20261119
QUOTA = {"车辆": 2, "舰船": 8, "油罐": 8, "飞机": 12}
MAX_DEV_IMAGES = 32


def scan(split):
    """返回 {image_id: {"classes": set, "boxes": Counter}}。"""
    info = {}
    for iid in list_image_ids(DATA, split):
        label = load_label(os.path.join(DATA, split, "labels_json", f"{iid}.json"))
        classes = set()
        boxes = Counter()
        for obj in label["objects"]:
            classes.add(obj["label"])
            boxes[obj["label"]] += 1
        info[iid] = {"classes": classes, "boxes": boxes}
    return info


def pick_dev4(train_info):
    images_by_class = defaultdict(list)
    for iid, meta in train_info.items():
        for name in meta["classes"]:
            images_by_class[name].append(iid)
    for name in images_by_class:
        images_by_class[name].sort(key=lambda value: (len(value), value))

    selected, selected_set, counts = [], set(), Counter()
    for name in ["车辆", "舰船", "油罐", "飞机"]:  # 稀缺类优先
        for iid in images_by_class[name]:
            if counts[name] >= QUOTA[name]:
                break
            if iid in selected_set:
                continue
            selected.append(iid)
            selected_set.add(iid)
            for present in train_info[iid]["classes"]:
                counts[present] += 1

    selected.sort(key=lambda value: (len(value), value))
    if len(selected) > MAX_DEV_IMAGES:
        keep = list(selected)
        # 先删“只含飞机”的图；永不删含车辆的图
        for iid in list(keep):
            if len(keep) <= MAX_DEV_IMAGES:
                break
            if "车辆" not in train_info[iid]["classes"] and train_info[iid]["classes"] == {"飞机"}:
                keep.remove(iid)
        selected = keep
    return selected


def summarize(ids, info):
    image_counts, box_counts = Counter(), Counter()
    for iid in ids:
        for name, count in info[iid]["boxes"].items():
            box_counts[name] += count
        for name in info[iid]["classes"]:
            image_counts[name] += 1
    return {
        "images": len(ids),
        "boxes": {name: box_counts[name] for name in CLASS_NAMES},
        "image_counts": {name: image_counts[name] for name in CLASS_NAMES},
    }


def build_manifest(name, ids, info, source):
    return {
        "name": name,
        "images": list(ids),
        "seed": SEED,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "summary": summarize(ids, info),
    }


def write_manifest(payload, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{payload['name']}.json")
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    data = text.encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    train_info, val_info = scan("train"), scan("val")
    train_ids, val_ids = list(train_info), list(val_info)
    dev4_ids = pick_dev4(train_info)
    dev4_set = set(dev4_ids)
    train_minus = [iid for iid in train_ids if iid not in dev4_set]
    dev_vehicle = [iid for iid in dev4_ids if "车辆" in train_info[iid]["classes"]]

    specs = [
        ("official_train", train_ids, train_info, "data/rs_grounding_train_v1/train"),
        ("official_val", val_ids, val_info, "data/rs_grounding_train_v1/val"),
        ("dev4", dev4_ids, train_info, "official_train"),
        ("dev_vehicle", dev_vehicle, train_info, "dev4"),
        ("train_minus_dev4", train_minus, train_info, "official_train - dev4"),
    ]

    lines = []
    for name, ids, info, source in specs:
        payload = build_manifest(name, ids, info, source)
        digest = write_manifest(payload, args.out)
        lines.append(f"{digest}  {name}.json")
        summary = payload["summary"]
        print(f"[{name}] images={summary['images']} boxes={summary['boxes']}")
        print(f"    image_counts={summary['image_counts']}")

    with open(os.path.join(args.out, "SHA256SUMS"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(lines)) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
