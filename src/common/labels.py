"""标签解析与类别映射（唯一来源，禁止其他脚本各自定义）。

训练 class_id 0..3，提交 category_id 1..4。
"""
import json
import os

CLASS_NAMES = ("油罐", "舰船", "车辆", "飞机")
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}
ID_TO_CATEGORY = {idx: idx + 1 for idx in range(len(CLASS_NAMES))}
CATEGORY_TO_ID = {cat: idx for idx, cat in ID_TO_CATEGORY.items()}


def bbox_xyxy(shape):
    """labelme 的 points 可能是 4 角点或 2 角点；统一取 min/max。"""
    points = shape.get("points") or []
    if len(points) < 2:
        raise ValueError(f"shape points < 2: {points!r}")
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid box: {x1, y1, x2, y2}")
    return [x1, y1, x2, y2]


def load_label(json_path):
    """读取一个 labelme JSON，返回规范化的图像与目标列表。"""
    with open(json_path, encoding="utf-8") as fh:
        row = json.load(fh)
    image_id = os.path.splitext(os.path.basename(json_path))[0]
    width = int(row.get("imageWidth") or 0)
    height = int(row.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"bad image size in {json_path}: {width}x{height}")
    objects = []
    for idx, shape in enumerate(row.get("shapes") or []):
        label = shape.get("label")
        if label not in CLASS_TO_ID:
            raise ValueError(f"unknown label {label!r} in {json_path}")
        class_id = CLASS_TO_ID[label]
        objects.append({
            "index": idx,
            "label": label,
            "class_id": class_id,
            "category_id": ID_TO_CATEGORY[class_id],
            "description": (shape.get("description") or "").strip(),
            "bbox_xyxy": bbox_xyxy(shape),
        })
    return {
        "image_id": str(image_id),
        "width": width,
        "height": height,
        "image_path": row.get("imagePath"),
        "objects": objects,
        "path": json_path,
    }


def split_label_dir(data_root, split):
    return os.path.join(data_root, split, "labels_json")


def list_image_ids(data_root, split):
    directory = split_label_dir(data_root, split)
    ids = [os.path.splitext(n)[0] for n in os.listdir(directory) if n.endswith(".json")]
    return sorted(ids, key=lambda value: (len(value), value))


def load_split(data_root, split, image_ids=None):
    ids = list_image_ids(data_root, split) if image_ids is None else list(image_ids)
    return [load_label(os.path.join(split_label_dir(data_root, split), f"{iid}.json"))
            for iid in ids]


def load_split_manifest(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
