"""Export the frozen v2 split without modifying images, labels or evaluation GT."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "z_evaluation"))
from evalkit import bbox_from_points, jpeg_size, read, sha, source_inventory, write


def assert_disjoint(groups):
    seen = set()
    for name, keys in groups.items():
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate source keys: {name}")
        overlap = seen & set(keys)
        if overlap:
            raise ValueError(f"Split leakage in {name}: {sorted(overlap)}")
        seen.update(keys)


def contained(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"Path escapes dataset: {relative}")
    return path


def clip_training_box(box, width, height):
    x, y, w, h = box
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Training box has no visible area")
    return [x1, y1, x2 - x1, y2 - y1]


def export(data, output):
    data, output = data.resolve(), output.resolve()
    if output == data or data in output.parents or output == REPO or REPO in output.parents:
        raise ValueError("Generated private data must be outside the source dataset and repository")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty export directory")
    splits = REPO / "z_evaluation" / "splits"
    manifest = read(splits / "manifest.json")
    groups = {"train": manifest["train_keys"], "dev": manifest["dev_keys"],
              "official_val": manifest["official_val_keys"]}
    assert_disjoint(groups)
    if source_inventory(data) != manifest["source_sha256"]:
        raise ValueError("Source dataset does not match frozen v2 fingerprints")
    mapping = read(data / "label_map.json")
    categories = [{"id": cid, "name": name} for name, cid in sorted(mapping.items(), key=lambda kv: kv[1])]
    report = {"manifest_sha256": sha(splits / "manifest.json"), "source_files_verified": len(manifest["source_sha256"]),
              "source_data": str(data), "source_unchanged": True, "splits": {}, "training_box_adjustments": []}
    files = {"train": "train_images.json", "dev": "dev_images.json", "official_val": "official_val_images.json"}
    for split, filename in files.items():
        rows = read(splits / filename)
        if [row["source_key"] for row in rows] != groups[split]:
            raise ValueError(f"Split list differs from manifest: {split}")
        coco = {"images": [], "annotations": [], "categories": categories,
                "info": {"split": split, "frozen_manifest_sha256": report["manifest_sha256"],
                         "box_policy": "clip_training_copy_only" if split == "train" else "original_json_float"}}
        pairs, category_objects, category_images, counts = [], collections.Counter(), collections.Counter(), []
        for image_id, row in enumerate(rows, 1):
            image = contained(data, row["image_path"])
            label = contained(data, row["label_json_path"])
            width, height = jpeg_size(image)
            if (width, height) != (row["width"], row["height"]):
                raise ValueError("Image dimension mismatch")
            coco["images"].append({"id": image_id, "file_name": image.as_posix(),
                                   "width": width, "height": height, "source_key": row["source_key"]})
            shapes = read(label)["shapes"]
            if len(shapes) != row["object_count"]:
                raise ValueError("Object count mismatch")
            per_image = set()
            counts.append(len(shapes))
            for shape_index, shape in enumerate(shapes):
                if shape["shape_type"] != "rectangle":
                    raise ValueError("Only rectangle annotations are supported")
                original = bbox_from_points(shape["points"])
                box = clip_training_box(original, width, height) if split == "train" else original
                cid = mapping[shape["label"]]
                annotation_id = len(coco["annotations"]) + 1
                coco["annotations"].append({"id": annotation_id, "image_id": image_id, "category_id": cid,
                                            "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
                category_objects[cid] += 1
                per_image.add(cid)
                if box != original:
                    report["training_box_adjustments"].append({"source_key": row["source_key"],
                        "shape_index": shape_index, "original_bbox": original, "training_bbox": box})
                if split == "train":
                    pairs.append({"source_key": row["source_key"], "image_path": image.as_posix(),
                                  "shape_index": shape_index, "text": shape["description"], "category_id": cid,
                                  "bbox": box, "original_bbox": original, "width": width, "height": height})
            category_images.update(per_image)
        destination = output / "train" if split == "train" else output / "evaluation_only" / split
        write(destination / "annotations.coco.json", coco)
        if split == "train":
            write(destination / "text_region_pairs.json", pairs)
        report["splits"][split] = {"images": len(rows), "objects": len(coco["annotations"]),
            "objects_by_category": dict(sorted(category_objects.items())),
            "images_by_category": dict(sorted(category_images.items())), "max_objects_per_image": max(counts),
            "empty_images": sum(n == 0 for n in counts)}
    if source_inventory(data) != manifest["source_sha256"]:
        raise ValueError("Source data changed during export")
    report["generated_sha256"] = {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*.json"))}
    write(output / "export_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.data, args.out)
