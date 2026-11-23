"""Deterministic low-memory training previews; no automatic semantic approval."""
from __future__ import annotations

import argparse
import collections
import html
import math
from pathlib import Path
import shutil
import subprocess

from common import REPO, Resources, fresh_output, read, write, sha, valid_box
from geometry import transform, geometry_is_joint_safe
from evalkit import load_data, source_inventory, jpeg_size

OPERATIONS = ("original", "brightness_095", "resample_blur_085", "jpeg_80", "flip_x", "rotate90_ccw", "crop_keep_all")


def render(jobs, output):
    if not shutil.which("pwsh") and not shutil.which("powershell"):
        raise RuntimeError("Preview renderer uses native Windows System.Drawing; generate on Windows or implement an audited Linux renderer")
    write(output / "render_jobs.json", jobs)
    shell = shutil.which("pwsh") or shutil.which("powershell")
    command = [shell, "-NoProfile", "-File", str(Path(__file__).with_name("render_windows.ps1")),
               "-JobsPath", str(output / "render_jobs.json"), "-ReportPath", str(output / "render_resources.json")]
    subprocess.run(command, check=True)


def clip(box, width, height):
    x, y, w, h = box
    x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Invisible source box")
    return [x1, y1, x2-x1, y2-y1]


def crop_window(boxes, width, height):
    margin = 16
    left = max(0, math.floor(min(b[0] for b in boxes)) - margin)
    top = max(0, math.floor(min(b[1] for b in boxes)) - margin)
    right = min(width, math.ceil(max(b[0]+b[2] for b in boxes)) + margin)
    bottom = min(height, math.ceil(max(b[1]+b[3] for b in boxes)) + margin)
    return [left, top, right-left, bottom-top]


def preview(data, output, per_class):
    resources = Resources()
    data = data.resolve()
    if data in output.resolve().parents or data == output.resolve():
        raise ValueError("Do not write previews into source data")
    manifest_path = REPO / "z_evaluation" / "splits" / "manifest.json"
    manifest = read(manifest_path)
    if source_inventory(data) != manifest["source_sha256"]:
        raise ValueError("Frozen source inventory mismatch")
    records, _, _ = load_data(data)
    train = [r for r in records if r["key"] in set(manifest["train_keys"]) and r["objects"]]
    picked = []
    for category in range(1,5):
        candidates = [r for r in train if category in {o["category_id"] for o in r["objects"]}]
        # Dense examples first so previews expose small-object geometry failures.
        candidates.sort(key=lambda r: (-sum(o["category_id"] == category for o in r["objects"]), r["key"]))
        for record in candidates[:per_class]:
            if record["key"] not in {r["key"] for r in picked}:
                picked.append(record)
    output = fresh_output(output)
    jobs, items = [], []
    for record in picked:
        width, height = record["width"], record["height"]
        boxes = [clip(o["bbox"], width, height) for o in record["objects"]]
        window = crop_window(boxes, width, height)
        for operation in OPERATIONS:
            mapping, new_w, new_h = transform(operation, width, height, window)
            changed = [mapping.box(b) for b in boxes]
            for source_box, changed_box in zip(boxes, changed):
                if not valid_box(changed_box):
                    raise ValueError("Invalid transformed box")
                x, y, w, h = changed_box
                if x < -1e-6 or y < -1e-6 or x+w > new_w+1e-6 or y+h > new_h+1e-6:
                    raise ValueError("Transformed box leaves output image")
                if any(abs(a-b) > 1e-6 for a,b in zip(source_box, mapping.inverse().box(changed_box))):
                    raise ValueError("Box mapping roundtrip failed")
            stem = record["key"].replace("/", "_") + "__" + operation
            relative = "images/" + stem + ".jpg"
            overlay = "overlays/" + stem + ".png"
            (output / "overlays").mkdir(exist_ok=True)
            source = data / record["split"] / "images" / (record["stem"] + ".jpg")
            jobs.append({"source": str(source), "destination": str(output / relative), "overlay": str(output / overlay),
                         "operation": operation, "crop": window, "width": new_w, "height": new_h, "boxes": changed})
            items.append({"source_key": record["key"], "source_sha256": sha(source), "image": relative, "overlay": overlay,
                "operation": operation, "width": new_w, "height": new_h, "affine": mapping.values(),
                "inverse_affine": mapping.inverse().values(), "boxes": changed, "original_clipped_boxes": boxes,
                "category_ids": [o["category_id"] for o in record["objects"]],
                "text_examples": [o["text"] for o in record["objects"][:2]],
                "joint_geometry_auto_eligible": geometry_is_joint_safe(operation),
                "review_status": "pending_semantic_review",
                "scope": "detection_geometry_preview_only" if not geometry_is_joint_safe(operation) else "attribute_preservation_preview"})
    render(jobs, output)
    for item in items:
        if jpeg_size(output / item["image"]) != (item["width"], item["height"]):
            raise ValueError("Saved image dimensions differ from transformed labels")
        item["generated_sha256"] = sha(output / item["image"])
    write(output / "preview_manifest.json", {"frozen_manifest_sha256": sha(manifest_path), "items": items})
    cards = []
    for item in items:
        label = f"{item['source_key']} / {item['operation']} / {item['scope']}"
        text = "\n".join(item["text_examples"])
        cards.append(f'<article><h3>{html.escape(label)}</h3><a href="{item["overlay"]}"><img loading="lazy" src="{item["overlay"]}"></a><p>{html.escape(text)}</p><p>语义审核：待确认；几何自动检查通过。</p></article>')
    page = '<!doctype html><meta charset="utf-8"><title>训练增强审核预览</title><style>body{font:16px sans-serif;margin:24px;background:#f5f5f5}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article{background:white;padding:16px}img{width:100%;max-height:640px;object-fit:contain}p{line-height:1.6}</style><h1>训练增强审核预览</h1><p>仅训练原图。绿色框与像素已同步转换；几何增强不自动用于中文图文定位。未启动训练。</p><main>' + ''.join(cards) + '</main>'
    (output / "index.html").write_text(page, encoding="utf-8")
    report = {"training_source_images": len(picked), "previews": len(items), "all_box_roundtrips_passed": True,
              "all_saved_dimensions_passed": True, "all_sources_in_train": True,
              "source_unchanged": source_inventory(data) == manifest["source_sha256"],
              "semantic_review": "pending; geometry checks do not establish text correctness",
              "resources": resources.report(), "renderer_resources": read(output / "render_resources.json")}
    write(output / "report.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=1)
    args = parser.parse_args()
    if args.per_class < 1 or args.per_class > 3:
        parser.error("Use 1..3 source images per category to keep previews small")
    preview(args.data, args.out, args.per_class)
