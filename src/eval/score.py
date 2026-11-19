"""本地评分器：复现官方公式，支持 strict / any-dup 两种定位口径与 bootstrap CI。

本地协议：split 里每个标注框既生成一条 grounding 查询，也生成一条 retrieval 查询。
官方查询数少于标注框（示例 157/30），所以本地数字只用于模型间相对比较。
"""
import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.boxes import clip_xyxy, iou, xywh_to_xyxy  # noqa: E402
from common.labels import CLASS_NAMES, ID_TO_CLASS, load_label  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "rs_grounding_train_v1")


def total_score(r1, r5, a50, a75):
    return 0.1 * r1 + 0.2 * r5 + 0.6 * a50 + 0.1 * a75


def percentile(values, q):
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


def build_eval_set(manifest_path, label_split=None):
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if label_split is None:
        label_split = "val" if manifest.get("name") == "official_val" else "train"
    label_dir = os.path.join(DATA, label_split, "labels_json")
    grounding, retrieval = [], []
    for iid in manifest["images"]:
        label = load_label(os.path.join(label_dir, f"{iid}.json"))
        by_text = defaultdict(list)
        for obj in label["objects"]:
            by_text[obj["description"]].append(obj["bbox_xyxy"])
        for obj in label["objects"]:
            grounding.append({
                "query_id": f"g_{iid}_{obj['index']}",
                "image_id": str(iid),
                "class_id": obj["class_id"],
                "class_name": obj["label"],
                "description": obj["description"],
                "gt_bbox": obj["bbox_xyxy"],
                "gt_group": by_text[obj["description"]],
                "width": label["width"],
                "height": label["height"],
            })
            retrieval.append({
                "query_id": f"r_{iid}_{obj['index']}",
                "image_id": str(iid),
                "class_id": obj["class_id"],
                "description": obj["description"],
            })
    return manifest, grounding, retrieval


def read_pred(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise ValueError("pred.json must be a list")
    return {str(row.get("query_id")): row for row in rows if row.get("query_id") is not None}


def pred_to_xyxy(row, fmt):
    bbox = row.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    values = [float(v) for v in bbox]
    return xywh_to_xyxy(values) if fmt == "xywh" else values


def grounding_records(queries, pred, fmt):
    records = []
    missing = 0
    for q in queries:
        row = pred.get(q["query_id"])
        if not row or row.get("type") not in (None, "grounding"):
            missing += 1
            records.append({**q, "iou_strict": 0.0, "iou_any": 0.0, "has_pred": False})
            continue
        box = pred_to_xyxy(row, fmt)
        if box is None:
            missing += 1
            records.append({**q, "iou_strict": 0.0, "iou_any": 0.0, "has_pred": False})
            continue
        box = clip_xyxy(box, q["width"], q["height"])
        records.append({
            **q,
            "iou_strict": iou(box, q["gt_bbox"]),
            "iou_any": max(iou(box, gt) for gt in q["gt_group"]),
            "has_pred": True,
        })
    return records, missing


def retrieval_records(queries, pred):
    records, missing = [], 0
    for q in queries:
        row = pred.get(q["query_id"])
        if not row or row.get("type") not in (None, "retrieval"):
            missing += 1
            records.append({**q, "rank": None, "has_pred": False})
            continue
        ranking = [str(item) for item in (row.get("ranking") or [])]
        source = str(q["image_id"])
        rank = ranking.index(source) if source in ranking else None
        records.append({**q, "rank": rank, "has_pred": rank is not None})
    return records, missing


def grounding_metrics(records, key):
    n = len(records)
    hits50 = sum(1 for r in records if r[key] >= 0.5)
    hits75 = sum(1 for r in records if r[key] >= 0.75)
    return {"n": n, "acc50": hits50 / n if n else 0.0, "acc75": hits75 / n if n else 0.0}


def retrieval_metrics(records):
    n = len(records)
    r1 = sum(1 for r in records if r["rank"] == 0)
    r5 = sum(1 for r in records if r["rank"] is not None and r["rank"] < 5)
    return {"n": n, "r1": r1 / n if n else 0.0, "r5": r5 / n if n else 0.0}


def per_class(records, key):
    out = {}
    for class_id, name in ID_TO_CLASS.items():
        subset = [r for r in records if r["class_id"] == class_id]
        if subset:
            out[name] = grounding_metrics(subset, key)
    return out


def group_metrics(records, key=None, retrieval=False):
    """按 image_id 分组后取未加权平均（image-macro），避免大图/多目标图主导。"""
    groups = defaultdict(list)
    for record in records:
        groups[str(record["image_id"])].append(record)
    values = []
    for group in groups.values():
        if retrieval:
            m = retrieval_metrics(group)
            values.append((m["r1"], m["r5"]))
        else:
            m = grounding_metrics(group, key)
            values.append((m["acc50"], m["acc75"]))
    if not values:
        return {"images": 0}
    if retrieval:
        return {"images": len(values),
                "r1": sum(v[0] for v in values) / len(values),
                "r5": sum(v[1] for v in values) / len(values)}
    return {"images": len(values),
            "acc50": sum(v[0] for v in values) / len(values),
            "acc75": sum(v[1] for v in values) / len(values)}


def bootstrap(ground, retr, key, rounds=1000, seed=20261119):
    rng = random.Random(seed)
    totals, acc50s, acc75s, r1s, r5s = [], [], [], [], []
    for _ in range(rounds):
        g = [ground[rng.randrange(len(ground))] for _ in range(len(ground))]
        r = [retr[rng.randrange(len(retr))] for _ in range(len(retr))]
        gm = grounding_metrics(g, key)
        rm = retrieval_metrics(r)
        acc50s.append(gm["acc50"])
        acc75s.append(gm["acc75"])
        r1s.append(rm["r1"])
        r5s.append(rm["r5"])
        totals.append(total_score(rm["r1"], rm["r5"], gm["acc50"], gm["acc75"]))
    return {
        "total": [percentile(totals, 2.5), percentile(totals, 97.5)],
        "acc50": [percentile(acc50s, 2.5), percentile(acc50s, 97.5)],
        "acc75": [percentile(acc75s, 2.5), percentile(acc75s, 97.5)],
        "r1": [percentile(r1s, 2.5), percentile(r1s, 97.5)],
        "r5": [percentile(r5s, 2.5), percentile(r5s, 97.5)],
    }


def evaluate(manifest_path, pred_path, fmt="xywh", label_split=None, rounds=1000):
    manifest, gq, rq = build_eval_set(manifest_path, label_split)
    pred = read_pred(pred_path)
    g_records, g_missing = grounding_records(gq, pred, fmt)
    r_records, r_missing = retrieval_records(rq, pred)

    result = {
        "split": manifest.get("name"),
        "images": len(manifest["images"]),
        "protocol": "local: one grounding + one retrieval query per annotated object",
        "counts": {
            "grounding": len(gq), "retrieval": len(rq),
            "pred_missing_grounding": g_missing, "pred_missing_retrieval": r_missing,
        },
        "grounding": {
            "strict": grounding_metrics(g_records, "iou_strict"),
            "anydup": grounding_metrics(g_records, "iou_any"),
            "per_class_strict": per_class(g_records, "iou_strict"),
            "per_class_anydup": per_class(g_records, "iou_any"),
        },
        "retrieval": retrieval_metrics(r_records),
    }
    strict = result["grounding"]["strict"]
    anydup = result["grounding"]["anydup"]
    retr = result["retrieval"]
    result["total"] = {
        "strict": total_score(retr["r1"], retr["r5"], strict["acc50"], strict["acc75"]),
        "anydup": total_score(retr["r1"], retr["r5"], anydup["acc50"], anydup["acc75"]),
    }
    img_strict = group_metrics(g_records, "iou_strict")
    img_anydup = group_metrics(g_records, "iou_any")
    img_retr = group_metrics(r_records, retrieval=True)
    result["image_macro"] = {
        "grounding_strict": img_strict,
        "grounding_anydup": img_anydup,
        "retrieval": img_retr,
        "total_strict": total_score(img_retr["r1"], img_retr["r5"],
                                    img_strict["acc50"], img_strict["acc75"]),
        "total_anydup": total_score(img_retr["r1"], img_retr["r5"],
                                    img_anydup["acc50"], img_anydup["acc75"]),
    }
    result["ci95"] = {
        "strict": bootstrap(g_records, r_records, "iou_strict", rounds),
        "anydup": bootstrap(g_records, r_records, "iou_any", rounds),
    }
    return result


def print_report(result):
    print(f"split={result['split']} images={result['images']} "
          f"grounding={result['counts']['grounding']} retrieval={result['counts']['retrieval']}")
    print(f"missing: grounding={result['counts']['pred_missing_grounding']} "
          f"retrieval={result['counts']['pred_missing_retrieval']}")
    for mode in ("strict", "anydup"):
        g = result["grounding"][mode]
        print(f"[{mode:6}] Acc@0.5={g['acc50']:.4f} Acc@0.75={g['acc75']:.4f} "
              f"total={result['total'][mode]:.4f}")
    r = result["retrieval"]
    print(f"[retrieval] R@1={r['r1']:.4f} R@5={r['r5']:.4f}")
    im = result["image_macro"]
    print(f"[box-weighted] total={result['total']['strict']:.4f} "
          f"| [image-macro] total={im['total_strict']:.4f} "
          f"(A50={im['grounding_strict']['acc50']:.3f} R5={im['retrieval']['r5']:.3f})")
    print("per-class (strict):")
    for name, m in result["grounding"]["per_class_strict"].items():
        print(f"  {name:<4} n={m['n']:<5} Acc@0.5={m['acc50']:.4f} Acc@0.75={m['acc75']:.4f}")
    ci = result["ci95"]["strict"]
    print(f"CI95 total={ci['total'][0]:.3f}~{ci['total'][1]:.3f} "
          f"Acc@0.5={ci['acc50'][0]:.3f}~{ci['acc50'][1]:.3f}")


def self_test():
    value = total_score(0.30, 0.60, 98 / 157, 61 / 157)
    assert abs(value - 0.563376) < 5e-4, value
    print(f"SELF_TEST_PASS total={value:.6f}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", help="data/splits/*.json")
    parser.add_argument("--pred", help="pred.json")
    parser.add_argument("--out", help="write metrics json")
    parser.add_argument("--label-split", choices=["train", "val"], default=None)
    parser.add_argument("--pred-format", choices=["xywh", "xyxy"], default="xywh")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.split:
        parser.error("--split is required (or use --self-test)")

    result = evaluate(args.split, args.pred, args.pred_format, args.label_split, args.bootstrap)
    print_report(result)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
