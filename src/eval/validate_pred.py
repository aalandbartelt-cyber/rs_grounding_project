"""提交前 schema 校验：对照本地 split 生成的 query 清单，检查 pred.json 的完整性/合法性。"""
import argparse
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import build_eval_set  # noqa: E402


def validate(manifest_path, pred_path, max_report=20, allow_out_of_bounds=False):
    _, gq, rq = build_eval_set(manifest_path)
    expected = {q["query_id"]: q for q in gq}
    expected.update({q["query_id"]: q for q in rq})

    with open(pred_path, encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        return ["pred.json 顶层必须是 list"], {}, {}

    errors, seen = [], Counter()
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"[{idx}] 不是对象")
            continue
        qid = str(row.get("query_id"))
        seen[qid] += 1
        if qid not in expected:
            errors.append(f"[{idx}] 未知 query_id={qid!r}")
            continue
        exp = expected[qid]
        qtype = row.get("type")
        want = "grounding" if qid.startswith("g_") else "retrieval"
        if qtype != want:
            errors.append(f"[{idx}] {qid} type={qtype!r}，应为 {want!r}")
            continue

        if qid.startswith("g_"):
            if str(row.get("image_id")) != str(exp["image_id"]):
                errors.append(f"[{idx}] {qid} image_id 不匹配")
            category = row.get("category_id")
            if category not in (1, 2, 3, 4):
                errors.append(f"[{idx}] {qid} category_id={category!r}，应为 1..4")
            score = row.get("score")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                errors.append(f"[{idx}] {qid} score 非有限数值")
            bbox = row.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"[{idx}] {qid} bbox 必须是长度 4 的 list")
            else:
                try:
                    x, y, w, h = [float(v) for v in bbox]
                except (TypeError, ValueError):
                    errors.append(f"[{idx}] {qid} bbox 含非数值")
                    continue
                if not all(math.isfinite(v) for v in (x, y, w, h)):
                    errors.append(f"[{idx}] {qid} bbox 含 NaN/Inf")
                if w <= 0 or h <= 0:
                    errors.append(f"[{idx}] {qid} bbox w/h 必须 >0（得到 {w},{h}）")
                if not allow_out_of_bounds:
                    if x < 0 or y < 0 or x + w > exp["width"] + 1e-6 or y + h > exp["height"] + 1e-6:
                        errors.append(f"[{idx}] {qid} bbox 越界")
        else:
            ranking = row.get("ranking")
            if not isinstance(ranking, list):
                errors.append(f"[{idx}] {qid} ranking 必须是 list")
                continue
            if len(ranking) < 5:
                errors.append(f"[{idx}] {qid} ranking 少于 5 个")
            normalized = [str(v) for v in ranking]
            if len(set(normalized)) != len(normalized):
                errors.append(f"[{idx}] {qid} ranking 有重复")
            if any(not v for v in normalized):
                errors.append(f"[{idx}] {qid} ranking 含空值")

    duplicates = [qid for qid, count in seen.items() if count > 1]
    missing = [qid for qid in expected if qid not in seen]
    extras = [qid for qid in seen if qid not in expected]

    summary = {
        "expected": len(expected),
        "rows": len(rows),
        "missing": len(missing),
        "extras": len(extras),
        "duplicates": len(duplicates),
        "errors": len(errors),
    }
    return errors, summary, {"missing": missing[:max_report], "extras": extras[:max_report],
                             "duplicates": duplicates[:max_report]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--max-report", type=int, default=20)
    parser.add_argument("--allow-out-of-bounds", action="store_true")
    args = parser.parse_args()

    errors, summary, samples = validate(args.split, args.pred, args.max_report,
                                        args.allow_out_of_bounds)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for key, values in samples.items():
        if values:
            print(f"{key} sample: {values}")
    for line in errors[: args.max_report]:
        print("ERROR", line)
    if errors:
        print(f"VALIDATION_FAILED ({len(errors)} errors)")
        return 1
    print("VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
