"""Independent GT-reading diagnostics; never called by the model adapter."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re

from common import Resources, KEYWORDS, read, strict_json, write, sha, iou, side_bin, fresh_output
from contract import load_public_input, check_candidates
from evalkit import evaluate, aggregate


def box_matches(a,b):
    return all(abs(x-y) <= 1e-5 for x,y in zip(a,b))


def grounding_error(prediction, overlap, pool):
    if prediction is None:
        return "missing_prediction"
    if overlap >= 0.5:
        return "correct_at_05"
    if pool is None:
        return "failure_candidate_pool_unavailable"
    if not pool:
        return "no_candidates"
    if max(pool) < 0.5:
        return "candidate_coverage_miss"
    return "selection_or_refinement_failure"


def diagnose(gt, predictions, queries, candidate_rows=None):
    result = evaluate(gt,predictions)
    by_q={q["query_id"]:q for q in queries}
    if set(by_q) != {a["query_id"] for a in gt["answers"]}:
        raise ValueError("Public query IDs do not match private protocol")
    pred={p["query_id"]:p for p in predictions}
    pools={r["image_id"]:r["candidates"] for r in candidate_rows} if candidate_rows is not None else None
    answers={a["query_id"]:a for a in gt["answers"]}
    density=collections.Counter(a["image_id"] for a in gt["answers"] if a["type"]=="grounding")
    errors,ret_errors=collections.Counter(),collections.Counter()
    group_rows=collections.defaultdict(list)
    coverage={"05":[],"75":[],"category05":[],"category75":[]}
    diagnostic=[]
    for detail in result["details"]:
        a=answers[detail["query_id"]]
        q=by_q[a["query_id"]]
        if q["type"] != a["type"] or (q["type"]=="grounding" and q["image_id"] != a["image_id"]):
            raise ValueError("Public query metadata differs from GT")
        p=pred.get(a["query_id"])
        row=dict(detail)
        row["short_side_bin"]=side_bin(a["bbox"])
        row["scene_density"]="dense_ge20" if density[a["image_id"]]>=20 else "sparse_lt20"
        row["text_flags"]={name:bool(re.search(pattern,q["text"])) for name,pattern in KEYWORDS.items()}
        row["same_text_multiple_targets"]=len(a["alternative_bboxes"])>1
        if a["type"]=="grounding":
            items=pools[a["image_id"]] if pools is not None else None
            overlaps=[iou(c["bbox"],a["bbox"]) for c in items] if items is not None else None
            row["error"]=grounding_error(p,detail["iou"],overlaps)
            errors[row["error"]]+=1
            if items is not None:
                best=max(overlaps,default=0.0)
                best_cat=max((v for v,c in zip(overlaps,items) if c["category_id"]==a["category_id"]),default=0.0)
                row.update(candidate_count=len(items),candidate_best_iou=best,category_candidate_best_iou=best_cat,
                    final_box_in_candidate_pool=bool(p and any(box_matches(p["bbox"],c["bbox"]) for c in items)))
                coverage["05"].append(best>=0.5); coverage["75"].append(best>=0.75)
                coverage["category05"].append(best_cat>=0.5); coverage["category75"].append(best_cat>=0.75)
        else:
            ranking=p["ranking"] if p else []
            row["error"]=("missing_prediction" if p is None else "top1_correct" if detail["hit1"]
                           else "top5_only" if detail["hit5"] else "outside_top5")
            ret_errors[row["error"]]+=1
            row["target_rank"]=ranking.index(a["image_id"])+1 if a["image_id"] in ranking else None
        diagnostic.append(row)
        for key in ("short_side_bin","scene_density"):
            group_rows[f"{key}/{row[key]}"].append(detail)
        for key,value in row["text_flags"].items():
            group_rows[f"{key}/{str(value).lower()}"].append(detail)
        group_rows[f"source_image/{a['source_key']}"].append(detail)
    image_metrics={key.split("/",1)[1]:aggregate(rows) for key,rows in group_rows.items() if key.startswith("source_image/")}
    image_totals=[v["total"] for v in image_metrics.values() if v["total"] is not None]
    result["diagnostics"]={"grounding_errors":dict(errors),"retrieval_errors":dict(ret_errors),
        "candidate_coverage":{key:{"hits":sum(items),"queries":len(items),"rate":sum(items)/len(items) if items else None} for key,items in coverage.items()},
        "image_balanced_total":sum(image_totals)/len(image_totals) if image_totals else None,
        "by_image":image_metrics,"by_group":{key:aggregate(rows) for key,rows in group_rows.items() if not key.startswith("source_image/")},
        "query_details":diagnostic,
        "limitations":["Keyword flags are heuristic, not validated language-understanding labels",
                       "Coverage is an upper bound only when final boxes are selected unchanged from the recorded pool",
                       "Selection-or-refinement failure does not identify language misunderstanding as the unique cause",
                       "Short-side bins use original pixels; dense means at least20 ground-truth targets in a queried image",
                       "Primary score and its category policy remain unchanged"]}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path",type=Path,required=True)
    parser.add_argument("--gt",type=Path,required=True)
    parser.add_argument("--pred",type=Path,required=True)
    parser.add_argument("--candidates",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    resources=Resources()
    output=fresh_output(args.out)
    images,queries=load_public_input(args.input_path.resolve())
    gt=read(args.gt)
    if gt["gallery"] != read(args.input_path/"info.json"):
        raise ValueError("Input gallery differs from private GT")
    rows=None
    if args.candidates:
        candidate=strict_json(args.candidates)
        fingerprints=candidate["input_fingerprints"]
        if (fingerprints["info_sha256"] != sha(args.input_path/"info.json") or
            fingerprints["queries_sha256"] != sha(args.input_path/"queries.json") or
            fingerprints["images_sha256"] != {str(r.image_id):sha(r.path) for r in images}):
            raise ValueError("Candidate cache fingerprint mismatch")
        rows=candidate["images"]
        check_candidates(rows,{r.image_id for r in images})
    report=diagnose(gt,strict_json(args.pred),queries,rows)
    input_evidence={"info":sha(args.input_path/"info.json"),"queries":sha(args.input_path/"queries.json"),
                    "images":{str(r.image_id):sha(r.path) for r in images}}
    input_sha=hashlib.sha256(json.dumps(input_evidence,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    report["provenance"]={"gt_sha256":sha(args.gt),"pred_sha256":sha(args.pred),"input_sha256":input_sha,
                          "evaluator_sha256":sha(Path(__file__).parents[2]/"z_evaluation"/"evalkit.py")}
    report["resources"]=resources.report()
    write(output/"diagnostics.json",report)
    lines=["# 预测诊断报告","","主分由原冻结评分器计算；分组与候选覆盖只作诊断。","",
           f"协议：{report['protocol']}；主分（100制）：{report['primary']['total_percent']:.4f}","",
           "## 定位错误统计","","| 类型 | 查询数 |","|---|---:|"]
    lines += [f"| {key} | {value} |" for key,value in report["diagnostics"]["grounding_errors"].items()]
    lines += ["","## 类别成绩","","| 类别ID | R1 | R5 | Acc0.5 | Acc0.75 | 总分 |","|---|---:|---:|---:|---:|---:|"]
    for key,value in report["by_category"].items():
        def pct(x): return f"{x*100:.2f}" if x is not None else "无样本"
        lines.append("| "+" | ".join([key]+[pct(value[k]) for k in ("R1","R5","Acc05","Acc75","total")])+" |")
    lines += ["","候选存在却预测失败只表示选择或细化路径有问题；语义原因需要检查错例。",
              "多答案放宽、类别门控、图片平衡分数均不替代主分。", "", "资源："+str(report["resources"])]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print({"primary":report["primary"],"grounding_errors":report["diagnostics"]["grounding_errors"],"resources":report["resources"]},flush=True)


if __name__=="__main__":
    main()
