"""Public-input-only model adapter contract; no GT argument or scoring imports."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path

from common import Resources, fresh_output, strict_json, write, sha, integer, number, valid_box, safe_child


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    path: Path
    width: int
    height: int


def check_predictions(images, queries, predictions):
    image_ids = {r.image_id for r in images}
    query_by_id = {q["query_id"]: q for q in queries}
    seen = set()
    for p in predictions:
        qid = p.get("query_id")
        if not integer(qid) or qid not in query_by_id or qid in seen:
            raise ValueError("Unknown or duplicate prediction query_id")
        seen.add(qid)
        q = query_by_id[qid]
        if p.get("type") != q["type"]:
            raise ValueError("Prediction type differs from input query")
        if q["type"] == "retrieval":
            ranking = p.get("ranking")
            if (not isinstance(ranking, list) or len(ranking) < min(5,len(image_ids))
                or any(not integer(i) or i not in image_ids for i in ranking) or len(set(ranking)) != len(ranking)):
                raise ValueError("Invalid Top-5 retrieval ranking")
        else:
            if not integer(p.get("image_id")) or p["image_id"] != q["image_id"]:
                raise ValueError("Grounding image differs from public query")
            if not integer(p.get("category_id")) or p["category_id"] not in (1,2,3,4):
                raise ValueError("Invalid category ID")
            if not number(p.get("score")) or not 0 <= p["score"] <= 1 or not valid_box(p.get("bbox")):
                raise ValueError("Invalid grounding confidence or xywh box")
    if seen != set(query_by_id):
        raise ValueError("Missing predictions")


def check_candidates(rows, image_ids):
    if not isinstance(rows, list):
        raise ValueError("Candidate image records must be a list")
    seen = set()
    for row in rows:
        iid = row.get("image_id")
        if not integer(iid) or iid not in image_ids or iid in seen:
            raise ValueError("Invalid or duplicate candidate image_id")
        seen.add(iid)
        items = row.get("candidates")
        if not isinstance(items, list):
            raise ValueError("Candidates must be a list")
        ids = set()
        for item in items:
            cid = item.get("candidate_id")
            if not isinstance(cid,str) or not cid or cid in ids:
                raise ValueError("Candidate IDs must be distinct nonempty strings per image")
            ids.add(cid)
            if not integer(item.get("category_id")) or item["category_id"] not in (1,2,3,4):
                raise ValueError("Invalid candidate category")
            if not valid_box(item.get("bbox")) or not number(item.get("score")) or not 0 <= item["score"] <= 1:
                raise ValueError("Invalid candidate geometry or confidence")
    if seen != set(image_ids):
        raise ValueError("Candidate file must explicitly include each image, even when its pool is empty")


def load_public_input(root):
    raw_images = strict_json(root / "info.json")
    queries = strict_json(root / "queries.json")
    images, ids = [], set()
    for row in raw_images:
        iid = row.get("image_id")
        if not integer(iid) or iid in ids:
            raise ValueError("Invalid public image IDs")
        ids.add(iid)
        for key in ("width","height"):
            if not integer(row.get(key)) or row[key] <= 0:
                raise ValueError("Invalid public image dimensions")
        path = safe_child(root / "images", row["file_name"])
        if not path.is_file():
            raise FileNotFoundError(path)
        images.append(ImageRecord(iid, path, row["width"], row["height"]))
    qids = set()
    for q in queries:
        if not integer(q.get("query_id")) or q["query_id"] in qids:
            raise ValueError("Invalid public query IDs")
        qids.add(q["query_id"])
        if q.get("type") not in ("retrieval","grounding") or not isinstance(q.get("text"),str) or not q["text"].strip():
            raise ValueError("Invalid public query")
        if q["type"] == "grounding" and (not integer(q.get("image_id")) or q["image_id"] not in ids):
            raise ValueError("Unknown public grounding image")
    return tuple(sorted(images,key=lambda r:r.image_id)), queries


def run(input_path, output_path, adapter_spec, config, budget_mb=256):
    resources = Resources(budget_mb)
    input_path = input_path.resolve()
    if output_path.resolve() == input_path or input_path in output_path.resolve().parents:
        raise ValueError("Never write into input_path")
    output = fresh_output(output_path)
    images, queries = load_public_input(input_path)
    module_name, separator, factory_name = adapter_spec.partition(":")
    if not separator:
        raise ValueError("Use module:factory for a trusted local adapter")
    module = importlib.import_module(module_name)
    adapter = getattr(module, factory_name)(config)
    adapter.prepare(images)
    rows = []
    if hasattr(adapter,"propose"):
        for image in images:
            rows.append({"image_id":image.image_id,"candidates":adapter.propose(image)})
            resources.check()
        check_candidates(rows, {r.image_id for r in images})
    by_id = {r.image_id:r for r in images}
    predictions, trace = [], []
    for q in queries:
        p = {"query_id":q["query_id"],"type":q["type"]}
        if q["type"] == "retrieval":
            p["ranking"] = adapter.retrieve(q["text"])
        else:
            result = adapter.ground(by_id[q["image_id"]],q["text"])
            p.update(image_id=q["image_id"],category_id=result["category_id"],score=result["score"],bbox=result["bbox"])
            trace.append({"query_id":q["query_id"],"candidate_id":result.get("candidate_id")})
        predictions.append(p)
        resources.check()
    check_predictions(images,queries,predictions)
    fingerprints = {"info_sha256":sha(input_path/"info.json"),"queries_sha256":sha(input_path/"queries.json"),
                    "images_sha256":{str(r.image_id):sha(r.path) for r in images}}
    write(output/"pred.json",predictions)
    write(output/"decision_trace.json",trace)
    if rows:
        write(output/"candidates.json",{"schema_version":"1.0","input_fingerprints":fingerprints,"images":rows})
    source = Path(inspect.getfile(module))
    report = {"adapter":adapter_spec,"adapter_source_sha256":sha(source),"config":config,
              "purpose":getattr(adapter,"purpose","model_inference"),"public_input":fingerprints,
              "images":len(images),"queries":len(queries),"gt_passed_to_runner":False,
              "resources":resources.report(),"prediction_sha256":sha(output/"pred.json")}
    write(output/"run.json",report)
    print({"purpose":report["purpose"],"queries":len(queries),"resources":report["resources"]},flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path",type=Path,required=True)
    parser.add_argument("--output-path",type=Path,required=True)
    parser.add_argument("--adapter",required=True)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--budget-mb",type=int,default=256,help="CPU RSS checkpoint budget; raise explicitly for cloud model inference")
    args=parser.parse_args()
    if args.budget_mb<1:
        parser.error("budget-mb must be positive")
    run(args.input_path,args.output_path,args.adapter,strict_json(args.config) if args.config else {},args.budget_mb)
