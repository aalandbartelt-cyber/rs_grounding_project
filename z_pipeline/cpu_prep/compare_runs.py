"""Paired source-image bootstrap, B minus A, for identical frozen evaluation protocols."""
from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path
import random

from common import Resources, read, write, fresh_output
from evalkit import WEIGHTS, aggregate


def percentile(items,p):
    items=sorted(items)
    index=(len(items)-1)*p
    lo,hi=math.floor(index),math.ceil(index)
    return items[lo]+(items[hi]-items[lo])*(index-lo)


def compare(a,b,samples=1000,seed=20261123,allow_input_shift=False):
    if a["protocol"] != b["protocol"]:
        raise ValueError("Cannot compare different protocols")
    for key in ("gt_sha256","evaluator_sha256"):
        if a["provenance"].get(key) != b["provenance"].get(key) or not a["provenance"].get(key):
            raise ValueError("Comparison requires the same GT and evaluator fingerprint")
    if not a["provenance"].get("input_sha256") or not b["provenance"].get("input_sha256"):
        raise ValueError("Input fingerprints are required; use the diagnostics reports")
    shifted=a["provenance"]["input_sha256"] != b["provenance"]["input_sha256"]
    if shifted and not allow_input_shift:
        raise ValueError("Input images differ; explicitly allow an input-shift contrast instead of a model comparison")
    def keys(rows):
        return {(r["query_id"],r["type"],r["source_key"],r["category_id"]) for r in rows}
    if keys(a["details"])!=keys(b["details"]) or len(a["details"])!=len(b["details"]):
        raise ValueError("Query identities differ")
    groups=[]
    for result in (a,b):
        grouped=collections.defaultdict(list)
        for row in result["details"]:
            grouped[row["source_key"]].append(row)
        groups.append(grouped)
    sources=sorted(groups[0])
    stats=[]
    for grouped in groups:
        per_image={}
        for source in sources:
            rows=grouped[source]
            r=[x for x in rows if x["type"]=="retrieval"]
            g=[x for x in rows if x["type"]=="grounding"]
            per_image[source]=[len(r),len(g),sum(x["hit1"] for x in r),sum(x["hit5"] for x in r),
                               sum(x["acc05"] for x in g),sum(x["acc75"] for x in g)]
        stats.append(per_image)
    rng=random.Random(seed)
    values={key:[] for key in (*WEIGHTS,"total")}
    for _ in range(samples):
        draw=[rng.choice(sources) for _ in sources]
        metrics=[]
        for per_image in stats:
            nr,ng,h1,h5,a5,a75=[sum(per_image[key][i] for key in draw) for i in range(6)]
            if not nr or not ng:
                break
            row={"R1":h1/nr,"R5":h5/nr,"Acc05":a5/ng,"Acc75":a75/ng}
            row["total"]=sum(WEIGHTS[k]*row[k] for k in WEIGHTS)
            metrics.append(row)
        if len(metrics)==2:
            for key in values:
                values[key].append(metrics[1][key]-metrics[0][key])
    left,right=aggregate(a["details"]),aggregate(b["details"])
    return {"protocol":a["protocol"],"difference":"B minus A; scores expressed as fractions",
        "comparison_kind":"input_shift_contrast" if shifted else "same_input_model_comparison",
        "point_difference":{k:right[k]-left[k] for k in values},
        "paired_image_bootstrap_95pct":{k:[percentile(v,0.025),percentile(v,0.975)] if v else None for k,v in values.items()},
        "source_images":len(sources),"requested_samples":samples,"valid_samples":len(values["total"]),"seed":seed,
        "scope":"Paired local image sampling only; does not estimate training randomness or hidden-test shift",
        "provenance":{"gt_sha256":a["provenance"]["gt_sha256"],"evaluator_sha256":a["provenance"]["evaluator_sha256"]}}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a",type=Path,required=True)
    parser.add_argument("--b",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--allow-input-shift",action="store_true")
    args=parser.parse_args()
    resources=Resources()
    output=fresh_output(args.out)
    report=compare(read(args.a),read(args.b),allow_input_shift=args.allow_input_shift)
    report["resources"]=resources.report()
    write(output/"comparison.json",report)
    print(report,flush=True)
