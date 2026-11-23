"""Frozen Qwen semantic evidence; no target annotations enter model prompts."""
import json

import torch

from data import BASE, Data
from common import TrainArchive, crop, write_json
from rerank_qwen import panel
from vlm import Qwen


def main():
    torch.set_num_threads(4)
    data = Data()
    destination = BASE / "training_v1/features/qwen_semantic.json"
    result = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else {"grounding":{},"retrieval":{}}
    model, archive = Qwen(), TrainArchive()
    for number,(key,row) in enumerate(data.grounding.items(),1):
        previous = result["grounding"].get(key,[])
        if len(previous) != len(row["boxes"]):
            image = archive.image(data.records[row["image"]]["label"])
            for j in range(len(previous),len(row["boxes"])):
                view = crop(image,row["boxes"][j],1.25)
                prompt = ("Does this cropped satellite image show a complete civilian passenger aircraft "
                          "that matches the following description? Judge visible features only. "
                          "Answer only Yes or No. Description: " + data.texts[row["query"]])
                previous.append(model.match_score(prompt,view))
                result["grounding"][key]=previous
                write_json(destination,result)
        if number % 10 == 0:
            print(f"Qwen region evidence {number}/{len(data.grounding)}",flush=True)
    jobs=[]
    for ids in (data.fit_ids,data.dev_ids):
        queries=sorted(set(data.texts.index(t) for i in ids for t in data.records[i]["texts"]))
        jobs.extend((q,i) for q in queries for i in ids)
    for number,(q,i) in enumerate(jobs,1):
        key=f"{q}:{i}"
        if key not in result["retrieval"]:
            record=data.records[i]
            image=archive.image(record["label"])
            # A complete-gallery pass can recover images missing from the old top-five shortlist.
            prompt=("Is there at least one civilian passenger aircraft in this satellite image that "
                    "matches the following description? Compare visible engine count, paint, wing "
                    "arrangement and surroundings. Answer only Yes or No. Description: "+data.texts[q])
            result["retrieval"][key]=model.match_score(prompt,image)
            write_json(destination,result)
        if number % 25 == 0 or number == len(jobs):
            print(f"Qwen complete-gallery evidence {number}/{len(jobs)}",flush=True)
    archive.audit(BASE / "training_v1/results/qwen_evidence_data_access.json")
    archive.close()


if __name__ == "__main__":
    main()
