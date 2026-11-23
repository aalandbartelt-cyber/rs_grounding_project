"""Generate GeoGround proposals from original-training grounding query inputs."""
import json

import torch

from common import ROOT, TrainArchive, write_json
from vlm import GeoGround


def main():
    torch.set_num_threads(4)
    jobs = json.loads((ROOT / "features/jobs.json").read_text(encoding="utf-8"))["grounding"]
    translations = json.loads((ROOT / "features/translations.json").read_text(encoding="utf-8"))
    output = ROOT / "features/geoground_answers.json"
    answers = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    model, data = GeoGround(), TrainArchive()
    for number, job in enumerate(jobs, 1):
        key = f"{job['query']}:{job['image']}"
        if key not in answers:
            image = data.image(job["label"])
            result = model.answer(translations[job["text"]], image)
            answers[key] = {"answer": result, "width": image.width, "height": image.height,
                            "query": job["query"], "image": job["image"]}
            write_json(output, answers)
        print(f"GeoGround {number}/{len(jobs)}", flush=True)
    data.audit(ROOT / "results/GeoGround_data_access.json")
    data.close()


if __name__ == "__main__":
    main()
