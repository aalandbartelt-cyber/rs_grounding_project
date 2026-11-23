"""Encode GeoGround predictions for a supervised training-only fusion head."""
import gc
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "code"))

import open_clip
import torch
from torch.nn.functional import normalize

from box_parser import parse
from common import TrainArchive, crop, digest
from encode_retrieval import SPECS


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    destination = BASE / "training_v1/features/geoground_visual.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    answer_path = BASE / "features/geoground_answers.json"
    signature = digest(answer_path)
    if destination.exists() and torch.load(destination, weights_only=True)["signature"] == signature:
        print("Reusing frozen GeoGround region visual features", flush=True)
        return
    answers = json.loads(answer_path.read_text(encoding="utf-8"))
    units = json.loads((BASE / "results/geoground_coordinate_units.json").read_text())["units"]
    data = TrainArchive()
    records = __import__("common").load_training_cache()["records"]
    output = {"signature": signature}
    for name, (architecture, checkpoint) in SPECS.items():
        model, _, transform = open_clip.create_model_and_transforms(architecture, pretrained=None)
        state = torch.load(checkpoint, weights_only=True, mmap=True, map_location="cpu")
        state = state.get("state_dict", state.get("model", state))
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
        del state
        model = model.cuda().eval()
        visual = {}
        for key, row in answers.items():
            label = records[row["image"]]["label"]
            image = data.image(label)
            box = parse(row["answer"], image.width, image.height, units)
            if box is None:
                continue
            batch = torch.stack([transform(crop(image, box, scale)) for scale in (1., 1.5)]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                embedding = normalize(model.encode_image(batch).float(), dim=-1).cpu()
            visual[key] = embedding.mean(0)
        output[name] = visual
        print(f"Encoded {len(visual)} {name} GeoGround proposal features", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    torch.save(output, destination)
    data.audit(BASE / "training_v1/results/region_features_data_access.json")
    data.close()


if __name__ == "__main__":
    main()
