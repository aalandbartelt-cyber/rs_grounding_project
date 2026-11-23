"""Score GeoGround's generated boxes using both frozen remote-sensing encoders."""
import gc
import json

import open_clip
import torch
from torch.nn.functional import normalize

from box_parser import parse
from common import ROOT, TrainArchive, crop, load_training_cache
from encode_retrieval import SPECS


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    cache = load_training_cache()
    answers = json.loads((ROOT / "features/geoground_answers.json").read_text(encoding="utf-8"))
    units = json.loads((ROOT / "results/geoground_coordinate_units.json").read_text())["units"]
    data, results = TrainArchive(), {}
    for name, (model_name, path) in SPECS.items():
        model, _, processor = open_clip.create_model_and_transforms(model_name, pretrained=None)
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = state.get("state_dict", state.get("model", state))
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
        del state
        model = model.cuda().eval()
        text_features = torch.load(ROOT / f"features/{name}_text.pt", weights_only=True)["text"]
        scores = {}
        for key, answer in answers.items():
            image_id, query = answer["image"], answer["query"]
            row = cache["records"][image_id]
            box = parse(answer["answer"], row["width"], row["height"], units)
            if box is None:
                continue
            image = data.image(row["label"])
            batch = torch.stack([processor(crop(image, box, scale)) for scale in (1.0, 1.5)]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                visuals = normalize(model.encode_image(batch).float(), dim=-1).cpu()
            scores[key] = float((visuals @ text_features[query]).mean())
        results[name] = scores
        print(f"{name}: scored {len(scores)} GeoGround region proposals", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    torch.save(results, ROOT / "features/geoground_region_scores.pt")
    data.audit(ROOT / "results/GeoGround_region_encoding_data_access.json")
    data.close()


if __name__ == "__main__":
    main()
