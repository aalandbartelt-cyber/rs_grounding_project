"""Encode only original-training images and DINO proposals with remote-sensing CLIP."""
import argparse
import gc
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import open_clip
import torch
from torch.nn.functional import normalize

from common import ROOT, TrainArchive, crop, digest, load_training_cache, write_json

SPECS = {
    "RemoteCLIP": ("ViT-L-14", ROOT / "weights/RemoteCLIP/RemoteCLIP-ViT-L-14.pt"),
    "GeoRSCLIP": ("ViT-L-14-336", ROOT / "weights/GeoRSCLIP/ckpt/RS5M_ViT-L-14-336.pt"),
}


@torch.inference_mode()
def encode(name):
    model_name, path = SPECS[name]
    output = ROOT / f"features/{name}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = load_training_cache()
    signature = {"checkpoint_sha256": digest(path), "labels": [r["label"] for r in cache["records"]],
                 "views": ["whole", "tight_dino_proposals", "context_dino_proposals_1.5"],
                 "ground_truth_crops_encoded": False}
    if output.exists() and torch.load(output, weights_only=True)["signature"] == signature:
        print(f"Reusing {name} original-training features", flush=True)
        return
    model, _, processor = open_clip.create_model_and_transforms(model_name, pretrained=None)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if "model" in state:
        state = state["model"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    del state
    model = model.cuda().eval()
    data = TrainArchive()
    records, visuals = [], []
    offset = 0
    for number, row in enumerate(cache["records"], 1):
        image = data.image(row["label"])
        views = [image]
        for scale in (1.0, 1.5):
            views.extend(crop(image, box, scale) for box in row["boxes"])
        for start in range(0, len(views), 4):
            batch = torch.stack([processor(view) for view in views[start:start + 4]]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                features = model.encode_image(batch)
            visuals.append(normalize(features.float(), dim=-1).cpu())
        records.append({"label": row["label"], "start": offset, "candidates": len(row["boxes"])})
        offset += len(views)
        if number % 10 == 0 or number == len(cache["records"]):
            print(f"{name}: encoded {number}/{len(cache['records'])} training images", flush=True)
    torch.save({"signature": signature, "records": records, "visual": torch.cat(visuals)}, output)
    translations_path = ROOT / "features/translations.json"
    if translations_path.exists():
        translations = __import__("json").loads(translations_path.read_text(encoding="utf-8"))
        tokenizer = open_clip.get_tokenizer(model_name)
        encoded = []
        for start in range(0, len(cache["texts"]), 16):
            texts = [translations[t] for t in cache["texts"][start:start + 16]]
            with torch.autocast("cuda", dtype=torch.float16):
                encoded.append(normalize(model.encode_text(tokenizer(texts).cuda()).float(), dim=-1).cpu())
        torch.save({"texts": cache["texts"], "text": torch.cat(encoded),
                    "translation_sha256": digest(translations_path)}, ROOT / f"features/{name}_text.pt")
    data.audit(ROOT / f"results/{name}_data_access.json")
    data.close()
    del model
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def encode_text_only(name):
    import json
    model_name, path = SPECS[name]
    cache = load_training_cache()
    translations_path = ROOT / "features/translations.json"
    translations = json.loads(translations_path.read_text(encoding="utf-8"))
    model = open_clip.create_model(model_name, pretrained=None)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = state.get("state_dict", state.get("model", state))
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    model = model.cuda().eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    encoded = []
    for start in range(0, len(cache["texts"]), 16):
        texts = [translations[t] for t in cache["texts"][start:start + 16]]
        with torch.autocast("cuda", dtype=torch.float16):
            encoded.append(normalize(model.encode_text(tokenizer(texts).cuda()).float(), dim=-1).cpu())
    torch.save({"texts": cache["texts"], "text": torch.cat(encoded),
                "translation_sha256": digest(translations_path)}, ROOT / f"features/{name}_text.pt")
    print(f"{name}: encoded {len(cache['texts'])} translated training descriptions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(SPECS))
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.text_only:
        encode_text_only(args.model)
    else:
        encode(args.model)


if __name__ == "__main__":
    main()
