import argparse
import io
import json
import os
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from prepare_check import WORKSPACE, digest, experiment_splits, read_reviewed_archive

MODEL_ID = "OFA-Sys/chinese-clip-vit-base-patch16"
MODEL_REVISION = "36e679e65c2a2fead755ae21162091293ad37834"
MODEL_WEIGHTS_SHA256 = "29cc0b2bcf6ff777f2e15742be92b110e4acbdb2068356e862c4637a4b15fe4f"


class VisualProjectionAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, pooled, initial):
        return F.normalize(initial + self.up(self.down(pooled)), dim=-1)


def encode_split(archive, label_files, model, processor, device):
    captions_by_image = []
    with zipfile.ZipFile(archive) as data:
        pooled, initial = [], []
        for start in range(0, len(label_files), 8):
            images = []
            for label in label_files[start:start + 8]:
                meta = json.loads(data.read(label))
                folder = label.split("/labels_json/")[0]
                with Image.open(io.BytesIO(data.read(f"{folder}/images/{meta['imagePath']}"))) as image:
                    images.append(image.convert("RGB"))
                captions_by_image.append([shape["description"].strip() for shape in meta["shapes"]])
            values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            vision = model.vision_model(pixel_values=values, return_dict=True).pooler_output
            pooled.append(vision)
            initial.append(model.visual_projection(vision))

    captions = [caption for group in captions_by_image for caption in group]
    unique = list(dict.fromkeys(captions))
    text_index = {caption: index for index, caption in enumerate(unique)}
    query_images = [(image_index, text_index[caption]) for image_index, group in enumerate(captions_by_image)
                    for caption in group]
    encodings = []
    for start in range(0, len(unique), 16):
        tokens = processor(
            text=unique[start:start + 16], padding=True, truncation=True, max_length=128, return_tensors="pt",
        ).to(device)
        encodings.append(F.normalize(model.get_text_features(**tokens), dim=-1))
    return torch.cat(pooled), torch.cat(initial), torch.cat(encodings), query_images


def train_adapter(pooled, initial, text, query_images, *, rank, epochs, lr, drift_weight):
    adapter = VisualProjectionAdapter(pooled.shape[1], text.shape[1], rank).to(pooled.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    positive = torch.zeros((len(pooled), len(text)), device=pooled.device)
    for image_index, text_index in query_images:
        positive[image_index, text_index] = 1
    by_image = positive / positive.sum(dim=1, keepdim=True)
    by_caption = positive / positive.sum(dim=0, keepdim=True)
    base = F.normalize(initial, dim=-1)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        tuned = adapter(pooled, initial)
        logits = (tuned @ text.T) * 20.0
        image_loss = -(logits.log_softmax(dim=1) * by_image).sum(dim=1).mean()
        caption_loss = -(logits.log_softmax(dim=0) * by_caption).sum(dim=0).mean()
        drift = (1.0 - (tuned * base).sum(dim=1)).mean()
        loss = 0.5 * (image_loss + caption_loss) + drift_weight * drift
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs} loss={losses[-1]:.4f}", flush=True)
    return adapter.eval(), losses


def measure_retrieval(visual, captions, query_images):
    scores = captions @ visual.T
    ranking = scores.argsort(dim=1, descending=True, stable=True)
    img_ids, cap_ids = zip(*query_images)
    matched = ranking[list(cap_ids)] == torch.tensor(img_ids, device=ranking.device)[:, None]
    ranks = matched.to(torch.int64).argmax(dim=1) + 1
    return {
        "R@1": (ranks == 1).sum().item() / len(ranks),
        "R@5": (ranks <= 5).sum().item() / len(ranks),
        "MRR": ranks.float().reciprocal().mean().item(),
        "hits_at_1": (ranks == 1).sum().item(),
        "hits_at_5": (ranks <= 5).sum().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark a visual projection adapter on the reviewed civilian split.")
    parser.add_argument("--archive", type=Path, default=WORKSPACE / "data" / "rs_civilian_reviewed.zip")
    parser.add_argument("--weights", type=Path, default=WORKSPACE / "weights" / "chinese-clip-vit-base-patch16")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "experiments" / "chinese-clip-retrieval")
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--drift-weight', type=float, default=0.25)
    parser.add_argument('--dev-manifest', type=Path)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    if args.rank < 1 or args.epochs < 1 or not 0 < args.lr < float('inf'):
        parser.error('rank, epochs and lr must be positive and finite')
    if not 0 <= args.drift_weight < float('inf'):
        parser.error('drift-weight must be non-negative and finite')

    manifest = read_reviewed_archive(args.archive)
    splits, split_name = experiment_splits(manifest, args.dev_manifest)
    if digest(args.weights / "model.safetensors") != MODEL_WEIGHTS_SHA256:
        raise ValueError("The safetensors checkpoint must match the packaged Chinese-CLIP weights.")
    torch.manual_seed(32)
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(requested_device)
    start = time.perf_counter()
    model = ChineseCLIPModel.from_pretrained(args.weights, local_files_only=True, use_safetensors=True).eval().requires_grad_(False).to(device)
    processor = ChineseCLIPProcessor.from_pretrained(args.weights, local_files_only=True)
    print(f"Loaded Chinese-CLIP on {device}.", flush=True)
    with torch.no_grad():
        train = encode_split(args.archive, splits["train"], model, processor, device)
        print(f"Encoded {len(train[0])} train images and {len(train[3])} annotated text queries.", flush=True)
        val = encode_split(args.archive, splits["val"], model, processor, device)
        print(f"Encoded {len(val[0])} validation images and {len(val[3])} annotated text queries.", flush=True)

    adapter, losses = train_adapter(*train, rank=args.rank, epochs=args.epochs,
                                    lr=args.lr, drift_weight=args.drift_weight)
    with torch.no_grad():
        pooled, initial, captions, query_images = val
        base = measure_retrieval(F.normalize(initial, dim=-1), captions, query_images)
        adapted = measure_retrieval(adapter(pooled, initial), captions, query_images)

    result = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": str(device),
        "data_split": split_name,
        "rank": args.rank, "lr": args.lr, "drift_weight": args.drift_weight, "seed": 32,
        "train_images": len(train[0]), "train_queries": len(train[3]),
        "validation_images": len(val[0]), "validation_queries": len(val[3]),
        "validation_unique_texts": len(val[2]),
        "trainable_parameters": sum(param.numel() for param in adapter.parameters()),
        "epochs": len(losses),
        "first_loss": losses[0], "last_loss": losses[-1],
        "zero_shot": base, "adapted": adapted,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save_file({key: tensor.cpu() for key, tensor in adapter.state_dict().items()}, args.output / "adapter.safetensors")
    report = json.dumps(result, indent=2)
    (args.output / "results.json").write_text(report + "\n", encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
