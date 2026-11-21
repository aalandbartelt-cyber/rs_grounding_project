"""A weak offline submission example that reads input_path only, never labels/GT."""
import argparse
import hashlib
import random
from pathlib import Path

from evalkit import read, write


def predict(input_path, mode):
    images = read(input_path / "info.json")
    queries = read(input_path / "queries.json")
    by_id = {r["image_id"]: r for r in images}
    ids = sorted(by_id)
    predictions = []
    for q in queries:
        # Identical text produces identical retrieval rankings, independent of query order/ID.
        seed = int.from_bytes(hashlib.sha256(q["text"].encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        p = {"query_id": q["query_id"], "type": q["type"]}
        if q["type"] == "retrieval":
            ranking = list(ids)
            if mode == "random":
                rng.shuffle(ranking)
            p["ranking"] = ranking
        else:
            image = by_id[q["image_id"]]
            w, h = image["width"], image["height"]
            bbox = [0, 0, w, h]
            category_id = 1
            if mode == "random":
                x, y = rng.uniform(0, w - 1), rng.uniform(0, h - 1)
                bbox = [x, y, rng.uniform(1, w - x), rng.uniform(1, h - y)]
                category_id = rng.randint(1, 4)
            p.update(image_id=q["image_id"], category_id=category_id, score=1.0, bbox=bbox)
        predictions.append(p)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--mode", choices=("random", "full_image"), default="full_image")
    args = parser.parse_args()
    input_path, output_path = args.input_path.resolve(), args.output_path.resolve()
    if output_path == input_path or input_path in output_path.parents:
        raise ValueError("Write predictions outside input_path")
    predictions = predict(input_path, args.mode)
    write(output_path / "pred.json", predictions)
    print(f"Wrote {len(predictions)} predictions to {output_path / 'pred.json'}")
