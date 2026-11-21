"""Compare decoded-image fingerprints; candidates require visual review."""
import collections
import itertools
import json

from evalkit import ROOT, read, write


def main():
    rows = read(ROOT / "audit" / "image_fingerprints.json")
    manifest = read(ROOT / "splits" / "manifest.json")
    train = set(manifest["train_keys"])
    heldout = set(manifest["dev_keys"]) | set(manifest["official_val_keys"])
    if {r["key"] for r in rows} != train | heldout or len(rows) != len(train | heldout):
        raise ValueError("Image fingerprints do not match frozen manifest")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["width"], r["height"], r["pixel_sha256"])].append(r["key"])
    duplicates = [v for v in groups.values() if len(v) > 1]
    candidates = []
    for a, b in itertools.combinations(rows, 2):
        distance = (int(a["dhash64"], 16) ^ int(b["dhash64"], 16)).bit_count()
        if distance <= 5:
            candidates.append({"a": a["key"], "b": b["key"], "dhash_hamming_distance": distance,
                               "cross_train_heldout": (a["key"] in train) != (b["key"] in train)})
    candidates.sort(key=lambda p: (p["dhash_hamming_distance"], p["a"], p["b"]))
    cross_duplicates = [g for g in duplicates if set(g) & train and set(g) & heldout]
    result = {"decoded_images": len(rows), "decoded_pixel_duplicate_groups": duplicates,
              "cross_train_heldout_pixel_duplicate_groups": cross_duplicates,
              "dhash_threshold": 5, "near_duplicate_candidates": candidates,
              "cross_train_heldout_near_duplicate_candidates": [p for p in candidates if p["cross_train_heldout"]],
              "scope": "Decoded-pixel equality and 64-bit horizontal dHash screening only. dHash similarity is not proof of duplication; crops, rotations and spatial overlap can be missed."}
    write(ROOT / "audit" / "image_similarity_audit.json", result)
    print(json.dumps({"decoded_images": len(rows), "pixel_duplicate_groups": len(duplicates),
                      "near_duplicate_candidates": len(candidates),
                      "cross_train_heldout_near_duplicate_candidates": len(result["cross_train_heldout_near_duplicate_candidates"])}))
    if cross_duplicates:
        raise ValueError("Exact decoded-pixel leakage detected; revise split before training")


if __name__ == "__main__":
    main()
