"""Materialize a self-contained archive containing only whitelisted train entries."""
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from common import ROOT, LOCAL_ARCHIVE, LOCAL_MANIFEST, TrainArchive, digest, load_training_cache, write_json


def main():
    if LOCAL_ARCHIVE.exists() or LOCAL_MANIFEST.exists():
        raise ValueError("The isolated training archive already exists")
    data = TrainArchive()
    cache = load_training_cache()
    LOCAL_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(LOCAL_ARCHIVE, "w", compression=ZIP_DEFLATED) as output:
        for entry in sorted(data.allowed):
            output.writestr(entry, data.read(entry))
    cache_path = ROOT / "data/pretrained_train_features.pt"
    torch.save(cache, cache_path)
    write_json(LOCAL_MANIFEST, {"training_labels": data.labels, "entries": sorted(data.allowed),
                               "source_split": "original public train",
                               "archive_sha256": digest(LOCAL_ARCHIVE),
                               "pretrained_features_sha256": digest(cache_path),
                               "previously_fitted_adapters_copied": False,
                               "official_validation_or_test_entries_included": False})
    data.audit(ROOT / "results/training_data_isolation.json")
    data.close()
    print(f"Isolated {len(cache['records'])} original train images and their annotations", flush=True)


if __name__ == "__main__":
    main()
