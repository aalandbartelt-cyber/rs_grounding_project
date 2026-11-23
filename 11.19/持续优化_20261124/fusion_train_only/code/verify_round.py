"""Verify weights, training-only access records and the completed score arithmetic."""
import json
from datetime import datetime, timezone
from zipfile import ZipFile

from common import ROOT, digest, split_labels, training_labels, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    manifest = read(ROOT / "weights/download_manifest.json")
    assert manifest["all_verified"] and not manifest["dataset_files_downloaded"]
    verified = []
    for row in manifest["files"]:
        path = ROOT / "weights" / row["folder"] / row["file"]
        assert path.stat().st_size == row["bytes"], path
        actual = digest(path)
        assert actual == row["sha256"], path
        if row["expected_sha256"]:
            assert actual == row["expected_sha256"], path
        verified.append(str(path.relative_to(ROOT)))
    dino = read(ROOT / "weights/GroundingDINO_provenance.json")
    for row in dino["files"]:
        path = ROOT / "weights/GroundingDINO" / row["file"]
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], path
        verified.append(str(path.relative_to(ROOT)))

    labels = set(training_labels())
    split = split_labels()
    assert not set(split["fit"]) & set(split["dev"])
    assert set(split["fit"]) | set(split["dev"]) == labels
    for group in split["overlap_groups"]:
        assert set(group) <= set(split["fit"]) or set(group) <= set(split["dev"])
    with ZipFile(ROOT / "data/original_train_only.zip") as data:
        entries = data.namelist()
    assert len(entries) == 98
    assert all("/train/images/" in name or "/train/labels_json/" in name for name in entries)
    access_logs = sorted((ROOT / "results").glob("*data_access.json"))
    for path in access_logs:
        log = read(path)
        assert log["all_accesses_within_training_whitelist"], path
        assert not log["holdout_or_test_entries_accessed"], path
        assert set(log["annotation_files"]) == labels, path
        assert set(log["accessed_entries"]) <= set(entries), path

    frozen = read(ROOT / "results/frozen_config.json")
    report = read(ROOT / "results/score_round.json")
    assert not frozen["dev_scores_used_for_selection"]
    assert frozen["split_sha256"] == digest(ROOT / "results/split.json")
    assert report["frozen_config_sha256"] == digest(ROOT / "results/frozen_config.json")
    assert frozen["frozen_at_utc"] < report["scored_at_utc"]
    for name, expected in frozen["code_sha256"].items():
        assert digest(ROOT / "code" / name) == expected, name
    for name, row in report["results"].items():
        n, counts = row["queries_per_task"], row["counts"]
        for metric in ("R@1", "R@5", "Acc@0.5", "Acc@0.75"):
            assert 0 <= counts[metric] <= n
            assert abs(row[metric] - counts[metric] / n) < 1e-12
        score = (counts["R@1"] + 2 * counts["R@5"] + 6 * counts["Acc@0.5"] + counts["Acc@0.75"]) / (10 * n)
        assert abs(score - row["score_out_of_1"]) < 1e-12, name
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "verified_weight_files": verified, "archive_training_entries": len(entries),
              "data_access_logs_checked": [p.name for p in access_logs],
              "group_split_disjoint": True, "all_weight_hashes_match": True,
              "score_arithmetic_matches": True, "frozen_before_dev_scoring": True,
              "code_matches_frozen_checksums": True, "official_validation_or_test_used": False}
    write_json(ROOT / "results/final_verification.json", result)
    print(f"Verified {len(verified)} official model files, {len(access_logs)} train-only access logs and all scores.", flush=True)


if __name__ == "__main__":
    main()
