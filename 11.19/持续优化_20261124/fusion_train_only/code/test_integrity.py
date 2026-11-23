"""Regression checks for the training whitelist and prediction/GT separation."""
import copy
import unittest
from unittest.mock import Mock

import torch

from box_parser import parse
from common import TrainArchive, split_labels
from score_round import predict


class IntegrityTests(unittest.TestCase):
    def test_archive_denies_non_whitelisted_entries_before_reading(self):
        archive = object.__new__(TrainArchive)
        archive.allowed = {"approved/train/images/a.jpg"}
        archive.accesses = []
        archive.data = Mock()
        with self.assertRaises(PermissionError):
            archive.read("unapproved/val/images/b.jpg")
        archive.data.read.assert_not_called()

    def test_overlap_groups_do_not_cross_internal_split(self):
        split = split_labels()
        fit, dev = set(split["fit"]), set(split["dev"])
        self.assertFalse(fit & dev)
        for group in split["overlap_groups"]:
            self.assertTrue(set(group) <= fit or set(group) <= dev)

    def test_coordinate_parser_clamps_and_rejects_invalid_boxes(self):
        box = parse("[[100, 200, 900, 800]]", 200, 100, "1000")
        torch.testing.assert_close(box, torch.tensor([20., 20., 180., 80.]))
        self.assertIsNone(parse("Cannot locate an aircraft", 100, 100, "1000"))

    def test_truth_cannot_change_fusion_predictions(self):
        records = [{"label": name, "width": 100, "height": 100,
                    "boxes": torch.tensor([[0., 0., 20., 20.]]), "scores": torch.tensor([0.8]),
                    "truth": torch.tensor([[0., 0., 20., 20.]])} for name in ("a", "b")]
        cache = {"records": records, "texts": ["query"]}
        features = {name: (torch.tensor([[0.5, 0.3]]),
                            [torch.tensor([[0.4]]), torch.tensor([[0.2]])])
                    for name in ("RemoteCLIP", "GeoRSCLIP")}
        jobs = {"grounding": [{"label": name, "query": 0, "image": i} for i, name in enumerate(("a", "b"))]}
        retrieval = {f"0:{i}": {"yes_minus_no_logit": 0.1} for i in range(2)}
        grounding = {f"0:{i}": {"chosen": 0} for i in range(2)}
        geo = {f"0:{i}": {"answer": "[0,0,20,20]"} for i in range(2)}
        geo_scores = {name: {f"0:{i}": 0.3 for i in range(2)} for name in features}
        data = (cache, features, torch.tensor([[0.5, 0.3]]), jobs, retrieval, grounding, geo, geo_scores, "pixels")
        config = {"remote_fraction": 0.5, "region_fraction": 0.5, "detector": 0.1,
                  "qwen_retrieval": 0.03, "qwen_grounding": 0.05, "geoground_vote": 0.05}
        first = predict(data, ["a", "b"], config)
        modified = copy.deepcopy(data)
        for record in modified[0]["records"]:
            record["truth"][:] = 999
        self.assertEqual(first, predict(modified, ["a", "b"], config))


if __name__ == "__main__":
    unittest.main()
