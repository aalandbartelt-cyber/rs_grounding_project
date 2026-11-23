"""Focused checks for signed Qwen image-pair aggregation and fit-only training."""
import unittest
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'training_v2/code'))
from train_gallery_ranker import pair_scores
from retrieval import fit


class GalleryPairTests(unittest.TestCase):
    def test_pair_direction_is_preserved_for_non_sorted_candidate_order(self):
        record = {'jobs': {'selections': {'cv:0': [2, 0, 1]}},
                  'preferences': {'0:0:1': {'symmetric_preference': 8.},
                                  '0:0:2': {'symmetric_preference': 5.},
                                  '0:1:2': {'symmetric_preference': 2.}}}
        values = pair_scores(None, record, 0, 'cv', 'wins')
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])

    def test_pairwise_fit_rejects_any_internal_dev_image(self):
        class Split:
            fit_ids = [0, 1]
            dev_ids = [2]
        with self.assertRaises(AssertionError):
            fit(Split(), {}, [0, 2], {'objective': 'pairwise', 'target': 'exact',
                                     'representation': 'raw', 'alpha': 1., 'family': 'extra'})

    def test_score_does_not_accept_target_image_mapping(self):
        self.assertNotIn('truths', pair_scores.__code__.co_names)
        self.assertNotIn('labels', pair_scores.__code__.co_names)


if __name__ == '__main__':
    unittest.main()
