"""Training/evaluation boundary and coordinate checks, without model dependencies."""
import unittest
from prepare_data import assert_disjoint, clip_training_box


class DataBoundaryTests(unittest.TestCase):
    def test_rejects_image_leakage(self):
        with self.assertRaises(ValueError):
            assert_disjoint({"train": ["train/a"], "dev": ["train/a"]})

    def test_rejects_duplicate_weighting_by_accident(self):
        with self.assertRaises(ValueError):
            assert_disjoint({"train": ["train/a", "train/a"], "dev": []})

    def test_real_boundary_outlier_preserves_positive_visible_box(self):
        original = [1010, 20, 14.3333333333333, 15]
        clipped = clip_training_box(original, 1024, 1024)
        self.assertEqual(clipped, [1010, 20, 14, 15])
        self.assertGreater(original[2], clipped[2])

    def test_wholly_invisible_training_box_rejected(self):
        with self.assertRaises(ValueError):
            clip_training_box([1025, 20, 5, 5], 1024, 1024)


if __name__ == "__main__":
    unittest.main()
