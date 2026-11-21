"""Hand-calculated metric cases and hostile submission cases, stdlib unittest."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import evalkit as e


def fixture():
    gallery = [{"image_id": i, "file_name": f"{i}.jpg", "width": 100, "height": 100} for i in range(1, 9)]
    answers = []
    for qid in range(1, 8):
        image_id = qid if qid <= 3 else 1
        answers.append({"query_id": qid, "type": "retrieval" if qid <= 3 else "grounding",
                        "image_id": image_id, "category_id": 1, "bbox": [0, 0, 10, 10],
                        "source_key": f"test/{image_id}", "alternative_image_ids": [image_id],
                        "alternative_bboxes": [[0, 0, 10, 10]]})
    gt = {"protocol": "hand_calculated", "gallery": gallery, "answers": answers}
    predictions = [
        {"query_id": 1, "type": "retrieval", "ranking": [1, 2, 3, 4, 5]},
        {"query_id": 2, "type": "retrieval", "ranking": [3, 4, 5, 6, 2]},
        {"query_id": 3, "type": "retrieval", "ranking": [1, 2, 4, 5, 6, 7, 3]},
    ]
    for qid, box in ((4, [0, 0, 10, 10]), (5, [0, 0, 10, 5]), (6, [0, 0, 10, 7.5])):
        predictions.append({"query_id": qid, "type": "grounding", "image_id": 1,
                            "category_id": 1, "score": 0.9, "bbox": box})
    return gt, predictions


class MetricTests(unittest.TestCase):
    def test_hand_calculated_total(self):
        gt, pred = fixture()
        result = e.evaluate(gt, pred)
        for key, expected in {"R1": 1 / 3, "R5": 2 / 3, "Acc05": 3 / 4, "Acc75": 2 / 4, "total": 2 / 3}.items():
            self.assertAlmostEqual(result["primary"][key], expected)
        self.assertEqual(result["missing_queries"], 1)

    def test_empty_predictions_keep_denominators(self):
        gt, _ = fixture()
        result = e.evaluate(gt, [])
        self.assertEqual(result["primary"]["total"], 0)
        self.assertEqual(result["missing_queries"], 7)
        self.assertEqual(result["primary"]["grounding_queries"], 4)

    def test_iou_exact_thresholds(self):
        self.assertEqual(e.iou([0, 0, 10, 10], [0, 0, 10, 5]), 0.5)
        self.assertEqual(e.iou([0, 0, 10, 10], [0, 0, 10, 7.5]), 0.75)
        gt, pred = fixture()
        rows = e.evaluate(gt, pred)["details"]
        self.assertEqual((rows[4]["acc05"], rows[4]["acc75"]), (1, 0))
        self.assertEqual((rows[5]["acc05"], rows[5]["acc75"]), (1, 1))

    def test_iou_just_below_threshold(self):
        self.assertLess(e.iou([0, 0, 10, 10], [0, 0, 10, 4.999999]), 0.5)
        self.assertLess(e.iou([0, 0, 10, 10], [0, 0, 10, 7.499999]), 0.75)

    def test_continuous_coordinates_no_plus_one(self):
        self.assertEqual(e.iou([0, 0, 1, 1], [1, 0, 1, 1]), 0)
        self.assertEqual(e.iou([0, 0, 2, 2], [0, 0, 1, 1]), 0.25)

    def test_boxes_are_xywh(self):
        self.assertEqual(e.iou([10, 20, 2, 2], [11, 20, 2, 2]), 1 / 3)

    def test_no_clipping_or_rounding(self):
        self.assertEqual(e.iou([-0.25, 0.5, 1.5, 2.0], [-0.25, 0.5, 1.5, 2.0]), 1)
        box = e.bbox_from_points([[1024.333333333, 20], [1000, 10]])
        self.assertEqual([box[0], box[1], box[3]], [1000, 10, 10])
        self.assertAlmostEqual(box[2], 24.333333333, places=12)

    def test_oracle(self):
        gt, _ = fixture()
        self.assertEqual(e.evaluate(gt, e.baseline(gt, "oracle"))["primary"]["total"], 1)

    def test_zero(self):
        gt, _ = fixture()
        self.assertEqual(e.evaluate(gt, e.baseline(gt, "zero"))["primary"]["total"], 0)

    def test_score_does_not_rescale_iou(self):
        gt, _ = fixture()
        pred = e.baseline(gt, "oracle")
        for p in pred:
            if p["type"] == "grounding":
                p["score"] = 0
        self.assertEqual(e.evaluate(gt, pred)["primary"]["total"], 1)

    def test_category_policy_is_explicit(self):
        gt, _ = fixture()
        pred = e.baseline(gt, "oracle")
        for p in pred:
            if p["type"] == "grounding":
                p["category_id"] = 2
        result = e.evaluate(gt, pred)
        self.assertEqual(result["primary"]["total"], 1)
        self.assertAlmostEqual(result["category_gated_diagnostic"]["total"], 0.3)

    def test_multitarget_never_changes_primary(self):
        gt, pred = fixture()
        gt["answers"][0]["alternative_image_ids"] = [1, 8]
        pred[0]["ranking"] = [8, 2, 3, 4, 5]
        result = e.evaluate(gt, pred)
        self.assertEqual(result["primary"]["R1"], 0)
        self.assertAlmostEqual(result["multi_target_diagnostic"]["R1"], 1 / 3)
        self.assertIsNone(result["by_category"]["3"]["total"])

    def test_four_points_and_two_points_equivalent(self):
        self.assertEqual(e.bbox_from_points([[4, 8], [1, 8], [1, 2], [4, 2]]), [1, 2, 3, 6])
        self.assertEqual(e.bbox_from_points([[4, 8], [1, 2]]), [1, 2, 3, 6])

    def test_nonrectangular_polygon_is_rejected(self):
        with self.assertRaises(ValueError):
            e.bbox_from_points([[0, 0], [2, 0], [3, 1], [0, 1]])

    def test_seeded_random_is_reproducible(self):
        gt, _ = fixture()
        self.assertEqual(e.baseline(gt, "random"), e.baseline(gt, "random"))

    def test_cluster_bootstrap_perfect_scores(self):
        gt, _ = fixture()
        rows = e.evaluate(gt, e.baseline(gt, "oracle"))["details"]
        result = e.image_bootstrap(rows, samples=100)
        self.assertEqual(result["source_images"], 3)
        self.assertEqual(result["intervals"]["total"], [1.0, 1.0])
        self.assertEqual(result, e.image_bootstrap(rows, samples=100))

    def test_grounding_multitarget_diagnostic_only(self):
        gt, pred = fixture()
        gt["answers"][3]["alternative_bboxes"].append([50, 50, 10, 10])
        pred[3]["bbox"] = [50, 50, 10, 10]
        result = e.evaluate(gt, pred)
        self.assertEqual(result["details"][3]["iou"], 0)
        self.assertEqual(result["primary"]["Acc05"], 0.5)
        self.assertEqual(result["multi_target_diagnostic"]["Acc05"], 0.75)


class FormatTests(unittest.TestCase):
    def setUp(self):
        self.gt, _ = fixture()
        self.pred = e.baseline(self.gt, "oracle")

    def reject(self, pred):
        with self.assertRaises(ValueError):
            e.evaluate(self.gt, pred)

    def test_duplicate_query_id(self):
        self.reject(self.pred + [self.pred[0]])

    def test_unknown_query_id(self):
        self.pred[0]["query_id"] = 999
        self.reject(self.pred)

    def test_noninteger_query_id(self):
        for qid in (True, 1.0, "1", None):
            with self.subTest(qid=qid):
                pred = copy.deepcopy(self.pred)
                pred[0]["query_id"] = qid
                self.reject(pred)

    def test_wrong_type(self):
        self.pred[0]["type"] = "grounding"
        self.reject(self.pred)

    def test_short_ranking(self):
        self.pred[0]["ranking"] = [1, 2, 3, 4]
        self.reject(self.pred)

    def test_repeated_image_id(self):
        self.pred[0]["ranking"] = [1, 1, 2, 3, 4]
        self.reject(self.pred)

    def test_invalid_ranking_id(self):
        for value in (999, "1", True, 1.0, None):
            with self.subTest(value=value):
                pred = copy.deepcopy(self.pred)
                pred[0]["ranking"] = [value, 2, 3, 4, 5]
                self.reject(pred)

    def test_wrong_grounding_image(self):
        self.pred[3]["image_id"] = 2
        self.reject(self.pred)

    def test_invalid_category(self):
        for value in (0, 5, "1", True):
            with self.subTest(value=value):
                pred = copy.deepcopy(self.pred)
                pred[3]["category_id"] = value
                self.reject(pred)

    def test_invalid_score(self):
        for value in (-1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(value=value):
                pred = copy.deepcopy(self.pred)
                pred[3]["score"] = value
                self.reject(pred)

    def test_invalid_box(self):
        for box in ([0, 0, 0, 1], [0, 0, -1, 1], [0, 0, 1, -1], [0, 0, 1],
                    [0, 0, float("nan"), 1], [float("inf"), 0, 1, 1], [True, 0, 1, 1],
                    [0, 0, 1e308, 1e308], [1e308, 0, 1e308, 1], "0,0,1,1"):
            with self.subTest(box=box):
                pred = copy.deepcopy(self.pred)
                pred[3]["bbox"] = box
                self.reject(pred)

    def test_wrong_container(self):
        self.reject({"predictions": self.pred})
        self.reject([None])

    def test_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "pred.json"
            p.write_text('[{"query_id":1,"query_id":2}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                e.strict_json(p)

    def test_nonstandard_json_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "pred.json"
            p.write_text('[{"score":NaN}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                e.strict_json(p)

    def test_json_unicode_bom(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "pred.json"
            p.write_text(json.dumps(self.pred), encoding="utf-8-sig")
            self.assertEqual(e.strict_json(p), self.pred)


if __name__ == "__main__":
    unittest.main(verbosity=2)
