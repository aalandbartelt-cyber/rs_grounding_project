"""P0 底座自测：标签解析、IoU、评分器、schema 校验。"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from common.boxes import clip_xyxy, iou, xyxy_to_xywh, xywh_to_xyxy  # noqa: E402
from common.labels import ID_TO_CATEGORY, load_label  # noqa: E402
from score import build_eval_set, evaluate, total_score  # noqa: E402
from validate_pred import validate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data", "rs_grounding_train_v1")
SPLITS = os.path.join(ROOT, "data", "splits")


class BoxTests(unittest.TestCase):
    def test_iou(self):
        a = [0, 0, 10, 10]
        self.assertAlmostEqual(iou(a, a), 1.0)
        self.assertAlmostEqual(iou(a, [10, 0, 20, 10]), 0.0)
        self.assertAlmostEqual(iou(a, [5, 0, 15, 10]), 1 / 3)

    def test_conversion(self):
        self.assertEqual(xyxy_to_xywh([10, 20, 30, 50]), [10.0, 20.0, 20.0, 30.0])
        self.assertEqual(xywh_to_xyxy([10, 20, 20, 30]), [10.0, 20.0, 30.0, 50.0])

    def test_clip(self):
        self.assertEqual(clip_xyxy([-5, -5, 1100, 1100], 1024, 1024),
                         [0.0, 0.0, 1024.0, 1024.0])


class LabelTests(unittest.TestCase):
    def test_four_point_rectangle(self):
        label = load_label(os.path.join(DATA, "train", "labels_json", "1001.json"))
        self.assertEqual(label["image_id"], "1001")
        self.assertEqual(label["width"], 1024)
        self.assertEqual(len(label["objects"]), 1)
        obj = label["objects"][0]
        self.assertEqual(obj["label"], "舰船")
        self.assertEqual(obj["class_id"], 1)
        self.assertEqual(obj["category_id"], 2)
        self.assertEqual(obj["bbox_xyxy"], [989.0, 487.0, 1009.0, 534.0])

    def test_two_point_rectangle_via_minmax(self):
        from common.labels import bbox_xyxy
        shape = {"points": [[10, 20], [30, 50]]}
        self.assertEqual(bbox_xyxy(shape), [10.0, 20.0, 30.0, 50.0])
        shape = {"points": [[30, 50], [10, 20]]}
        self.assertEqual(bbox_xyxy(shape), [10.0, 20.0, 30.0, 50.0])


class ScoreTests(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(total_score(0.30, 0.60, 98 / 157, 61 / 157), 0.563376, places=6)

    def _perfect_pred(self, manifest_path):
        manifest, gq, rq = build_eval_set(manifest_path)
        gallery = [str(i) for i in manifest["images"]]
        rows = []
        for q in gq:
            rows.append({
                "query_id": q["query_id"], "type": "grounding",
                "image_id": q["image_id"], "category_id": ID_TO_CATEGORY[q["class_id"]],
                "score": 1.0, "bbox": xyxy_to_xywh(q["gt_bbox"]),
            })
        for q in rq:
            source = str(q["image_id"])
            others = [i for i in gallery if i != source][:5]
            rows.append({
                "query_id": q["query_id"], "type": "retrieval",
                "ranking": [source] + others,
            })
        return rows

    def test_perfect_prediction_scores_one(self):
        manifest_path = os.path.join(SPLITS, "official_val.json")
        rows = self._perfect_pred(manifest_path)
        with tempfile.TemporaryDirectory() as tmp:
            pred_path = os.path.join(tmp, "pred.json")
            with open(pred_path, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, ensure_ascii=False)
            result = evaluate(manifest_path, pred_path, rounds=20)
        self.assertAlmostEqual(result["grounding"]["strict"]["acc50"], 1.0)
        self.assertAlmostEqual(result["grounding"]["strict"]["acc75"], 1.0)
        self.assertAlmostEqual(result["retrieval"]["r1"], 1.0)
        self.assertAlmostEqual(result["retrieval"]["r5"], 1.0)
        self.assertAlmostEqual(result["total"]["strict"], 1.0)

    def test_empty_prediction_scores_zero(self):
        manifest_path = os.path.join(SPLITS, "official_val.json")
        with tempfile.TemporaryDirectory() as tmp:
            pred_path = os.path.join(tmp, "pred.json")
            with open(pred_path, "w", encoding="utf-8") as fh:
                fh.write("[]")
            result = evaluate(manifest_path, pred_path, rounds=10)
        self.assertAlmostEqual(result["total"]["strict"], 0.0)
        self.assertEqual(result["counts"]["pred_missing_grounding"],
                         result["counts"]["grounding"])


class ValidateTests(unittest.TestCase):
    def test_validator_accepts_perfect_and_rejects_bad_box(self):
        manifest_path = os.path.join(SPLITS, "official_val.json")
        rows = ScoreTests()._perfect_pred(manifest_path)
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.json")
            with open(good, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, ensure_ascii=False)
            errors, summary, _ = validate(manifest_path, good)
            self.assertEqual(errors, [])
            self.assertEqual(summary["missing"], 0)

            bad = json.loads(json.dumps(rows))
            for row in bad:
                if row["type"] == "grounding":
                    row["bbox"] = [10.0, 10.0, -5.0, 20.0]
                    break
            bad_path = os.path.join(tmp, "bad.json")
            with open(bad_path, "w", encoding="utf-8") as fh:
                json.dump(bad, fh, ensure_ascii=False)
            errors, _, _ = validate(manifest_path, bad_path)
            self.assertTrue(any("w/h" in e for e in errors))


if __name__ == "__main__":
    unittest.main(verbosity=2)
