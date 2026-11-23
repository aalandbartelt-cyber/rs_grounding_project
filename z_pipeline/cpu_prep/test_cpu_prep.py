"""Meaningful geometry, leakage, interface and diagnostic checks without model libraries."""
import copy
from pathlib import Path
import tempfile
import unittest

from common import safe_child
from geometry import transform, geometry_is_joint_safe
from contract import ImageRecord, check_predictions, check_candidates
from diagnose import diagnose, grounding_error
from compare_runs import compare
from export_augmented import yolo_line, labelme_shape


def example():
    gallery=[{"image_id":1,"file_name":"a.jpg","width":100,"height":80}]
    answers=[{"query_id":1,"type":"retrieval","image_id":1,"category_id":3,"bbox":[10,20,20,10],
              "source_key":"train/a","shape_index":0,"alternative_image_ids":[1],"alternative_bboxes":[[10,20,20,10]]},
             {"query_id":2,"type":"grounding","image_id":1,"category_id":3,"bbox":[10,20,20,10],
              "source_key":"train/a","shape_index":0,"alternative_image_ids":[1],"alternative_bboxes":[[10,20,20,10]]}]
    gt={"schema_version":"2.0.0","protocol":"test","gallery":gallery,"answers":answers}
    queries=[{"query_id":1,"type":"retrieval","text":"左侧的白色车辆"},
             {"query_id":2,"type":"grounding","image_id":1,"text":"左侧的白色车辆"}]
    predictions=[{"query_id":1,"type":"retrieval","ranking":[1]},
                 {"query_id":2,"type":"grounding","image_id":1,"category_id":3,"score":0.9,"bbox":[10,20,20,10]}]
    candidates=[{"image_id":1,"candidates":[{"candidate_id":"hit","bbox":[10,20,20,10],"category_id":3,"score":0.9}]}]
    return gt,queries,predictions,candidates


class GeometryTests(unittest.TestCase):
    def test_yolo_export_has_hand_computed_normalized_box(self):
        line=yolo_line(3,[10,20,20,10],100,80).split()
        self.assertEqual(line[0],"2")
        self.assertEqual(list(map(float,line[1:])),[0.2,0.3125,0.2,0.125])

    def test_augmented_caption_is_withheld_but_original_preserved(self):
        shape=labelme_shape("车辆",3,[10,20,20,10],"左侧的车辆","flip_x",0)
        self.assertEqual(shape["description"],"")
        self.assertEqual(shape["original_description_for_review"],"左侧的车辆")
        self.assertTrue(shape["flags"]["text_review_required"])

    def test_original_caption_remains_usable(self):
        shape=labelme_shape("车辆",3,[10,20,20,10],"左侧的车辆","original",0)
        self.assertEqual(shape["description"],"左侧的车辆")

    def test_flip_has_hand_computed_coordinates(self):
        mapping,w,h=transform("flip_x",100,80)
        self.assertEqual(mapping.box([10,20,20,10]),[70,20,20,10])
        self.assertEqual((w,h),(100,80))

    def test_non_square_rotation_has_correct_frame(self):
        mapping,w,h=transform("rotate90_ccw",100,80)
        self.assertEqual(mapping.box([10,20,20,10]),[20,70,10,20])
        self.assertEqual((w,h),(80,100))

    def test_crop_roundtrip_preserves_float_coordinates(self):
        mapping,_,_=transform("crop_keep_all",100,80,[5,6,80,70])
        box=[10.125,20.25,20.5,10.75]
        self.assertEqual(mapping.inverse().box(mapping.box(box)),box)

    def test_geometry_never_auto_approves_chinese_caption(self):
        for operation in ("flip_x","rotate90_ccw","crop_keep_all"):
            self.assertFalse(geometry_is_joint_safe(operation))

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                safe_child(Path(folder),"../private/gt.json")


class ContractTests(unittest.TestCase):
    def test_complete_predictions_without_gt(self):
        _,queries,predictions,_=example()
        check_predictions([ImageRecord(1,Path("unused"),100,80)],queries,predictions)

    def test_missing_query_is_rejected(self):
        _,queries,predictions,_=example()
        with self.assertRaises(ValueError):
            check_predictions([ImageRecord(1,Path("unused"),100,80)],queries,predictions[:1])

    def test_boolean_id_and_nan_rejected(self):
        for key,value in (("image_id",True),("score",float("nan")),("bbox",[0,0,-1,2])):
            _,queries,predictions,_=example()
            predictions[1][key]=value
            with self.assertRaises(ValueError):
                check_predictions([ImageRecord(1,Path("unused"),100,80)],queries,predictions)

    def test_duplicate_candidate_ids_rejected(self):
        *_,candidates=example()
        candidates[0]["candidates"]*=2
        with self.assertRaises(ValueError):
            check_candidates(candidates,{1})

    def test_missing_candidate_image_rejected(self):
        with self.assertRaises(ValueError):
            check_candidates([],{1})


class DiagnosticTests(unittest.TestCase):
    def test_candidate_miss_differs_from_selection_failure(self):
        self.assertEqual(grounding_error({},0.0,[0.0]),"candidate_coverage_miss")
        self.assertEqual(grounding_error({},0.0,[1.0]),"selection_or_refinement_failure")
        self.assertEqual(grounding_error({},0.0,None),"failure_candidate_pool_unavailable")

    def test_exact_iou_threshold_is_success(self):
        self.assertEqual(grounding_error({},0.5,[0.5]),"correct_at_05")

    def test_oracle_primary_remains_one(self):
        gt,queries,predictions,candidates=example()
        result=diagnose(gt,predictions,queries,candidates)
        self.assertEqual(result["primary"]["total"],1.0)
        self.assertEqual(result["diagnostics"]["candidate_coverage"]["05"]["rate"],1.0)

    def test_missing_predictions_keep_denominator(self):
        gt,queries,predictions,candidates=example()
        result=diagnose(gt,predictions[:1],queries,candidates)
        self.assertEqual(result["missing_queries"],1)
        self.assertAlmostEqual(result["primary"]["total"],0.3)

    def test_wrong_category_does_not_change_primary_policy(self):
        gt,queries,predictions,candidates=example()
        predictions[1]["category_id"]=1
        result=diagnose(gt,predictions,queries,candidates)
        self.assertEqual(result["primary"]["total"],1.0)
        self.assertAlmostEqual(result["category_gated_diagnostic"]["total"],0.3)

    def test_refined_box_can_succeed_outside_pool(self):
        gt,queries,predictions,candidates=example()
        candidates[0]["candidates"][0]["bbox"]=[80,70,10,5]
        result=diagnose(gt,predictions,queries,candidates)
        self.assertEqual(result["primary"]["Acc05"],1.0)
        self.assertEqual(result["diagnostics"]["candidate_coverage"]["05"]["rate"],0.0)
        self.assertFalse(result["diagnostics"]["query_details"][1]["final_box_in_candidate_pool"])

    def test_query_protocol_mismatch_is_rejected(self):
        gt,queries,predictions,candidates=example()
        queries[1]["query_id"]=999
        with self.assertRaises(ValueError):
            diagnose(gt,predictions,queries,candidates)


class PairedComparisonTests(unittest.TestCase):
    def reports(self):
        gt,queries,predictions,candidates=example()
        report=diagnose(gt,predictions,queries,candidates)
        report["provenance"]={"gt_sha256":"gt","evaluator_sha256":"eval","input_sha256":"images"}
        return report,copy.deepcopy(report)

    def test_same_run_is_exactly_zero(self):
        a,b=self.reports()
        result=compare(a,b,samples=100)
        self.assertEqual(result["point_difference"]["total"],0)
        self.assertEqual(result["paired_image_bootstrap_95pct"]["total"],[0,0])

    def test_changed_input_requires_explicit_stress_contrast(self):
        a,b=self.reports()
        b["provenance"]["input_sha256"]="blurred"
        with self.assertRaises(ValueError):
            compare(a,b,samples=10)
        self.assertEqual(compare(a,b,samples=10,allow_input_shift=True)["comparison_kind"],"input_shift_contrast")

    def test_different_gt_rejected_even_for_stress(self):
        a,b=self.reports()
        b["provenance"]["gt_sha256"]="different"
        with self.assertRaises(ValueError):
            compare(a,b,samples=10,allow_input_shift=True)


if __name__=="__main__":
    unittest.main()
