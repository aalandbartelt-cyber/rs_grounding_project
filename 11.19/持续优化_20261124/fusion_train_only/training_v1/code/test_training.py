"""Checks for fold integrity, trained predictions and supervision boundaries."""
import copy
import unittest

import torch

from data import BASE,Data,decode_deltas,encode_deltas
from evaluate import predict
from train import supervision,retrieval_supervision


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=Data()

    def test_training_folds_preserve_scene_groups_and_exclude_internal_dev(self):
        d=self.data
        folds=d.folds()
        self.assertEqual(sorted(i for fold in folds for i in fold),sorted(d.fit_ids))
        for fold in folds:
            self.assertFalse(set(fold)&set(d.dev_ids))
        lookup={r["label"]:i for i,r in enumerate(d.records)}
        for group in d.split["overlap_groups"]:
            indices={lookup[label] for label in group}
            if indices<=set(d.fit_ids):
                self.assertEqual(sum(indices<=set(fold) for fold in folds),1)

    def test_training_rejects_internal_dev_supervision(self):
        with self.assertRaises(AssertionError):
            supervision(self.data,self.data.dev_ids,"scalar")

    def test_box_regression_round_trip(self):
        before=torch.tensor([[20.,20.,80.,80.]])
        target=torch.tensor([[22.,21.,78.,79.]])
        result=decode_deltas(before,encode_deltas(before,target),100,100)
        torch.testing.assert_close(target,result)

    def test_frozen_predictions_do_not_read_target_boxes(self):
        bundle=torch.load(BASE/"training_v1/weights/trained_fusion_final.pt",weights_only=True)
        ids=self.data.fit_ids[:2]
        first=predict(self.data,ids,bundle)
        modified=copy.deepcopy(self.data)
        for row in modified.records:
            row["truth"][:]=999
        second=predict(modified,ids,bundle)
        self.assertEqual(first,second)


if __name__=="__main__":
    unittest.main()
