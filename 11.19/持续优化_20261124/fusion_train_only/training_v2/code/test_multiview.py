"""Verify transformed coordinates and training/target separation."""
import copy
import unittest
from unittest.mock import patch

import torch
from PIL import Image,ImageDraw

from inputs import inverse_box,Data
from fusion_features import Features
from train_round import targets
from encode_prompts import prompts


class MultiViewTests(unittest.TestCase):
    def test_rotated_non_square_image_coordinates_map_back_exactly(self):
        original=Image.new('L',(32,24),0)
        ImageDraw.Draw(original).rectangle((4,5,11,13),fill=255)
        truth=torch.tensor([4.,5.,12.,14.])
        for angle in (0,90,180,270):
            observed=torch.tensor(original.rotate(angle,expand=True).getbbox())
            torch.testing.assert_close(inverse_box(observed,32,24,angle),truth)

    def test_predicted_features_do_not_depend_on_ground_truth_boxes(self):
        features=Features()
        cache=copy.deepcopy(features.data.cache)
        for row in cache['records']:row['truth'][:]=999
        with patch('data.load_training_cache',return_value=cache):
            modified=Features()
        for key,row in features.rows.items():
            torch.testing.assert_close(row['features'],modified.rows[key]['features'])
            torch.testing.assert_close(row['boxes'],modified.rows[key]['boxes'])

    def test_training_target_interface_denies_internal_dev(self):
        features=Features()
        with self.assertRaises(AssertionError):
            targets(features,features.data.dev_ids,'utility')

    def test_query_normalization_does_not_invent_an_engine_count(self):
        unknown=prompts('民航客机，具有双翼布局和翼下发动机。','An airliner with biplane configuration.')
        self.assertIsNone(unknown['engine_count_explicit_in_query'])
        self.assertNotIn('two engines',unknown['concise'])
        self.assertNotIn('biplane',unknown['literal'])
        explicit=prompts('民航客机，四台发动机。','An airliner with four engines.')
        self.assertEqual(explicit['engine_count_explicit_in_query'],4)


if __name__=='__main__':unittest.main()
