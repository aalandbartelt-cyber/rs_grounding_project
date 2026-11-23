"""Check feature and input-job independence from annotation answers."""
import copy
import unittest
from unittest.mock import patch

import torch

from train_attributes import prepare
from extract_attributes import jobs
from inputs import BASE


class AttributeIntegrityTests(unittest.TestCase):
    def test_annotation_answers_do_not_change_predicted_features_or_jobs(self):
        features, retrieval = prepare()
        cache = copy.deepcopy(features.data.cache)
        for record in cache['records']:
            record['truth'][:] = -999
        candidates = torch.load(BASE/'training_v2/features/candidates.pt', weights_only=True)['regions']
        expected_jobs = jobs(features.data, candidates)
        with patch('data.load_training_cache', return_value=cache):
            altered, altered_retrieval = prepare()
        self.assertEqual(expected_jobs, jobs(altered.data, candidates))
        for key, row in features.rows.items():
            torch.testing.assert_close(row['complementary'], altered.rows[key]['complementary'])
        for key, value in retrieval.items():
            torch.testing.assert_close(value, altered_retrieval[key])


if __name__ == '__main__':
    unittest.main()
