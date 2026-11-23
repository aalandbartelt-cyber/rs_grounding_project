"""Training data boundary checks for fold-specific visual adaptation."""
import unittest

from cv_lora import train


class Split:
    fit_ids = [0, 1]


class LoRACVBoundaryTests(unittest.TestCase):
    def test_dev_label_cannot_enter_adapter_training(self):
        config = {'rank': 8, 'blocks': 2, 'epochs': 3}
        with self.assertRaises(AssertionError):
            train(Split(), [0, 2], config, archive=None)


if __name__ == '__main__':
    unittest.main()
