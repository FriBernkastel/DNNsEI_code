import math
import unittest

import numpy as np
import torch

from experiments.mnist_continuous_ei import (
    Config,
    downsample_images_to_5x5,
    filter_digits_zero_to_four,
    make_model,
)


class MnistContinuousEiTests(unittest.TestCase):
    def test_default_config_matches_paper_reduced_mnist_setup(self):
        config = Config()

        self.assertEqual(config.epochs, 500)
        self.assertEqual(config.batch_size, 50)
        self.assertAlmostEqual(config.lr, 0.01)
        self.assertEqual(config.ei_samples, 2000)

    def test_model_uses_25_6_6_5_no_bias_with_paper_init(self):
        torch.manual_seed(123)
        model = make_model()

        dims = [(25, 6), (6, 6), (6, 5)]
        for layer, (n_in, n_out) in zip(model.layers, dims):
            self.assertEqual(layer.in_features, n_in)
            self.assertEqual(layer.out_features, n_out)
            self.assertIsNone(layer.bias)
            limit = 1.0 / math.sqrt(n_in)
            self.assertLessEqual(float(torch.max(layer.weight).item()), limit)
            self.assertGreaterEqual(float(torch.min(layer.weight).item()), -limit)

    def test_filter_digits_zero_to_four_keeps_only_first_five_classes(self):
        images = np.arange(10 * 28 * 28, dtype=np.uint8).reshape(10, 28, 28)
        labels = np.arange(10, dtype=np.uint8)

        filtered_images, filtered_labels = filter_digits_zero_to_four(images, labels)

        self.assertEqual(filtered_images.shape[0], 5)
        np.testing.assert_array_equal(filtered_labels, np.arange(5, dtype=np.int64))

    def test_downsample_images_to_5x5_returns_unit_scaled_flattened_vectors(self):
        images = np.full((2, 28, 28), 255, dtype=np.uint8)

        downsampled = downsample_images_to_5x5(images)

        self.assertEqual(downsampled.shape, (2, 25))
        self.assertEqual(downsampled.dtype, torch.float64)
        torch.testing.assert_close(downsampled, torch.ones((2, 25), dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
