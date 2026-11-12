import unittest

import numpy as np
import torch

from eicore import IrisSigmoidNet, compute_model_layer_ei_discrete, discrete_layer_ei
from experiments.iris_discrete_ei import DiscreteConfig


class IrisDiscreteEiTests(unittest.TestCase):
    def test_default_config_matches_paper_training_setup(self):
        config = DiscreteConfig()

        self.assertEqual(config.epochs, 4000)
        self.assertEqual(config.batch_size, 10)
        self.assertAlmostEqual(config.lr, 0.01)
        self.assertEqual(config.bins, 32)

    def test_identity_layer_has_zero_degeneracy(self):
        rng = np.random.default_rng(0)
        weight = np.eye(3)

        metrics = discrete_layer_ei(weight, n_samples=20000, bins=16, rng=rng)

        self.assertLess(metrics["h_e_given_c_bits"], 0.35)
        self.assertAlmostEqual(metrics["sensitivity_bits"], metrics["ei_bits"], delta=0.2)
        self.assertAlmostEqual(metrics["degeneracy_bits"], 0.0, delta=0.2)

    def test_zero_weight_layer_has_zero_ei_and_sensitivity(self):
        rng = np.random.default_rng(0)
        weight = np.zeros((2, 3))

        metrics = discrete_layer_ei(weight, n_samples=5000, bins=8, rng=rng)

        self.assertAlmostEqual(metrics["ei_bits"], 0.0, places=6)
        self.assertAlmostEqual(metrics["sensitivity_bits"], 0.0, places=6)
        self.assertAlmostEqual(metrics["degeneracy_bits"], 0.0, places=6)

    def test_normalization_divides_by_input_dim(self):
        rng = np.random.default_rng(0)
        weight = np.eye(2)

        metrics = discrete_layer_ei(weight, n_samples=5000, bins=8, rng=rng)

        self.assertAlmostEqual(metrics["ei_norm_bits"], metrics["ei_bits"] / 2.0)
        self.assertAlmostEqual(metrics["sensitivity_norm_bits"], metrics["sensitivity_bits"] / 2.0)
        self.assertAlmostEqual(metrics["degeneracy_norm_bits"], metrics["degeneracy_bits"] / 2.0)

    def test_compute_model_layer_ei_discrete_covers_all_layers(self):
        torch.manual_seed(0)
        model = IrisSigmoidNet()

        rows = compute_model_layer_ei_discrete(model, n_samples=2000, bins=8, seed=1)

        self.assertEqual([row["layer"] for row in rows], ["L1", "L2", "L3"])
        for row in rows:
            self.assertAlmostEqual(
                row["degeneracy_bits"],
                row["sensitivity_bits"] - row["ei_bits"],
                places=9,
            )


if __name__ == "__main__":
    unittest.main()
