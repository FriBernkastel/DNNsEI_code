import math
import unittest

import numpy as np

from experiments.figure1_ei_decomposition import (
    ACTS,
    Figure1Config,
    single_edge_continuous,
    single_edge_discrete,
    two_to_one_continuous,
    two_to_one_discrete,
)


class Figure1EiDecompositionTests(unittest.TestCase):
    def test_act_specs_match_paper_ranges(self):
        self.assertEqual(ACTS["sigmoid"]["x_range"], (0.0, 1.0))
        self.assertEqual(ACTS["sigmoid"]["y_range"], (0.0, 1.0))
        self.assertEqual(ACTS["tanh"]["x_range"], (-1.0, 1.0))
        self.assertEqual(ACTS["tanh"]["y_range"], (-1.0, 1.0))
        self.assertEqual(ACTS["relu"]["x_range"], (0.0, 1.0))
        self.assertEqual(ACTS["relu"]["y_range"], (0.0, 1.0))

    def test_default_config_matches_paper_figure1_setup(self):
        config = Figure1Config()

        self.assertEqual(config.bins_list, (8, 16, 32, 64))
        self.assertEqual(config.grid_size, 41)
        self.assertAlmostEqual(config.max_weight, 27.0)
        self.assertAlmostEqual(config.grid_max_weight, 9.0)

    def test_single_edge_discrete_zero_weight_has_zero_ei(self):
        rng = np.random.default_rng(0)
        rows = single_edge_discrete("sigmoid", weights=[0.0, 3.0], bins_list=[16], n_samples=10000, rng=rng)

        ei_at_zero = [row for row in rows if row["weight"] == 0.0][0]
        ei_at_three = [row for row in rows if row["weight"] == 3.0][0]
        self.assertAlmostEqual(ei_at_zero["ei_bits"], 0.0, places=6)
        self.assertGreater(ei_at_three["ei_bits"], 0.5)
        for row in rows:
            self.assertAlmostEqual(row["sensitivity_bits"], row["ei_bits"])
            self.assertEqual(row["degeneracy_bits"], 0.0)

    def test_single_edge_continuous_matches_analytic_sigmoid_derivative(self):
        rng = np.random.default_rng(0)
        sigma = 0.1
        weights = [2.0]

        rows = single_edge_continuous("sigmoid", weights, sigma, n_samples=200000, rng=rng)

        a = np.random.default_rng(1).uniform(0.0, 1.0, size=200000)
        row = rows[0]
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(row["sensitivity_bits"], row["ei_bits"])
        self.assertEqual(row["degeneracy_bits"], 0.0)
        expected_term1 = -0.5 * math.log(2.0 * math.pi * math.e * sigma * sigma) / math.log(2.0)
        self.assertAlmostEqual(row["term1_bits"], expected_term1, places=6)

    def test_two_to_one_discrete_zero_weights_have_zero_ei(self):
        rng = np.random.default_rng(0)
        grid = np.array([0.0, 2.0])

        data = two_to_one_discrete("sigmoid", grid, bins=8, n_samples=10000, rng=rng)

        self.assertAlmostEqual(data["ei"][0, 0], 0.0, places=6)
        self.assertAlmostEqual(data["sensitivity"][0, 0], 0.0, places=6)
        self.assertTrue(np.all(data["ei"] >= -1e-9))
        np.testing.assert_allclose(
            data["degeneracy"],
            data["sensitivity"] - data["ei"],
            atol=1e-9,
        )

    def test_two_to_one_discrete_degeneracy_vanishes_when_one_weight_is_zero(self):
        rng = np.random.default_rng(0)
        grid = np.array([0.0, 3.0])

        data = two_to_one_discrete("sigmoid", grid, bins=8, n_samples=20000, rng=rng)

        self.assertAlmostEqual(data["degeneracy"][1, 0], 0.0, delta=0.15)
        self.assertAlmostEqual(data["degeneracy"][0, 1], 0.0, delta=0.15)

    def test_two_to_one_continuous_shapes_and_degeneracy_identity(self):
        rng = np.random.default_rng(0)
        grid = np.array([-2.0, 0.0, 2.0])

        data = two_to_one_continuous("sigmoid", grid, sigma=0.1, n_samples=5000, rng=rng)

        for key in ("ei", "sensitivity", "degeneracy"):
            self.assertEqual(data[key].shape, (3, 3))
            self.assertTrue(np.all(np.isfinite(data[key])))
        np.testing.assert_allclose(
            data["degeneracy"],
            data["sensitivity"] - data["ei"],
            atol=1e-9,
        )

    def test_two_to_one_continuous_degeneracy_is_nonpositive(self):
        rng = np.random.default_rng(0)
        grid = np.array([-3.0, -1.0, 1.0, 3.0])

        data = two_to_one_continuous("sigmoid", grid, sigma=0.1, n_samples=5000, rng=rng)

        self.assertTrue(np.all(data["degeneracy"] <= 1e-9))


if __name__ == "__main__":
    unittest.main()
