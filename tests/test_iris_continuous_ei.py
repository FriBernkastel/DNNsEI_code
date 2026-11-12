import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from eicore import (
    GradientSigmaAccumulator,
    IrisSigmoidNet,
    ResidualSigmaAccumulator,
    compute_model_layer_ei,
    continuous_layer_ei,
    continuous_layer_sensitivity,
    load_iris_tensors,
    minmax_normalize,
    sigmoid_layer_jacobian,
)
from experiments.iris_continuous_ei import (
    Config,
    plot_curves,
    summarize_overall_ei,
)


class IrisContinuousEiTests(unittest.TestCase):
    def test_minmax_normalize_maps_columns_to_unit_interval(self):
        x = np.array(
            [
                [1.0, 10.0],
                [3.0, 20.0],
                [5.0, 30.0],
            ],
            dtype=np.float64,
        )

        x_norm, mins, spans = minmax_normalize(x)

        np.testing.assert_allclose(x_norm.min(axis=0), [0.0, 0.0])
        np.testing.assert_allclose(x_norm.max(axis=0), [1.0, 1.0])
        np.testing.assert_allclose(mins, [1.0, 10.0])
        np.testing.assert_allclose(spans, [4.0, 20.0])

    def test_load_iris_tensors_uses_raw_input_features(self):
        x_train, y_train, x_test, y_test, feature_mins, feature_widths = load_iris_tensors(seed=42)

        self.assertGreater(float(torch.max(x_train).item()), 1.0)
        self.assertGreater(float(torch.max(x_test).item()), 1.0)
        self.assertEqual(x_train.shape[1], 4)
        self.assertEqual(y_train.shape[1], 3)
        self.assertEqual(feature_mins.shape[0], 4)
        self.assertEqual(feature_widths.shape[0], 4)

    def test_default_config_matches_paper_iris_training_setup(self):
        config = Config()

        self.assertEqual(config.epochs, 4000)
        self.assertEqual(config.batch_size, 10)
        self.assertAlmostEqual(config.lr, 0.01)

    def test_model_has_no_bias_and_paper_uniform_weight_initialization(self):
        torch.manual_seed(123)
        model = IrisSigmoidNet()

        for layer in model.layers:
            self.assertIsNone(layer.bias)
            limit = 1.0 / math.sqrt(layer.in_features)
            self.assertLessEqual(float(torch.max(layer.weight).item()), limit)
            self.assertGreaterEqual(float(torch.min(layer.weight).item()), -limit)

    def test_first_layer_ei_samples_from_unit_activation_range(self):
        torch.manual_seed(123)
        model = IrisSigmoidNet()
        rows = compute_model_layer_ei(
            model,
            sigma=0.5,
            n_samples=8,
            device=torch.device("cpu"),
            seed=7,
            first_layer_mins=torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=torch.float64),
            first_layer_widths=torch.tensor([20.0, 20.0, 20.0, 20.0], dtype=torch.float64),
        )

        l1 = rows[0]
        self.assertEqual(l1["input_width"], 1.0)

    def test_gradient_sigma_accumulates_std_per_layer_from_training_batches(self):
        acc = GradientSigmaAccumulator(layer_names=["L1", "L2"], eps=0.0)
        acc.update(
            {
                "L1": torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
                "L2": torch.tensor([[2.0, 4.0]]),
            }
        )
        acc.update({"L1": torch.tensor([[2.0, 4.0]])})

        expected_l1 = np.array([1.0, 3.0, 5.0, 7.0, 2.0, 4.0], dtype=np.float64).std(ddof=0)
        expected_l2 = np.array([2.0, 4.0], dtype=np.float64).std(ddof=0)
        sigmas = acc.sigmas()
        self.assertAlmostEqual(sigmas["L1"], expected_l1)
        self.assertAlmostEqual(sigmas["L2"], expected_l2)

    def test_residual_sigma_accumulates_output_residual_std_for_overall_ei(self):
        acc = ResidualSigmaAccumulator(eps=0.0)
        acc.update(torch.tensor([[1.0, 3.0], [5.0, 7.0]]), torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
        acc.update(torch.tensor([[2.0, 4.0]]), torch.tensor([[1.0, 2.0]]))

        residuals = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0], dtype=np.float64)
        self.assertAlmostEqual(acc.sigma(), residuals.std(ddof=0))

    def test_sigmoid_layer_jacobian_matches_diag_sigmoid_prime_times_weight(self):
        weight = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)
        bias = torch.tensor([0.25, -0.75], dtype=torch.float64)
        x = torch.tensor([[0.2, 0.4]], dtype=torch.float64)

        jac = sigmoid_layer_jacobian(weight, bias, x)

        z = x @ weight.t() + bias
        sp = torch.sigmoid(z) * (1.0 - torch.sigmoid(z))
        expected = sp[:, :, None] * weight[None, :, :]
        torch.testing.assert_close(jac, expected)

    def test_continuous_layer_ei_uses_scalar_residual_sigma_and_svd_volume(self):
        weight = torch.eye(2, dtype=torch.float64)
        bias = torch.zeros(2, dtype=torch.float64)
        samples = torch.zeros((1, 2), dtype=torch.float64)
        sigma = 0.5

        metrics = continuous_layer_ei(weight, bias, samples, sigma)

        sp = 0.25
        expected_term1 = -0.5 * 2 * math.log(2.0 * math.pi * math.e * sigma * sigma)
        expected_term2 = 2.0 * math.log(sp)
        expected_ei = ((expected_term1 + expected_term2) / 2.0) / math.log(2.0)
        self.assertAlmostEqual(metrics["term1_bits"], expected_term1 / math.log(2.0), places=6)
        self.assertAlmostEqual(metrics["term2_bits"], expected_term2 / math.log(2.0), places=6)
        self.assertAlmostEqual(metrics["ei_bits"], expected_ei, places=6)

    def test_continuous_layer_ei_uses_per_feature_input_widths(self):
        weight = torch.eye(2, dtype=torch.float64)
        bias = torch.zeros(2, dtype=torch.float64)
        samples = torch.zeros((1, 2), dtype=torch.float64)
        sigma = 0.5

        metrics = continuous_layer_ei(weight, bias, samples, sigma, input_width=np.array([2.0, 3.0]))

        expected_term1 = -0.5 * 2 * math.log(2.0 * math.pi * math.e * sigma * sigma)
        expected_term2 = math.log(2.0) + math.log(3.0) + 2.0 * math.log(0.25)
        expected_ei = ((expected_term1 + expected_term2) / 2.0) / math.log(2.0)
        self.assertAlmostEqual(metrics["term2_bits"], expected_term2 / math.log(2.0), places=6)
        self.assertAlmostEqual(metrics["ei_bits"], expected_ei, places=6)

    def test_summarize_overall_ei_sums_layer_ei_by_epoch(self):
        history = [
            {
                "epoch": 1,
                "layer": "L1",
                "ei_bits": 1.0,
                "overall_ei_bits": 10.0,
                "train_mse": 0.3,
                "test_mse": 0.4,
                "sigma": 0.5,
                "overall_sigma": 0.7,
            },
            {
                "epoch": 1,
                "layer": "L2",
                "ei_bits": 2.0,
                "overall_ei_bits": 10.0,
                "train_mse": 0.3,
                "test_mse": 0.4,
                "sigma": 0.5,
                "overall_sigma": 0.7,
            },
            {
                "epoch": 2,
                "layer": "L1",
                "ei_bits": 4.0,
                "overall_ei_bits": 20.0,
                "train_mse": 0.2,
                "test_mse": 0.25,
                "sigma": 0.3,
                "overall_sigma": 0.6,
            },
            {
                "epoch": 2,
                "layer": "L2",
                "ei_bits": 8.0,
                "overall_ei_bits": 20.0,
                "train_mse": 0.2,
                "test_mse": 0.25,
                "sigma": 0.3,
                "overall_sigma": 0.6,
            },
        ]

        rows = summarize_overall_ei(history)

        self.assertEqual([row["epoch"] for row in rows], [1, 2])
        self.assertEqual([row["overall_ei_bits"] for row in rows], [10.0, 20.0])
        self.assertEqual([row["overall_sigma"] for row in rows], [0.7, 0.6])

    def test_plot_curves_writes_loss_and_single_ei_figure(self):
        history = [
            {"epoch": 1, "layer": "L1", "ei_bits": 1.0, "train_mse": 0.3, "test_mse": 0.4, "sigma": 0.5},
            {"epoch": 1, "layer": "L2", "ei_bits": 2.0, "train_mse": 0.3, "test_mse": 0.4, "sigma": 0.5},
            {"epoch": 2, "layer": "L1", "ei_bits": 4.0, "train_mse": 0.2, "test_mse": 0.25, "sigma": 0.3},
            {"epoch": 2, "layer": "L2", "ei_bits": 8.0, "train_mse": 0.2, "test_mse": 0.25, "sigma": 0.3},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_curves(history, Path(tmpdir))

            self.assertTrue(paths["loss"].exists())
            self.assertTrue(paths["ei"].exists())
            self.assertGreater(paths["loss"].stat().st_size, 0)
            self.assertGreater(paths["ei"].stat().st_size, 0)
    def test_per_neuron_sigmas_accumulate_std_per_output_neuron(self):
        acc = GradientSigmaAccumulator(layer_names=["L1"], eps=0.0)
        acc.update({"L1": torch.tensor([[1.0, 3.0], [5.0, 7.0]])})
        acc.update({"L1": torch.tensor([[2.0, 4.0]])})

        neuron_sigmas = acc.per_neuron_sigmas()

        expected_col0 = np.array([1.0, 5.0, 2.0], dtype=np.float64).std(ddof=0)
        expected_col1 = np.array([3.0, 7.0, 4.0], dtype=np.float64).std(ddof=0)
        np.testing.assert_allclose(neuron_sigmas["L1"], [expected_col0, expected_col1])

    def test_continuous_layer_sensitivity_matches_edgewise_sum(self):
        weight = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)
        bias = torch.zeros(2, dtype=torch.float64)
        samples = torch.zeros((1, 2), dtype=torch.float64)
        neuron_sigmas = np.array([0.5, 0.25])

        metrics = continuous_layer_sensitivity(weight, bias, samples, neuron_sigmas)

        sp = 0.25
        expected_transmission = 4.0 * math.log(sp) + math.log(1.0 * 2.0 * 1.0 * 0.5)
        expected_term1 = -0.5 * (
            math.log(2.0 * math.pi * math.e * 0.5 * 0.5)
            + math.log(2.0 * math.pi * math.e * 0.25 * 0.25)
        )
        expected_sensitivity = ((expected_term1 + expected_transmission) / 2.0) / math.log(2.0)
        self.assertAlmostEqual(metrics["sens_term1_bits"], expected_term1 / math.log(2.0), places=6)
        self.assertAlmostEqual(metrics["sens_transmission_bits"], expected_transmission / math.log(2.0), places=6)
        self.assertAlmostEqual(metrics["sensitivity_bits"], expected_sensitivity, places=6)

    def test_compute_model_layer_ei_adds_sensitivity_and_degeneracy(self):
        torch.manual_seed(123)
        model = IrisSigmoidNet()
        neuron_sigmas = {
            "L1": np.full(5, 0.4),
            "L2": np.full(5, 0.3),
            "L3": np.full(3, 0.2),
        }

        rows = compute_model_layer_ei(
            model,
            sigma=0.5,
            n_samples=8,
            device=torch.device("cpu"),
            seed=7,
            first_layer_mins=torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=torch.float64),
            first_layer_widths=torch.tensor([20.0, 20.0, 20.0, 20.0], dtype=torch.float64),
            neuron_sigmas=neuron_sigmas,
        )

        for row in rows:
            self.assertIn("sensitivity_bits", row)
            self.assertIn("degeneracy_bits", row)
            self.assertAlmostEqual(row["degeneracy_bits"], row["sensitivity_bits"] - row["ei_bits"], places=9)

    def test_plot_curves_writes_decomposition_figure_when_sensitivity_present(self):
        history = [
            {"epoch": 1, "layer": "L1", "ei_bits": 1.0, "sensitivity_bits": 2.0, "degeneracy_bits": 1.0, "train_mse": 0.3, "test_mse": 0.4, "sigma": 0.5},
            {"epoch": 1, "layer": "L2", "ei_bits": 2.0, "sensitivity_bits": 3.0, "degeneracy_bits": 1.0, "train_mse": 0.3, "test_mse": 0.4, "sigma": 0.5},
            {"epoch": 2, "layer": "L1", "ei_bits": 4.0, "sensitivity_bits": 5.0, "degeneracy_bits": 1.0, "train_mse": 0.2, "test_mse": 0.25, "sigma": 0.3},
            {"epoch": 2, "layer": "L2", "ei_bits": 8.0, "sensitivity_bits": 9.0, "degeneracy_bits": 1.0, "train_mse": 0.2, "test_mse": 0.25, "sigma": 0.3},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_curves(history, Path(tmpdir))

            self.assertTrue(paths["decomposition"].exists())
            self.assertGreater(paths["decomposition"].stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
