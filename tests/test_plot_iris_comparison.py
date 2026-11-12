import csv
import tempfile
import unittest
from pathlib import Path

from experiments.plot_iris_comparison import latest_run_dir, load_history, plot_comparison


def fake_history(layers=("L1", "L2"), epochs=(1, 2)):
    rows = []
    for epoch in epochs:
        for idx, layer in enumerate(layers, start=1):
            rows.append(
                {
                    "epoch": epoch,
                    "layer": layer,
                    "ei_bits": float(epoch * idx),
                    "sensitivity_bits": float(epoch * idx + 1),
                    "degeneracy_bits": 1.0,
                    "total_ei_bits": 10.0,
                    "total_sensitivity_bits": 12.0,
                    "total_degeneracy_bits": 2.0,
                }
            )
    return rows


class PlotIrisComparisonTests(unittest.TestCase):
    def test_load_history_converts_numeric_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "layer_ei_history.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "layer", "ei_bits"])
                writer.writeheader()
                writer.writerow({"epoch": 1, "layer": "L1", "ei_bits": 0.5})

            rows = load_history(csv_path)

            self.assertEqual(rows[0]["layer"], "L1")
            self.assertIsInstance(rows[0]["epoch"], float)
            self.assertAlmostEqual(rows[0]["ei_bits"], 0.5)

    def test_latest_run_dir_picks_most_recent_prefix_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "iris_discrete_ei_20260101_000000").mkdir()
            Path(tmpdir, "iris_discrete_ei_20260201_000000").mkdir()
            Path(tmpdir, "iris_continuous_ei_20260301_000000").mkdir()

            latest = latest_run_dir(tmpdir, "iris_discrete_ei")

            self.assertEqual(latest.name, "iris_discrete_ei_20260201_000000")

    def test_plot_comparison_writes_figure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "comparison.png"

            plot_comparison(fake_history(), fake_history(), out_path)

            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
