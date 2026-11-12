import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eicore.plotting import apply_style, finalize_figure, plot_extra_series, plot_layer_metric
from eicore.report import publish_file


def load_history(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            converted = {}
            for key, value in row.items():
                if key == "layer":
                    converted[key] = value
                    continue
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
    return rows


def latest_run_dir(outdir, prefix):
    candidates = sorted(
        (path for path in Path(outdir).glob(f"{prefix}_*") if path.is_dir()),
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No run directory matching {prefix}_* under {outdir}")
    return candidates[-1]


def per_epoch_values(history, key):
    by_epoch = {}
    for row in history:
        by_epoch[int(row["epoch"])] = float(row[key])
    epochs = sorted(by_epoch)
    return epochs, [by_epoch[e] for e in epochs]


def sum_layers_by_epoch(history, key):
    totals = {}
    for row in history:
        epoch = int(row["epoch"])
        totals[epoch] = totals.get(epoch, 0.0) + float(row[key])
    epochs = sorted(totals)
    return epochs, [totals[e] for e in epochs]


def plot_comparison(continuous_history, discrete_history, out_path, continuous_overall=None):
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    columns = [
        ("ei_bits", "EI (bits)", "Layer EI"),
        ("sensitivity_bits", "Sensitivity (bits)", "Sensitivity"),
        ("degeneracy_bits", "Degeneracy (bits)", "Degeneracy"),
    ]
    for col, (key, ylabel, title) in enumerate(columns):
        ax_cont = axes[0][col]
        plot_layer_metric(ax_cont, continuous_history, key)
        if key == "ei_bits" and continuous_overall is not None:
            epochs, values = per_epoch_values(continuous_overall, "overall_ei_bits")
            plot_extra_series(ax_cont, epochs, values, label="Overall EI")
        else:
            epochs, values = sum_layers_by_epoch(continuous_history, key)
            plot_extra_series(ax_cont, epochs, values, label="Total")
        ax_cont.set_ylabel(ylabel)
        ax_cont.set_title(f"Continuous {title}")

        ax_disc = axes[1][col]
        plot_layer_metric(ax_disc, discrete_history, key)
        epochs, values = per_epoch_values(
            discrete_history,
            {"ei_bits": "total_ei_bits", "sensitivity_bits": "total_sensitivity_bits", "degeneracy_bits": "total_degeneracy_bits"}[key],
        )
        plot_extra_series(ax_disc, epochs, values, label="Total")
        ax_disc.set_ylabel(ylabel)
        ax_disc.set_title(f"Discrete {title}")

    finalize_figure(fig, out_path, suptitle="IRIS: Continuous vs Discrete EI Decomposition")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot continuous vs discrete IRIS EI decomposition comparison.")
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--continuous-dir", type=str, default=None)
    parser.add_argument("--discrete-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    continuous_dir = Path(args.continuous_dir) if args.continuous_dir else latest_run_dir(args.results, "iris_continuous_ei")
    discrete_dir = Path(args.discrete_dir) if args.discrete_dir else latest_run_dir(args.results, "iris_discrete_ei")
    out_path = Path(args.out) if args.out else Path(args.results) / "iris_continuous_vs_discrete.png"

    continuous_history = load_history(continuous_dir / "layer_ei_history.csv")
    discrete_history = load_history(discrete_dir / "layer_ei_history.csv")
    continuous_overall = None
    overall_csv = continuous_dir / "overall_ei_history.csv"
    if overall_csv.exists():
        continuous_overall = load_history(overall_csv)

    plot_comparison(continuous_history, discrete_history, out_path, continuous_overall=continuous_overall)
    latest_path = publish_file(out_path, args.results)
    print(f"Continuous run: {continuous_dir}")
    print(f"Discrete run: {discrete_dir}")
    print(f"Wrote {out_path}")
    print(f"Wrote {latest_path}")


if __name__ == "__main__":
    main()
