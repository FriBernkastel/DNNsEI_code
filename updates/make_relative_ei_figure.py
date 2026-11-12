"""临时审核脚本：把图 2 左图（Layer EI + Overall EI）改成"相对初始化"版本。

不修改原程序，也不写入 results/latest（HTML 引用的位置）。只读取已有运行的 CSV，
输出到 results/updates/ 目录。

用法（项目根目录）：
    python -m updates.make_relative_ei_figure [--run-dir results/iris_continuous_ei_xxxx]
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAYER_COLORS = {"L1": "#1f77b4", "L2": "#ff7f0e", "L3": "#2ca02c"}
TOTAL_COLOR = "#222222"


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def per_epoch_layer(rows, key="ei_bits"):
    series = {}
    for row in rows:
        series.setdefault(row["layer"], {})[int(row["epoch"])] = float(row[key])
    return {layer: (sorted(v), [v[e] for e in sorted(v)]) for layer, v in series.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    package = Path(__file__).resolve().parents[1]           # D:\PythonProject\DNNsEI_code_20261029
    run_dir = Path(args.run_dir) if args.run_dir else sorted(
        (package / "results").glob("iris_continuous_ei_*")
    )[-1]
    outdir = Path(args.outdir) if args.outdir else package / "results" / "updates"
    if not outdir.is_absolute():
        outdir = package / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    layers = per_epoch_layer(load_csv(run_dir / "layer_ei_history.csv"))
    overall_rows = load_csv(run_dir / "overall_ei_history.csv")
    overall_epochs = [int(r["epoch"]) for r in overall_rows]
    overall_vals = [float(r["overall_ei_bits"]) for r in overall_rows]

    # 每条曲线各自的基线 = epoch 1 的取值
    base_layer = {layer: vals[0] for layer, (_, vals) in layers.items()}
    base_overall = overall_vals[0]

    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 160, "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.8,
        "legend.fontsize": 9.5, "legend.framealpha": 0.9, "lines.linewidth": 2.0,
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.9))

    ax = axes[0]
    for layer, (epochs, vals) in sorted(layers.items()):
        ax.plot(epochs, vals, color=LAYER_COLORS.get(layer), marker="o", markersize=3.5,
                markevery=max(1, len(vals) // 10), label=layer)
    ax.plot(overall_epochs, overall_vals, color=TOTAL_COLOR, linewidth=2.8,
            linestyle="--", label="Overall EI")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("EI (bits)")
    ax.set_title("Original: raw differential entropy")
    ax.legend(loc="best")

    ax = axes[1]
    for layer, (epochs, vals) in sorted(layers.items()):
        shifted = [v - base_layer[layer] for v in vals]
        ax.plot(epochs, shifted, color=LAYER_COLORS.get(layer), marker="o", markersize=3.5,
                markevery=max(1, len(shifted) // 10), label=layer)
    shifted_overall = [v - base_overall for v in overall_vals]
    ax.plot(overall_epochs, shifted_overall, color=TOTAL_COLOR, linewidth=2.8,
            linestyle="--", label="Overall EI (relative)")
    ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ΔEI (bits)")
    ax.set_title("Relative to epoch 1 (each series zeroed at t=1)")
    ax.legend(loc="best")

    ax.annotate(
        f"Overall: 0 → {shifted_overall[-1]:+.2f} bits\n(raw {base_overall:+.2f} → {overall_vals[-1]:+.2f})",
        xy=(overall_epochs[-1], shifted_overall[-1]),
        xytext=(0.35, 0.72), textcoords="axes fraction",
        fontsize=9.5, color=TOTAL_COLOR,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc"),
        arrowprops=dict(arrowstyle="->", color=TOTAL_COLOR, linewidth=1.0),
    )

    fig.suptitle(
        "IRIS continuous EI: raw vs. relative-to-initialization  "
        f"(run {run_dir.name})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    png = outdir / "ei_decomposition_relative.png"
    fig.savefig(png)
    plt.close(fig)

    csv_path = outdir / "ei_relative.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "series", "raw_bits", "relative_bits", "baseline_epoch1"])
        for layer, (epochs, vals) in sorted(layers.items()):
            for e, v in zip(epochs, vals):
                writer.writerow([e, layer, f"{v:.6f}", f"{v - base_layer[layer]:.6f}", f"{base_layer[layer]:.6f}"])
        for e, v in zip(overall_epochs, overall_vals):
            writer.writerow([e, "Overall", f"{v:.6f}", f"{v - base_overall:.6f}", f"{base_overall:.6f}"])

    print(f"run dir : {run_dir}")
    print(f"figure  : {png}")
    print(f"csv     : {csv_path}")
    print("\nbaseline (epoch 1):")
    for layer in sorted(base_layer):
        print(f"  {layer:8s} {base_layer[layer]:+8.3f} bits  ->  final {layers[layer][1][-1]:+8.3f}")
    print(f"  {'Overall':8s} {base_overall:+8.3f} bits  ->  final {overall_vals[-1]:+8.3f}  "
          f"(relative {shifted_overall[-1]:+.3f})")


if __name__ == "__main__":
    main()
