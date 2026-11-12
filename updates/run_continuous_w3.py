"""临时审核脚本：把注入区间宽度设为 w=3，重算 IRIS 连续版每层 EI 与 Overall EI。

不修改原程序。做法：
  1. 复用 experiments.iris_continuous_ei 的训练循环（monkeypatch 其
     compute_model_layer_ei，换成"采样 U(0,w) 且 input_width=w"的版本）；
  2. 重新训练 4000 epoch，每 10 epoch 记录一次；
  3. 画原始 EI（不是相对差值），并与 w=1 的已有结果叠图对照。

用法（工作区根目录）：
    python -m updates.run_continuous_w3 --w 3
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import experiments.iris_continuous_ei as ice  # noqa: E402
from eicore.continuous import continuous_layer_ei, continuous_layer_sensitivity  # noqa: E402

LAYER_COLORS = {"L1": "#1f77b4", "L2": "#ff7f0e", "L3": "#2ca02c"}
TOTAL_COLOR = "#222222"


def make_patched_ei(width):
    """返回一个与 compute_model_layer_ei 同签名、但注入区间为 [0, width) 的版本。"""

    def patched(model, sigma, n_samples, device, seed, first_layer_mins=None,
                first_layer_widths=None, neuron_sigmas=None):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        rows = []
        for idx, layer in enumerate(model.layers, start=1):
            n_in = layer.in_features
            samples = torch.rand(
                (n_samples, n_in), generator=generator, dtype=torch.float64, device=device
            ) * float(width)
            layer_name = f"L{idx}"
            layer_sigma = sigma[layer_name] if isinstance(sigma, dict) else sigma
            weight = layer.weight.detach()
            bias = layer.bias.detach() if layer.bias is not None else None
            metrics = continuous_layer_ei(
                weight, bias, samples, sigma=layer_sigma, input_width=float(width)
            )
            if neuron_sigmas is not None:
                layer_neuron_sigmas = (
                    neuron_sigmas[layer_name] if isinstance(neuron_sigmas, dict) else neuron_sigmas
                )
                sens = continuous_layer_sensitivity(
                    weight, bias, samples, neuron_sigmas=layer_neuron_sigmas,
                    input_width=float(width),
                )
                sens["degeneracy_bits"] = sens["sensitivity_bits"] - metrics["ei_bits"]
                metrics.update(sens)
            rows.append(
                {
                    "layer": layer_name,
                    "n_in": n_in,
                    "n_out": layer.out_features,
                    "input_width": float(width),
                    **metrics,
                }
            )
        return rows

    return patched


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def series(rows, key="ei_bits"):
    out = {}
    for row in rows:
        out.setdefault(row["layer"], {})[int(row["epoch"])] = float(row[key])
    return {layer: (sorted(v), [v[e] for e in sorted(v)]) for layer, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--ei-samples", type=int, default=2000)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir) if args.outdir else PACKAGE / "results" / "updates"
    if not outdir.is_absolute():
        outdir = PACKAGE / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    ice.compute_model_layer_ei = make_patched_ei(args.w)
    config = ice.Config(
        epochs=args.epochs, record_every=args.record_every, ei_samples=args.ei_samples
    )
    history = ice.train_and_measure(config)

    # 写 CSV（原始值）
    csv_path = outdir / f"ei_w{args.w:g}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "layer", "ei_bits", "sensitivity_bits", "degeneracy_bits",
                         "term1_bits", "log_pdet_mean_nats", "sigma", "overall_sigma",
                         "overall_ei_bits"])
        for row in history:
            writer.writerow([row["epoch"], row["layer"], f"{float(row['ei_bits']):.6f}",
                             f"{float(row['sensitivity_bits']):.6f}",
                             f"{float(row['degeneracy_bits']):.6f}",
                             f"{float(row['term1_bits']):.6f}",
                             f"{float(row['log_pdet_mean_nats']):.6f}",
                             f"{float(row['sigma']):.6f}",
                             f"{float(row['overall_sigma']):.6f}",
                             f"{float(row['overall_ei_bits']):.6f}"])

    layers_w = series(history)
    overall_w = {int(r["epoch"]): float(r["overall_ei_bits"]) for r in history}

    # w=1 的已有结果做对照
    base_run = sorted((PACKAGE / "results").glob("iris_continuous_ei_*"))[-1]
    layers_1 = series(load_csv(base_run / "layer_ei_history.csv"))
    overall_1 = {int(r["epoch"]): float(r["overall_ei_bits"])
                 for r in load_csv(base_run / "overall_ei_history.csv")}

    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 160, "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.8,
        "legend.fontsize": 9.5, "legend.framealpha": 0.9, "lines.linewidth": 2.0,
    })
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.9))

    ax = axes[0]
    for layer, (epochs, vals) in sorted(layers_w.items()):
        ax.plot(epochs, vals, color=LAYER_COLORS.get(layer), marker="o", markersize=3.5,
                markevery=max(1, len(vals) // 10), label=f"{layer} (w={args.w:g})")
    ax.plot(sorted(overall_w), [overall_w[e] for e in sorted(overall_w)],
            color=TOTAL_COLOR, linewidth=2.8, linestyle="--", label=f"Overall (w={args.w:g})")
    ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("EI (bits)")
    ax.set_title(f"Layer EI + Overall EI with w = {args.w:g}")
    ax.legend(loc="lower right")

    ax = axes[1]
    for layer, (epochs, vals) in sorted(layers_1.items()):
        ax.plot(epochs, vals, color=LAYER_COLORS.get(layer), linewidth=1.2, linestyle=":",
                label=f"{layer} (w=1)")
    ax.plot(sorted(overall_1), [overall_1[e] for e in sorted(overall_1)],
            color=TOTAL_COLOR, linewidth=1.2, linestyle=":", label="Overall (w=1)")
    for layer, (epochs, vals) in sorted(layers_w.items()):
        ax.plot(epochs, vals, color=LAYER_COLORS.get(layer), linewidth=2.0,
                label=f"{layer} (w={args.w:g})")
    ax.plot(sorted(overall_w), [overall_w[e] for e in sorted(overall_w)],
            color=TOTAL_COLOR, linewidth=2.6, linestyle="--", label=f"Overall (w={args.w:g})")
    ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("EI (bits)")
    ax.set_title("w = 1 (dotted) vs. the new run (solid)")
    ax.legend(loc="lower right", ncol=2, fontsize=8.5)

    fig.suptitle(
        f"IRIS continuous EI, injection range w = {args.w:g} (raw bits, not differenced)",
        fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = outdir / f"ei_decomposition_w{args.w:g}.png"
    fig.savefig(png)
    plt.close(fig)

    print(f"\nfigure: {png}")
    print(f"csv   : {csv_path}")
    print("\n首末值（原始 bits）：")
    print(f"{'series':10s} {'epoch 1':>10s} {'epoch 4000':>12s} {'变化':>9s}")
    for layer in sorted(layers_w):
        v = layers_w[layer][1]
        print(f"{layer:10s} {v[0]:>10.3f} {v[-1]:>12.3f} {v[-1] - v[0]:>+9.3f}")
    ov = [overall_w[e] for e in sorted(overall_w)]
    print(f"{'Overall':10s} {ov[0]:>10.3f} {ov[-1]:>12.3f} {ov[-1] - ov[0]:>+9.3f}")


if __name__ == "__main__":
    main()
