"""IRIS 训练全程的第三条 EI 口径：transport-map (TM-EI) 逐轮估计。

产出与 `iris_continuous_ei` / `iris_discrete_ei` 同构的 EI-epoch 曲线，用于在
`experiment_figures_*.html` 的图 2 上做三路对照。

两个口径：
  * uniform：与连续版完全同分布 —— cause ~ U(0,1)^n_in，effect = sigmoid(Wc)+N(0,σ²)。
    可直接和 Jacobian 连续版、分箱离散版同图比较。
  * real   ：把每轮训练数据真实激活当作样本 —— I(层输入; 层输出)，只有采样型估计器
    （TM-EI）能算，连续解析式假设 cause 均匀、给不出这个量。
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_PKG = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_PKG))

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eicore import (
    GradientSigmaAccumulator,
    IrisSigmoidNet,
    ResidualSigmaAccumulator,
    compute_model_layer_ei,
    evaluate_mse,
    load_iris_tensors,
)
from eicore.plotting import apply_style, finalize_figure
from eicore.report import publish_latest
from experiments.transport_map_ei import estimate_transport_map_mutual_information

LN2 = np.log(2.0)
CONTINUOUS_CSV = Path("results/latest/iris_continuous_ei/layer_ei_history.csv")
DISCRETE_CSV = Path("results/latest/iris_discrete_ei/layer_ei_history.csv")

COLORS = {
    "continuous": "#4c78a8",
    "discrete": "#e45756",
    "tm_uniform": "#54a24b",
    "tm_real": "#b279a2",
}


@dataclass(frozen=True)
class Config:
    epochs: int = 4000
    batch_size: int = 10
    lr: float = 0.01
    seed: int = 42
    ei_samples: int = 2000
    tm_samples: int = 20000
    record_every: int = 10
    outdir: str = "results"


def tm_ei_bits(cause: np.ndarray, effect: np.ndarray) -> float:
    return estimate_transport_map_mutual_information(cause, effect)["mi"] / LN2


def measure_epoch(model, sigma, neuron_sigmas, pairs, config, epoch, rng_seed):
    """返回该 epoch 的三路 EI（单位 bit，已除 n_in）。

    ``pairs``: {layer_name: (inputs, outputs)}，是该 epoch 内**每次前向真正经过
    权重计算的那对 (输入, 输出)** 按 batch 收集后拼接起来的样本对。
    """
    device = next(model.parameters()).device
    rows = compute_model_layer_ei(
        model,
        sigma=sigma,
        n_samples=config.ei_samples,
        device=device,
        seed=config.seed + epoch,
        first_layer_mins=None,
        first_layer_widths=None,
        neuron_sigmas=neuron_sigmas,
    )

    out = []
    for idx, row in enumerate(rows):
        name = row["layer"]
        layer = model.layers[idx]
        weight = layer.weight.detach()
        n_in, n_out = int(row["n_in"]), int(row["n_out"])
        layer_sigma = float(row["sigma"])

        # --- uniform 口径：与连续版同分布 ---
        gen = torch.Generator(device="cpu").manual_seed(config.seed + epoch + idx)
        cause = torch.rand((config.tm_samples, n_in), generator=gen, dtype=torch.float64)
        effect = torch.sigmoid(cause @ weight.t().cpu())
        rng = np.random.default_rng(rng_seed + idx)
        noisy = effect.numpy() + layer_sigma * rng.normal(size=effect.shape)
        tm_uniform = tm_ei_bits(cause.numpy(), noisy) / n_in

        # --- real 口径：本 epoch 实际流经权重的 (输入, 输出) 样本对 ---
        real_in, real_out = pairs[name]
        real_noisy = real_out + layer_sigma * rng.normal(size=real_out.shape)
        tm_real = tm_ei_bits(real_in, real_noisy) / n_in
        # 不加人为噪声（只靠 jitter）的版本，作为噪声模型必要性的诊断
        tm_actual = tm_ei_bits(real_in, real_out) / n_in

        out.append(
            {
                "epoch": epoch,
                "layer": name,
                "n_in": n_in,
                "n_out": n_out,
                "sigma": layer_sigma,
                "n_pairs": int(real_in.shape[0]),
                "cont_ei_bits": float(row["ei_bits"]),
                "tm_ei_bits": float(tm_uniform),
                "tm_ei_real_bits": float(tm_real),
                "tm_ei_actual_bits": float(tm_actual),
                "ratio": float(tm_uniform / row["ei_bits"]) if abs(row["ei_bits"]) > 1e-12 else float("nan"),
            }
        )
    return out


def train_and_measure(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, x_test, y_test, _, _ = load_iris_tensors(config.seed)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    model = IrisSigmoidNet().to(device=device, dtype=torch.float64)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(x_train.cpu(), y_train.cpu()),
        batch_size=config.batch_size,
        shuffle=True,
        generator=gen,
    )

    layer_names = [f"L{i}" for i in range(1, len(model.layers) + 1)]
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        sigma_acc = GradientSigmaAccumulator(layer_names=layer_names)
        residual_acc = ResidualSigmaAccumulator()
        train_loss_sum = 0.0
        train_seen = 0
        # 本 epoch 每次前向真正流经权重的 (输入, 输出)，按 batch 收集
        pair_inputs = {name: [] for name in layer_names}
        pair_outputs = {name: [] for name in layer_names}
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            y_pred, acts = model.forward_with_activations(xb)
            loss = loss_fn(y_pred, yb)
            residual_acc.update(y_pred, yb)
            loss.backward()
            sigma_acc.update({f"L{i}": a.grad for i, a in enumerate(acts, start=1)})
            optimizer.step()
            layer_inputs = [xb] + acts[:-1]
            for name, layer_in, layer_out in zip(layer_names, layer_inputs, acts):
                pair_inputs[name].append(layer_in.detach().cpu().numpy().copy())
                pair_outputs[name].append(layer_out.detach().cpu().numpy().copy())
            train_loss_sum += float(loss.item()) * xb.shape[0]
            train_seen += xb.shape[0]

        if epoch == 1 or epoch % config.record_every == 0 or epoch == config.epochs:
            pairs = {
                name: (
                    np.concatenate(pair_inputs[name], axis=0),
                    np.concatenate(pair_outputs[name], axis=0),
                )
                for name in layer_names
            }
            rows = measure_epoch(
                model,
                sigma_acc.sigmas(),
                sigma_acc.per_neuron_sigmas(),
                pairs,
                config,
                epoch,
                rng_seed=1000 * epoch,
            )
            train_mse = train_loss_sum / train_seen
            test_mse = evaluate_mse(model, x_test, y_test, loss_fn)
            for row in rows:
                row["train_mse"] = train_mse
                row["test_mse"] = test_mse
                history.append(row)
            if epoch % 500 == 0 or epoch == 1:
                fits = ", ".join(
                    f"{r['layer']}={r['tm_ei_bits']:.2f}/{r['cont_ei_bits']:.2f}" for r in rows
                )
                print(f"  epoch {epoch:4d}/{config.epochs}  TM/cont bits: {fits}", flush=True)
    return history


def load_archived(path, value_col):
    """读取归档 CSV（同种子同一批权重）里的逐轮曲线，返回 {layer: [(epoch, value)]}。"""
    if not path.exists():
        return {}
    series = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            series.setdefault(row["layer"], []).append(
                (int(float(row["epoch"])), float(row[value_col]))
            )
    return {k: sorted(v) for k, v in series.items()}


def _by_layer(history):
    layers = sorted({r["layer"] for r in history})
    return layers, {
        name: sorted([r for r in history if r["layer"] == name], key=lambda r: r["epoch"])
        for name in layers
    }


def plot_routes(history, run_dir):
    """图 A：三个估计器，同一个 estimand（cause ~ U(0,1)^n_in）。"""
    apply_style()
    run_dir = Path(run_dir)
    layers, by_layer = _by_layer(history)
    disc = load_archived(DISCRETE_CSV, "ei_norm_bits")

    fig, axes = plt.subplots(1, len(layers), figsize=(15.5, 4.7), sharex=True, sharey=False)
    for ax, name in zip(axes, layers):
        rows = by_layer[name]
        epochs = [r["epoch"] for r in rows]
        ax.plot(epochs, [r["cont_ei_bits"] for r in rows], color=COLORS["continuous"], lw=2.0, label="continuous (Jacobian + σ)")
        if name in disc:
            de, dv = zip(*disc[name])
            ax.plot(de, dv, color=COLORS["discrete"], ls="--", lw=1.5, label="discrete (bins=32)")
        ax.plot(epochs, [r["tm_ei_bits"] for r in rows], color=COLORS["tm_uniform"], lw=2.6, label="TM-EI (Gaussian fit)")
        ax.axhline(0.0, color="gray", lw=0.6)
        ax.set_title(f"{name}  ({rows[0]['n_in']}→{rows[0]['n_out']})")
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("EI (bit / input dim)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Three estimators, one estimand:  I(C;E) with C ~ uniform,  E = sigmoid(WC) + N(0, σ²)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    path = run_dir / "tm_ei_routes_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_sample_source(history, run_dir):
    """图 B：同一个 TM-EI 估计器，只换样本来源 / 噪声处理。"""
    apply_style()
    run_dir = Path(run_dir)
    layers, by_layer = _by_layer(history)

    fig, axes = plt.subplots(1, len(layers), figsize=(15.5, 4.7), sharex=True, sharey=False)
    for ax, name in zip(axes, layers):
        rows = by_layer[name]
        epochs = [r["epoch"] for r in rows]
        ax.plot(epochs, [r["tm_ei_bits"] for r in rows], color=COLORS["tm_uniform"], lw=2.2, ls="--",
                label="uniform cause + N(0,σ²)  (Fig. A reference)")
        ax.plot(epochs, [r["tm_ei_real_bits"] for r in rows], color=COLORS["tm_real"], lw=2.6,
                label="actual (in, out) pairs + N(0,σ²)")
        ax.plot(epochs, [r["tm_ei_actual_bits"] for r in rows], color=COLORS["tm_real"], lw=1.5, ls=":",
                label="actual (in, out) pairs, no added noise")
        ax.axhline(0.0, color="gray", lw=0.6)
        ax.set_title(f"{name}  ({rows[0]['n_in']}→{rows[0]['n_out']},  n_pairs={rows[0]['n_pairs']})")
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("TM-EI (bit / input dim)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        "Same estimator (TM-EI): sample source and noise treatment",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    path = run_dir / "tm_ei_sample_source.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_all(history, run_dir):
    return {"routes": plot_routes(history, run_dir), "sample_source": plot_sample_source(history, run_dir)}


def write_outputs(history, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.outdir) / f"iris_transport_map_ei_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "tm_ei_history.csv"
    fields = ["epoch", "layer", "n_in", "n_out", "sigma", "n_pairs", "cont_ei_bits",
              "tm_ei_bits", "tm_ei_real_bits", "tm_ei_actual_bits", "ratio", "train_mse", "test_mse"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    fig_paths = plot_all(history, run_dir)
    latest = publish_latest(run_dir, config.outdir)
    return {"csv": csv_path, "routes_figure": fig_paths["routes"],
            "sample_source_figure": fig_paths["sample_source"], "latest": latest}


def parse_args():
    parser = argparse.ArgumentParser(description="TM-EI on the IRIS training trajectory (third EI route).")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--ei-samples", type=int, default=Config.ei_samples)
    parser.add_argument("--tm-samples", type=int, default=Config.tm_samples)
    parser.add_argument("--record-every", type=int, default=Config.record_every)
    parser.add_argument("--outdir", type=str, default=Config.outdir)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        ei_samples=args.ei_samples,
        tm_samples=args.tm_samples,
        record_every=args.record_every,
        outdir=args.outdir,
    )
    history = train_and_measure(config)
    outputs = write_outputs(history, config)
    for key, path in outputs.items():
        print(f"Wrote {key}: {path}")


if __name__ == "__main__":
    main()
