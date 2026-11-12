"""IRIS：连续 EI 的"噪声下限"修正 vs 原公式，逐轮对比 + Overall EI 对比。

修正规则（见 eicore/continuous.py::continuous_layer_ei）：
  每个奇异方向的净贡献 = ln s_k − h_noise；当 ln s_k < h_noise（该方向信号
  弱于噪声底）时截断为 0，不再扣该方向的噪声；噪声只对 min(n_out, n_in) 个
  真实方向计成本。
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
from eicore.plotting import apply_style
from eicore.report import publish_latest

OLD_COLOR = "#4c78a8"
NEW_COLOR = "#54a24b"


@dataclass(frozen=True)
class Config:
    epochs: int = 4000
    batch_size: int = 10
    lr: float = 0.01
    seed: int = 42
    ei_samples: int = 2000
    record_every: int = 10
    outdir: str = "results"


def train_and_compare(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, x_test, y_test, feature_mins, feature_widths = load_iris_tensors(config.seed)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    feature_mins = feature_mins.to(device)
    feature_widths = feature_widths.to(device)

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
            train_loss_sum += float(loss.item()) * xb.shape[0]
            train_seen += xb.shape[0]

        if epoch == 1 or epoch % config.record_every == 0 or epoch == config.epochs:
            layer_sigmas = sigma_acc.sigmas()
            neuron_sigmas = sigma_acc.per_neuron_sigmas()
            residual_sigma = residual_acc.sigma()
            train_mse = train_loss_sum / train_seen
            test_mse = evaluate_mse(model, x_test, y_test, loss_fn)

            rows = compute_model_layer_ei(
                model,
                sigma=layer_sigmas,
                n_samples=config.ei_samples,
                device=device,
                seed=config.seed + epoch,
                first_layer_mins=feature_mins,
                first_layer_widths=feature_widths,
                neuron_sigmas=neuron_sigmas,
            )
            overall_rows = compute_model_layer_ei(
                model,
                sigma=residual_sigma,
                n_samples=config.ei_samples,
                device=device,
                seed=config.seed + epoch,
                first_layer_mins=feature_mins,
                first_layer_widths=feature_widths,
            )
            overall_old = sum(r["ei_bits"] for r in overall_rows)
            overall_new = sum(r["ei_bits_floored"] for r in overall_rows)
            for row in rows:
                history.append(
                    {
                        "epoch": epoch,
                        "layer": row["layer"],
                        "n_in": row["n_in"],
                        "n_out": row["n_out"],
                        "sigma": row["sigma"],
                        "overall_sigma": residual_sigma,
                        "ei_bits": row["ei_bits"],
                        "ei_bits_floored": row["ei_bits_floored"],
                        "floor_gain_bits": row["floor_gain_bits"],
                        "tax_fix_bits": row["tax_fix_bits"],
                        "overall_ei_bits": overall_old,
                        "overall_ei_bits_floored": overall_new,
                        "train_mse": train_mse,
                        "test_mse": test_mse,
                    }
                )
            if epoch % 500 == 0 or epoch == 1:
                print(
                    f"  epoch {epoch:4d}  overall: old={overall_old:+8.3f}  new={overall_new:+8.3f} bit"
                    f"   | test MSE={test_mse:.6f}",
                    flush=True,
                )
    return history


def plot_comparison(history, run_dir):
    apply_style()
    run_dir = Path(run_dir)
    layers = sorted({r["layer"] for r in history})
    by_layer = {n: sorted([r for r in history if r["layer"] == n], key=lambda r: r["epoch"]) for n in layers}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    for ax, name in zip(axes.ravel()[:3], layers):
        rows = by_layer[name]
        ep = [r["epoch"] for r in rows]
        ax.plot(ep, [r["ei_bits"] for r in rows], color=OLD_COLOR, lw=2.2, label="original formula")
        ax.plot(ep, [r["ei_bits_floored"] for r in rows], color=NEW_COLOR, lw=2.2, label="noise-floor corrected")
        ax.axhline(0.0, color="gray", lw=0.6)
        ax.set_title(f"{name}  ({rows[0]['n_in']}→{rows[0]['n_out']})")
        ax.set_ylabel("EI (bit / input dim)")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8, loc="best")

    ax = axes[1][1]
    rows = by_layer[layers[0]]
    ep = [r["epoch"] for r in rows]
    ax.plot(ep, [r["overall_ei_bits"] for r in rows], color=OLD_COLOR, lw=2.4,
            label="overall (residual σ), original")
    ax.plot(ep, [r["overall_ei_bits_floored"] for r in rows], color=NEW_COLOR, lw=2.4,
            label="overall (residual σ), corrected")
    sum_old = [sum(r["ei_bits"] for r in history if r["epoch"] == e) for e in ep]
    sum_new = [sum(r["ei_bits_floored"] for r in history if r["epoch"] == e) for e in ep]
    ax.plot(ep, sum_old, color=OLD_COLOR, lw=1.6, ls="--", label="Σ layers (layer σ), original")
    ax.plot(ep, sum_new, color=NEW_COLOR, lw=1.6, ls="--", label="Σ layers (layer σ), corrected")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_title("Overall EI: residual σ (solid) vs Σ layers (dashed)")
    ax.set_ylabel("EI (bit)")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7.5, loc="best")

    fig.suptitle(
        "Continuous EI: original vs noise-floor corrected (IRIS, seed 42)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = run_dir / "noise_floor_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_outputs(history, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.outdir) / f"iris_ei_noise_floor_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "noise_floor_history.csv"
    fields = ["epoch", "layer", "n_in", "n_out", "sigma", "overall_sigma", "ei_bits",
              "ei_bits_floored", "floor_gain_bits", "tax_fix_bits",
              "overall_ei_bits", "overall_ei_bits_floored", "train_mse", "test_mse"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    fig_path = plot_comparison(history, run_dir)
    latest = publish_latest(run_dir, config.outdir)
    return {"csv": csv_path, "figure": fig_path, "latest": latest}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare original continuous EI with noise-floor corrected version.")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--ei-samples", type=int, default=Config.ei_samples)
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
        record_every=args.record_every,
        outdir=args.outdir,
    )
    history = train_and_compare(config)
    for key, path in write_outputs(history, config).items():
        print(f"Wrote {key}: {path}")


if __name__ == "__main__":
    main()
