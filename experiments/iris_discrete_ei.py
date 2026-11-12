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

from eicore import IrisSigmoidNet, compute_model_layer_ei_discrete, evaluate_mse, load_iris_tensors
from eicore.report import publish_latest


@dataclass(frozen=True)
class DiscreteConfig:
    epochs: int = 4000
    batch_size: int = 10
    lr: float = 0.01
    seed: int = 42
    ei_samples: int = 100000
    bins: int = 32
    record_every: int = 10
    outdir: str = "results"


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
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(x_train.cpu(), y_train.cpu()),
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_seen = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * xb.shape[0]
            train_seen += xb.shape[0]

        if epoch == 1 or epoch % config.record_every == 0 or epoch == config.epochs:
            train_mse = train_loss_sum / train_seen
            test_mse = evaluate_mse(model, x_test, y_test, loss_fn)
            ei_rows = compute_model_layer_ei_discrete(
                model,
                n_samples=config.ei_samples,
                bins=config.bins,
                seed=config.seed + epoch,
            )
            total_ei = sum(row["ei_bits"] for row in ei_rows)
            total_sens = sum(row["sensitivity_bits"] for row in ei_rows)
            total_deg = sum(row["degeneracy_bits"] for row in ei_rows)
            for row in ei_rows:
                history.append(
                    {
                        "epoch": epoch,
                        "bins": config.bins,
                        "train_mse": train_mse,
                        "test_mse": test_mse,
                        "total_ei_bits": total_ei,
                        "total_sensitivity_bits": total_sens,
                        "total_degeneracy_bits": total_deg,
                        **row,
                    }
                )
            print(
                f"Epoch {epoch:3d}/{config.epochs} | "
                f"Train MSE={train_mse:.6f} | Test MSE={test_mse:.6f} | "
                f"Total EI={total_ei:.4f} bits | Total Sens={total_sens:.4f} bits | "
                f"Total Deg={total_deg:.4f} bits",
                flush=True,
            )

    return history


def summarize_totals(history):
    by_epoch = {}
    for row in history:
        epoch = int(row["epoch"])
        by_epoch.setdefault(
            epoch,
            {
                "epoch": epoch,
                "train_mse": float(row["train_mse"]),
                "test_mse": float(row["test_mse"]),
                "total_ei_bits": float(row["total_ei_bits"]),
                "total_sensitivity_bits": float(row["total_sensitivity_bits"]),
                "total_degeneracy_bits": float(row["total_degeneracy_bits"]),
            },
        )
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def plot_curves(history, run_dir):
    from eicore.plotting import apply_style, finalize_figure, plot_extra_series, plot_layer_metric

    apply_style()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    totals = summarize_totals(history)
    epochs = [row["epoch"] for row in totals]

    loss_path = run_dir / "loss_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [row["train_mse"] for row in totals], label="Train MSE")
    ax.plot(epochs, [row["test_mse"] for row in totals], label="Test MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("IRIS MSE During Training (Discrete EI Run)")
    ax.legend(loc="best")
    finalize_figure(fig, loss_path)

    decomposition_path = run_dir / "ei_decomposition.png"
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    plot_layer_metric(axes[0], history, "ei_bits")
    plot_extra_series(axes[0], epochs, [row["total_ei_bits"] for row in totals], label="Total")
    axes[0].set_ylabel("EI (bits)")
    axes[0].set_title("Layer EI")
    plot_layer_metric(axes[1], history, "sensitivity_bits")
    plot_extra_series(axes[1], epochs, [row["total_sensitivity_bits"] for row in totals], label="Total")
    axes[1].set_ylabel("Sensitivity (bits)")
    axes[1].set_title("Discrete Sensitivity")
    plot_layer_metric(axes[2], history, "degeneracy_bits")
    plot_extra_series(axes[2], epochs, [row["total_degeneracy_bits"] for row in totals], label="Total")
    axes[2].set_ylabel("Degeneracy (bits)")
    axes[2].set_title("Discrete Degeneracy")
    finalize_figure(fig, decomposition_path, suptitle="IRIS Discrete EI Decomposition During Training")

    return {"loss": loss_path, "decomposition": decomposition_path}


def write_history(history, outdir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(outdir) / f"iris_discrete_ei_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "layer_ei_history.csv"
    fieldnames = [
        "epoch",
        "layer",
        "n_in",
        "n_out",
        "bins",
        "train_mse",
        "test_mse",
        "ei_bits",
        "h_e_bits",
        "h_e_given_c_bits",
        "sensitivity_bits",
        "degeneracy_bits",
        "ei_norm_bits",
        "sensitivity_norm_bits",
        "degeneracy_norm_bits",
        "total_ei_bits",
        "total_sensitivity_bits",
        "total_degeneracy_bits",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    plot_paths = plot_curves(history, run_dir)
    latest_dir = publish_latest(run_dir, outdir)
    return {"layer_csv": csv_path, **plot_paths, "latest_dir": latest_dir}


def parse_args():
    parser = argparse.ArgumentParser(description="Train IRIS sigmoid network and track discrete layer EI (paper method).")
    parser.add_argument("--epochs", type=int, default=DiscreteConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=DiscreteConfig.batch_size)
    parser.add_argument("--lr", type=float, default=DiscreteConfig.lr)
    parser.add_argument("--seed", type=int, default=DiscreteConfig.seed)
    parser.add_argument("--ei-samples", type=int, default=DiscreteConfig.ei_samples)
    parser.add_argument("--bins", type=int, default=DiscreteConfig.bins)
    parser.add_argument("--record-every", type=int, default=DiscreteConfig.record_every)
    parser.add_argument("--outdir", type=str, default=DiscreteConfig.outdir)
    return parser.parse_args()


def main():
    args = parse_args()
    config = DiscreteConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        ei_samples=args.ei_samples,
        bins=args.bins,
        record_every=args.record_every,
        outdir=args.outdir,
    )
    history = train_and_measure(config)
    outputs = write_history(history, config.outdir)
    for path in outputs.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
