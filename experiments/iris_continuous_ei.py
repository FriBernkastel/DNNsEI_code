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
from eicore.report import publish_latest


@dataclass(frozen=True)
class Config:
    epochs: int = 4000
    batch_size: int = 10
    lr: float = 0.01
    seed: int = 42
    ei_samples: int = 2000
    record_every: int = 10
    outdir: str = "results"


def train_and_measure(config):
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
        layer_names = [f"L{i}" for i in range(1, len(model.layers) + 1)]
        sigma_acc = GradientSigmaAccumulator(layer_names=layer_names)
        residual_sigma_acc = ResidualSigmaAccumulator()
        train_loss_sum = 0.0
        train_seen = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            y_pred, activations = model.forward_with_activations(xb)
            loss = loss_fn(y_pred, yb)
            residual_sigma_acc.update(y_pred, yb)
            loss.backward()
            sigma_acc.update(
                {
                    f"L{i}": activation.grad
                    for i, activation in enumerate(activations, start=1)
                }
            )
            optimizer.step()

            train_loss_sum += float(loss.item()) * xb.shape[0]
            train_seen += xb.shape[0]

        if epoch == 1 or epoch % config.record_every == 0 or epoch == config.epochs:
            layer_sigmas = sigma_acc.sigmas()
            layer_neuron_sigmas = sigma_acc.per_neuron_sigmas()
            residual_sigma = residual_sigma_acc.sigma()
            train_mse = train_loss_sum / train_seen
            test_mse = evaluate_mse(model, x_test, y_test, loss_fn)
            ei_rows = compute_model_layer_ei(
                model,
                sigma=layer_sigmas,
                n_samples=config.ei_samples,
                device=device,
                seed=config.seed + epoch,
                first_layer_mins=feature_mins,
                first_layer_widths=feature_widths,
                neuron_sigmas=layer_neuron_sigmas,
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
            overall_ei = sum(row["ei_bits"] for row in overall_rows)
            total_sensitivity = sum(row["sensitivity_bits"] for row in ei_rows)
            total_degeneracy = sum(row["degeneracy_bits"] for row in ei_rows)
            for row in ei_rows:
                history.append(
                    {
                        "epoch": epoch,
                        "train_mse": train_mse,
                        "test_mse": test_mse,
                        "overall_sigma": residual_sigma,
                        "overall_ei_bits": overall_ei,
                        **row,
                    }
                )
            sigma_text = ", ".join(f"{name}={layer_sigmas[name]:.6f}" for name in sorted(layer_sigmas))
            print(
                f"Epoch {epoch:3d}/{config.epochs} | "
                f"Train MSE={train_mse:.6f} | Test MSE={test_mse:.6f} | "
                f"layer_sigma({sigma_text}) | overall_sigma={residual_sigma:.6f} | "
                f"Overall EI={overall_ei:.6f} bits | "
                f"Total Sens={total_sensitivity:.6f} bits | Total Deg={total_degeneracy:.6f} bits",
                flush=True,
            )

    return history


def summarize_overall_ei(history):
    by_epoch = {}
    for row in history:
        epoch = int(row["epoch"])
        entry = by_epoch.setdefault(
            epoch,
            {
                "epoch": epoch,
                "train_mse": float(row["train_mse"]),
                "test_mse": float(row["test_mse"]),
                "overall_sigma": float(row.get("overall_sigma", row["sigma"])),
                "overall_ei_bits": float(row.get("overall_ei_bits", 0.0)),
            },
        )
        if "overall_ei_bits" not in row:
            entry["overall_ei_bits"] += float(row["ei_bits"])
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def plot_curves(history, run_dir):
    from eicore.plotting import apply_style, finalize_figure, plot_extra_series, plot_layer_metric

    apply_style()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    overall_rows = summarize_overall_ei(history)
    epochs = [row["epoch"] for row in overall_rows]

    loss_path = run_dir / "loss_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [row["train_mse"] for row in overall_rows], label="Train MSE")
    ax.plot(epochs, [row["test_mse"] for row in overall_rows], label="Test MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("IRIS MSE During Training")
    ax.legend(loc="best")
    finalize_figure(fig, loss_path)

    paths = {"loss": loss_path}
    has_decomposition = any("sensitivity_bits" in row for row in history)
    if has_decomposition:
        decomposition_path = run_dir / "ei_decomposition.png"
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        plot_layer_metric(axes[0], history, "ei_bits")
        plot_extra_series(
            axes[0],
            epochs,
            [row["overall_ei_bits"] for row in overall_rows],
            label="Overall EI",
        )
        axes[0].set_ylabel("EI (bits)")
        axes[0].set_title("Layer EI")
        plot_layer_metric(axes[1], history, "sensitivity_bits", total_label="Total")
        axes[1].set_ylabel("Sensitivity (bits)")
        axes[1].set_title("Continuous Sensitivity")
        plot_layer_metric(axes[2], history, "degeneracy_bits", total_label="Total")
        axes[2].set_ylabel("Degeneracy (bits)")
        axes[2].set_title("Continuous Degeneracy")
        finalize_figure(fig, decomposition_path, suptitle="IRIS Continuous EI Decomposition During Training")
        paths["decomposition"] = decomposition_path
    else:
        ei_path = run_dir / "ei_curves.png"
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_layer_metric(ax, history, "ei_bits")
        plot_extra_series(
            ax,
            epochs,
            [row["overall_ei_bits"] for row in overall_rows],
            label="Overall EI",
        )
        ax.set_ylabel("EI (bits)")
        ax.set_title("Layer EI and Overall EI During Training")
        finalize_figure(fig, ei_path)
        paths["ei"] = ei_path
    return paths


def write_history(history, outdir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(outdir) / f"iris_continuous_ei_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "layer_ei_history.csv"
    overall_csv_path = run_dir / "overall_ei_history.csv"
    fieldnames = [
        "epoch",
        "layer",
        "n_in",
        "n_out",
        "input_width",
        "train_mse",
        "test_mse",
        "sigma",
        "overall_sigma",
        "overall_ei_bits",
        "ei_bits",
        "term1_bits",
        "term2_bits",
        "log_pdet_mean_nats",
        "sensitivity_bits",
        "sens_term1_bits",
        "sens_transmission_bits",
        "degeneracy_bits",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    overall_fieldnames = ["epoch", "train_mse", "test_mse", "overall_sigma", "overall_ei_bits"]
    with overall_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=overall_fieldnames)
        writer.writeheader()
        writer.writerows(summarize_overall_ei(history))
    plot_paths = plot_curves(history, run_dir)
    latest_dir = publish_latest(run_dir, outdir)
    return {
        "layer_csv": csv_path,
        "overall_csv": overall_csv_path,
        **plot_paths,
        "latest_dir": latest_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train IRIS sigmoid network and track continuous layer EI.")
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
    history = train_and_measure(config)
    outputs = write_history(history, config.outdir)
    for path in outputs.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
