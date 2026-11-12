import argparse
import csv
import gzip
import math
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from eicore import (
    GradientSigmaAccumulator,
    IrisSigmoidNet,
    ResidualSigmaAccumulator,
    compute_model_layer_ei,
    one_hot,
)
from experiments.iris_continuous_ei import plot_curves, summarize_overall_ei


@dataclass(frozen=True)
class Config:
    epochs: int = 500
    batch_size: int = 50
    lr: float = 0.01
    seed: int = 42
    ei_samples: int = 2000
    record_every: int = 10
    data_dir: str = "data/MNIST/raw"
    outdir: str = "results"


def make_model():
    return IrisSigmoidNet(dims=(25, 6, 6, 5))


def _open_idx(path):
    path = Path(path)
    if path.exists():
        return path.open("rb")
    gz_path = path.with_suffix(path.suffix + ".gz")
    if gz_path.exists():
        return gzip.open(gz_path, "rb")
    raise FileNotFoundError(f"Missing MNIST IDX file: {path} or {gz_path}")


def read_idx_images(path):
    with _open_idx(path) as f:
        magic, n_images, n_rows, n_cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image IDX magic {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n_images, n_rows, n_cols)


def read_idx_labels(path):
    with _open_idx(path) as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label IDX magic {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n_labels)


def filter_digits_zero_to_four(images, labels):
    labels = labels.astype(np.int64)
    mask = labels < 5
    return images[mask], labels[mask]


def downsample_images_to_5x5(images):
    x = torch.tensor(images, dtype=torch.float64).unsqueeze(1) / 255.0
    x = F.interpolate(x, size=(5, 5), mode="area")
    return x.reshape(x.shape[0], 25)


def load_reduced_mnist_tensors(data_dir):
    data_dir = Path(data_dir)
    train_images = read_idx_images(data_dir / "train-images-idx3-ubyte")
    train_labels = read_idx_labels(data_dir / "train-labels-idx1-ubyte")
    test_images = read_idx_images(data_dir / "t10k-images-idx3-ubyte")
    test_labels = read_idx_labels(data_dir / "t10k-labels-idx1-ubyte")

    train_images, train_labels = filter_digits_zero_to_four(train_images, train_labels)
    test_images, test_labels = filter_digits_zero_to_four(test_images, test_labels)

    x_train = downsample_images_to_5x5(train_images)
    x_test = downsample_images_to_5x5(test_images)
    y_train = torch.tensor(one_hot(train_labels, 5), dtype=torch.float64)
    y_test = torch.tensor(one_hot(test_labels, 5), dtype=torch.float64)
    return x_train, y_train, x_test, y_test


def evaluate_mse(model, x, y, loss_fn):
    model.eval()
    with torch.no_grad():
        return float(loss_fn(model(x), y).item())


def train_and_measure(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, x_test, y_test = load_reduced_mnist_tensors(config.data_dir)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    model = make_model().to(device=device, dtype=torch.float64)
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
    first_layer_mins = torch.zeros(25, dtype=torch.float64, device=device)
    first_layer_widths = torch.ones(25, dtype=torch.float64, device=device)

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
            residual_sigma = residual_sigma_acc.sigma()
            train_mse = train_loss_sum / train_seen
            test_mse = evaluate_mse(model, x_test, y_test, loss_fn)
            ei_rows = compute_model_layer_ei(
                model,
                sigma=layer_sigmas,
                n_samples=config.ei_samples,
                device=device,
                seed=config.seed + epoch,
                first_layer_mins=first_layer_mins,
                first_layer_widths=first_layer_widths,
            )
            overall_rows = compute_model_layer_ei(
                model,
                sigma=residual_sigma,
                n_samples=config.ei_samples,
                device=device,
                seed=config.seed + epoch,
                first_layer_mins=first_layer_mins,
                first_layer_widths=first_layer_widths,
            )
            overall_ei = sum(row["ei_bits"] for row in overall_rows)
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
                f"Overall EI={overall_ei:.6f} bits",
                flush=True,
            )

    return history


def write_history(history, outdir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(outdir) / f"mnist_continuous_ei_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    layer_csv_path = run_dir / "layer_ei_history.csv"
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
    ]
    with layer_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    overall_fieldnames = ["epoch", "train_mse", "test_mse", "overall_sigma", "overall_ei_bits"]
    with overall_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=overall_fieldnames)
        writer.writeheader()
        writer.writerows(summarize_overall_ei(history))

    plot_paths = plot_curves(history, run_dir)
    return {
        "layer_csv": layer_csv_path,
        "overall_csv": overall_csv_path,
        **plot_paths,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train reduced-MNIST sigmoid network and track continuous EI.")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--ei-samples", type=int, default=Config.ei_samples)
    parser.add_argument("--record-every", type=int, default=Config.record_every)
    parser.add_argument("--data-dir", type=str, default=Config.data_dir)
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
        data_dir=args.data_dir,
        outdir=args.outdir,
    )
    history = train_and_measure(config)
    outputs = write_history(history, config.outdir)
    for path in outputs.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
