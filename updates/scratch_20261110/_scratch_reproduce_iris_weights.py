"""Reproduce the IRIS sigmoid net from seed 42 and save its weights as .npz.

Verifies train/test MSE and per-layer continuous EI against the archived run
results/latest/iris_continuous_ei/layer_ei_history.csv (epoch 4000).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(r"D:\PythonProject\DNNsEI_code_20261029")
sys.path.insert(0, str(ROOT))

from eicore import (  # noqa: E402
    GradientSigmaAccumulator,
    IrisSigmoidNet,
    ResidualSigmaAccumulator,
    compute_model_layer_ei,
    evaluate_mse,
    load_iris_tensors,
)
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

SEED = 42
EPOCHS = 4000
BATCH = 10
LR = 0.01
EI_SAMPLES = 2000
OUT_NPZ = Path(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "transport_map_EI" / "data" / "iris_weights_seed42.npz"))


def reference_row(epoch: int = 4000) -> dict[str, dict]:
    path = ROOT / "results" / "latest" / "iris_continuous_ei" / "layer_ei_history.csv"
    out = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["epoch"]) == epoch:
                out[row["layer"]] = {k: float(v) for k, v in row.items() if k not in ("layer",)}
    return out


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")

    x_train, y_train, x_test, y_test, feature_mins, feature_widths = load_iris_tensors(SEED)
    model = IrisSigmoidNet().to(device=device, dtype=torch.float64)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=BATCH, shuffle=True, generator=gen
    )

    layer_names = [f"L{i}" for i in range(1, len(model.layers) + 1)]
    for epoch in range(1, EPOCHS + 1):
        model.train()
        sigma_acc = GradientSigmaAccumulator(layer_names=layer_names)
        residual_acc = ResidualSigmaAccumulator()
        for xb, yb in loader:
            optimizer.zero_grad()
            y_pred, acts = model.forward_with_activations(xb)
            loss = loss_fn(y_pred, yb)
            residual_acc.update(y_pred, yb)
            loss.backward()
            sigma_acc.update({f"L{i}": a.grad for i, a in enumerate(acts, start=1)})
            optimizer.step()
        if epoch % 500 == 0:
            print(f"  epoch {epoch}/{EPOCHS} done", flush=True)

    train_mse = evaluate_mse(model, x_train, y_train, loss_fn)
    test_mse = evaluate_mse(model, x_test, y_test, loss_fn)
    layer_sigmas = sigma_acc.sigmas()
    neuron_sigmas = sigma_acc.per_neuron_sigmas()
    residual_sigma = residual_acc.sigma()

    ref = reference_row(EPOCHS)
    print("\n== 复现校验（对照 results/latest/iris_continuous_ei, epoch 4000）==")
    print(f"  train_mse  now={train_mse:.10f}  ref={ref['L1']['train_mse']:.10f}  "
          f"diff={train_mse - ref['L1']['train_mse']:+.2e}")
    print(f"  test_mse   now={test_mse:.10f}  ref={ref['L1']['test_mse']:.10f}  "
          f"diff={test_mse - ref['L1']['test_mse']:+.2e}")

    rows = compute_model_layer_ei(
        model,
        sigma=layer_sigmas,
        n_samples=EI_SAMPLES,
        device=device,
        seed=SEED + EPOCHS,
        first_layer_mins=feature_mins,
        first_layer_widths=feature_widths,
        neuron_sigmas=neuron_sigmas,
    )
    print("\n  layer  EI(now)   EI(ref)    diff_bits   sigma(now)    sigma(ref)")
    for row in rows:
        name = row["layer"]
        print(
            f"  {name:5s} {row['ei_bits']:9.6f} {ref[name]['ei_bits']:9.6f} "
            f"{row['ei_bits'] - ref[name]['ei_bits']:+11.2e} "
            f"{row['sigma']:.9f} {ref[name]['sigma']:.9f}"
        )
    overall_rows = compute_model_layer_ei(
        model,
        sigma=residual_sigma,
        n_samples=EI_SAMPLES,
        device=device,
        seed=SEED + EPOCHS,
        first_layer_mins=feature_mins,
        first_layer_widths=feature_widths,
    )
    overall_ei = sum(r["ei_bits"] for r in overall_rows)
    print(f"\n  overall_sigma now={residual_sigma:.10f} ref={ref['L1']['overall_sigma']:.10f}")
    print(f"  overall EI now={overall_ei:.6f} bits  ref={ref['L1']['overall_ei_bits']:.6f} bits")

    payload = {"residual_sigma": np.float64(residual_sigma)}
    for i, layer in enumerate(model.layers, start=1):
        payload[f"W{i}"] = layer.weight.detach().cpu().numpy().astype(np.float64)
        payload[f"sigma{i}"] = np.float64(layer_sigmas[f"L{i}"])
        payload[f"neuron_sigma{i}"] = np.asarray(neuron_sigmas[f"L{i}"], dtype=np.float64)
    payload["feature_mins"] = feature_mins.cpu().numpy()
    payload["feature_widths"] = feature_widths.cpu().numpy()
    np.savez(OUT_NPZ, **payload)
    print(f"\nWrote {OUT_NPZ}")
    for i in (1, 2, 3):
        print(f"  W{i} shape={payload[f'W{i}'].shape}")


if __name__ == "__main__":
    main()
