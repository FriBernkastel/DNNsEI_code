"""Compute EI on the reproduced IRIS weights with the transport-map (TM-EI) estimator."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\PythonProject\DNNsEI_code_20261029")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, r"D:\PythonProject")

from experiments.transport_map_ei import estimate_transport_map_mutual_information as tm_mi  # noqa: E402

LN2 = np.log(2.0)
NPZ = Path(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "transport_map_EI" / "data" / "iris_weights_seed42.npz"))

# 归档结果（epoch 4000），见 results/latest/iris_{continuous,discrete}_ei/layer_ei_history.csv
REF = {
    "L1": {"n_in": 4, "n_out": 5, "cont_perdim": 5.695192766368154, "cont_raw": 22.780771065472616,
           "disc_perdim": 2.514700558146253, "disc_raw": 10.058802232585013},
    "L2": {"n_in": 5, "n_out": 5, "cont_perdim": 2.189728377158652, "cont_raw": 10.948641885793260,
           "disc_perdim": 2.1044596358131984, "disc_raw": 10.522298179065992},
    "L3": {"n_in": 5, "n_out": 3, "cont_perdim": 0.9215081678801678, "cont_raw": 4.607540839400839,
           "disc_perdim": 2.016371029517777, "disc_raw": 10.081855147588884},
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def tm_ei_bits(weight, cause, noise_std, *, jitter=1e-6):
    """c ~ cause, e = sigmoid(c @ W.T) + N(0, noise_std^2). Returns TM-EI in bits."""
    effect = sigmoid(cause @ weight.T)
    if noise_std > 0:
        effect = effect + noise_std * np.random.default_rng(1234).normal(size=effect.shape)
    return tm_mi(cause, effect, jitter=jitter)["mi"] / LN2


def main():
    data = np.load(NPZ)
    n_samples = 20000
    rng = np.random.default_rng(42)
    print("TM-EI on reproduced IRIS weights (seed 42, epoch 4000)\n")

    print("== A. 与归档结果同口径：cause ~ U(0,1)^n_in，effect = sigmoid(Wc) + N(0, sigma_layer^2) ==")
    print(f"   n={n_samples}，TM-EI 已除 n_in 换算成与 ei_bits 一致的口径\n")
    hdr = (f"  {'layer':5s} {'n_in':>4} {'n_out':>5} {'sigma':>9} "
           f"{'TM-EI':>9} {'cont EI':>9} {'disc EI':>9}   {'TM/cont':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i in (1, 2, 3):
        name = f"L{i}"
        W = data[f"W{i}"]
        sigma = float(data[f"sigma{i}"])
        n_in, n_out = W.shape[1], W.shape[0]
        cause = rng.random((n_samples, n_in))
        total = tm_ei_bits(W, cause, sigma)
        perdim = total / n_in
        print(f"  {name:5s} {n_in:4d} {n_out:5d} {sigma:9.6f} {perdim:9.4f} "
              f"{REF[name]['cont_perdim']:9.4f} {REF[name]['disc_perdim']:9.4f} "
              f"{perdim / REF[name]['cont_perdim']:8.3f}")

    print("\n== B. 对噪声尺度的敏感性（TM-EI 必须靠噪声才有定义）==")
    print(f"  {'layer':5s} {'eps=0(.1σ)':>11} {'1σ':>9} {'10σ':>9} {'100σ':>9} {'确定性(ε=0)':>12}")
    for i in (1, 2, 3):
        W = data[f"W{i}"]
        sigma = float(data[f"sigma{i}"])
        n_in = W.shape[1]
        cause = rng.random((n_samples, n_in))
        row = [tm_ei_bits(W, cause, k * sigma) / n_in for k in (0.1, 1.0, 10.0, 100.0)]
        det = tm_ei_bits(W, cause, 0.0) / n_in
        print(f"  L{i}   {row[0]:11.4f} {row[1]:9.4f} {row[2]:9.4f} {row[3]:9.4f} {det:12.4f}")

    print("\n== C. 确定性情形（ε=0）只由 jitter 决定 ==")
    cause = rng.random((n_samples, 4))
    for jit in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        v = tm_ei_bits(data["W1"], cause, 0.0, jitter=jit) / 4
        print(f"  L1 jitter={jit:8.1e}  TM-EI={v:9.4f} bit   (连续口径参考 5.6952)")

    print("\n== D. 用真实 IRIS 输入分布（n=150，非 U(0,1)）==")
    from eicore import load_iris_tensors  # noqa: E402
    import torch  # noqa: E402

    x_tr, y_tr, x_te, y_te, fmin, fwid = load_iris_tensors(42)
    x_all = torch.cat([x_tr, x_te]).numpy()
    print(f"  x_all shape={x_all.shape}  min={x_all.min(axis=0)}  max={x_all.max(axis=0)}")
    acts = x_all
    for i in (1, 2, 3):
        W = data[f"W{i}"]
        out = sigmoid(acts @ W.T)
        total = tm_mi(acts, out, jitter=1e-6)["mi"] / LN2
        print(f"  L{i}: I(in ; out) = {total:8.4f} bit total, {total / W.shape[1]:8.4f} bit per-dim")
        acts = out


if __name__ == "__main__":
    main()
