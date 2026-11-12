"""组合扫描：激活 x 权重缩放 x 注入宽度，找"体积项最不负"的组合，并算出对应的 MSE 门槛。"""

import math
import sys
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from eicore import IrisSigmoidNet, load_iris_tensors  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_OUT, N_IN = [5, 5, 3], [4, 5, 5]
K = 2 * math.pi * math.e
B_SUM = sum(o / n for o, n in zip(N_OUT, N_IN))

ACTS = {
    "sigmoid": lambda z: (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z))),
    "tanh": lambda z: 1 - np.tanh(z) ** 2,
}


def train(epochs=4000):
    torch.manual_seed(42)
    np.random.seed(42)
    xtr, ytr, _, _, _, _ = load_iris_tensors(42)
    xtr, ytr = xtr.to(DEV), ytr.to(DEV)
    m = IrisSigmoidNet().to(device=DEV, dtype=torch.float64)
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    lf = nn.MSELoss()
    g = torch.Generator(device="cpu"); g.manual_seed(42)
    loader = DataLoader(TensorDataset(xtr.cpu(), ytr.cpu()), batch_size=10, shuffle=True, generator=g)
    for _ in range(epochs):
        m.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
    with torch.no_grad():
        mse = float(lf(m(xtr), ytr).item())
    return [m.layers[i].weight.detach().cpu().numpy().astype(np.float64) for i in range(3)], mse


def contrib(Ws, act, scale, width, n=3000, seed=7):
    rng = np.random.default_rng(seed)
    total = 0.0
    for W, n_in in zip(Ws, N_IN):
        W = W * scale
        x = rng.uniform(0, width, (n, W.shape[1]))
        ap = ACTS[act](x @ W.T)
        J = ap[:, :, None] * W[None, :, :]
        sv = np.linalg.svd(J, compute_uv=False)
        lp = np.mean(np.log2(np.clip(sv, 1e-300, None)).sum(axis=1))
        total += (lp + n_in * math.log2(width)) / n_in
    return total


if __name__ == "__main__":
    Ws, mse = train(4000)
    print(f"baseline MSE={mse:.4f}\n")
    rows = []
    for act, scale, width in product(("sigmoid", "tanh"), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)):
        c = contrib(Ws, act, scale, width)
        rows.append((act, scale, width, c, 2 ** (-2 * abs(c) / B_SUM) / K))
    rows.sort(key=lambda r: -abs(r[3]))
    print(f"{'激活':8s} {'scale':>5s} {'w':>4s} {'体积项(含H(X))':>14s} {'MSE*(EI=0)':>11s} {'Adam能否达到(0.0056)':>20s}")
    for act, s, w, c, ms in rows:
        ok = "✓" if 0.0056 <= ms else "✗"
        print(f"{act:8s} {s:>5.1f} {w:>4.1f} {c:>14.2f} {ms:>11.5f} {ok:>20s}")
