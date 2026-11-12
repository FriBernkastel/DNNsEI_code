"""固定 sigmoid + SGD + w=1，在 4000 轮和 20000 轮两个时点拆解体积项。"""

import math
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from eicore import IrisSigmoidNet, load_iris_tensors  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_IN = [4, 5, 5]
N_OUT = [5, 5, 3]


def snapshot(model, tag, mse, rng):
    print(f"\n=== {tag} (train MSE={mse:.4f}) ===")
    print(f"{'层':>3} {'|W|均值':>8} {'a′(z)均值':>10} {'单奇异值≈':>10} "
          f"{'log2 pdet':>11} {'每层贡献':>9}")
    total = 0.0
    for idx, layer in enumerate(model.layers, start=1):
        W = layer.weight.detach().cpu().numpy().astype(np.float64)
        n_out, n_in = W.shape
        x = rng.uniform(0, 1, (3000, n_in))
        y = 1 / (1 + np.exp(-(x @ W.T)))
        ap = y * (1 - y)
        J = ap[:, :, None] * W[None, :, :]
        sv = np.linalg.svd(J, compute_uv=False)
        lp_bits = np.mean(np.log2(np.clip(sv, 1e-300, None)).sum(axis=1))
        contrib = lp_bits / n_in
        total += contrib
        print(f"L{idx:>2} {np.abs(W).mean():>8.3f} {ap.mean():>10.4f} "
              f"{2 ** (lp_bits / sv.shape[1]):>10.3f} {lp_bits:>11.2f} {contrib:>9.2f}")
    print(f"{'合计体积项':>3} {'':>8} {'':>10} {'':>10} {'':>11} {total:>9.2f}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    xtr, ytr, _, _, _, _ = load_iris_tensors(42)
    xtr, ytr = xtr.to(DEV), ytr.to(DEV)
    model = IrisSigmoidNet().to(device=DEV, dtype=torch.float64)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    lf = nn.MSELoss()
    g = torch.Generator(device="cpu"); g.manual_seed(42)
    loader = DataLoader(TensorDataset(xtr.cpu(), ytr.cpu()), batch_size=10, shuffle=True, generator=g)
    rng = np.random.default_rng(3)
    marks = {4000, 20000}
    for epoch in range(1, 20001):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); lf(model(xb), yb).backward(); opt.step()
        if epoch in marks:
            with torch.no_grad():
                mse = float(lf(model(xtr), ytr).item())
            snapshot(model, f"epoch {epoch}", mse, rng)


if __name__ == "__main__":
    main()
