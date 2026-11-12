"""扫描"体积项"的可调杠杆：权重幅度、激活函数、注入区间宽度。

只测量，不改模型结构，也不改 EI 公式。输出每个配置下 log2 pdet 的合计贡献，
以及"要让 Overall EI = 0 所需的最低 MSE"。
"""

import math
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from eicore import IrisSigmoidNet, load_iris_tensors  # noqa: E402
from eicore.discrete import sigmoid_np  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_OUT, N_IN = [5, 5, 3], [4, 5, 5]
K = 2 * math.pi * math.e
B_SUM = sum(o / n for o, n in zip(N_OUT, N_IN))     # 2.85


def train(epochs=4000):
    torch.manual_seed(42)
    np.random.seed(42)
    xtr, ytr, _, _, _, _ = load_iris_tensors(42)
    xtr, ytr = xtr.to(DEV), ytr.to(DEV)
    model = IrisSigmoidNet().to(device=DEV, dtype=torch.float64)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    g = torch.Generator(device="cpu"); g.manual_seed(42)
    loader = DataLoader(TensorDataset(xtr.cpu(), ytr.cpu()), batch_size=10,
                        shuffle=True, generator=g)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    with torch.no_grad():
        mse = float(loss_fn(model(xtr), ytr).item())
    return [model.layers[i].weight.detach().cpu().numpy().astype(np.float64) for i in range(3)], mse


ACTS = {
    "sigmoid": lambda z: (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z))),
    "tanh": lambda z: 1 - np.tanh(z) ** 2,
}


def log2_pdet_by_layer(Ws, scale=1.0, act="sigmoid", width=1.0, n=3000, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for W in Ws:
        W = W * scale
        n_out, n_in = W.shape
        x = rng.uniform(0, width, (n, n_in))
        z = x @ W.T
        ap = ACTS[act](z)
        J = ap[:, :, None] * W[None, :, :]
        sv = np.linalg.svd(J, compute_uv=False)
        lp = np.mean(np.log2(np.clip(sv, 1e-300, None)).sum(axis=1))
        out.append(lp)
    return out


def report(label, Ws, **kw):
    pd = log2_pdet_by_layer(Ws, **kw)
    contrib = sum(p / n for p, n in zip(pd, N_IN))
    A = abs(contrib)
    mse_star = 2 ** (-2 * A / B_SUM) / K
    print(f"{label:34s} pdet={ ['%+.1f' % p for p in pd] }  体积项={contrib:+6.2f}  "
          f"MSE*(EI=0)={mse_star:.5f}")
    return contrib, mse_star


if __name__ == "__main__":
    Ws, mse = train(4000)
    print(f"baseline: train MSE={mse:.4f}\n")

    print("--- 杠杆 A：权重整体放大（不重训，只缩放） ---")
    for s in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        report(f"scale W x {s:g}", Ws, scale=s)

    print("\n--- 杠杆 B：激活函数（同一组权重，换导数） ---")
    report("sigmoid", Ws, act="sigmoid")
    report("tanh", Ws, act="tanh")

    print("\n--- 杠杆 C：注入区间宽度 w ---")
    for w in (0.5, 1.0, 2.0, 3.0, 5.0):
        pd = log2_pdet_by_layer(Ws, width=w)
        hx_bits = math.log2(w)                      # 每层每维 +log2 w
        contrib = sum((p + n * hx_bits) / n for p, n in zip(pd, N_IN))
        A = abs(contrib)
        print(f"w={w:<4g} pdet={ ['%+.1f' % p for p in pd] }  H(X)={hx_bits:+.2f}/层  "
              f"体积项+H(X)={contrib:+6.2f}  MSE*(EI=0)={2**(-2*A/B_SUM)/K:.5f}")
