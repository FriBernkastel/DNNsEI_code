"""方法②预览：用 kNN 熵估计（Kozachenko-Leonenko）替代"局部体积"近似来算 h(Y)。

EI = h(Y) - h(eps)，其中
  Y = f(X) + eps,  X ~ U(0,1)^n_in,  eps ~ N(0, sigma^2)
  h(eps)  用高斯解析式：0.5 * sum log(2*pi*e*sigma_j^2)
  h(Y)    用 kNN 估计：psi(N) - psi(k) + log(V_d) + d/N * sum log(rho_i)

只读模型、不改原程序。先验证估计器，再给出每层 EI 的预览值。
"""

import math
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from scipy.special import digamma
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from eicore import IrisSigmoidNet, load_iris_tensors  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def knn_entropy_nats(Y, k=5):
    """Kozachenko-Leonenko kNN 熵估计（单位 nats）。Y: (N, d)。"""
    N, d = Y.shape
    tree = cKDTree(Y)
    dist, _ = tree.query(Y, k=k + 1, workers=-1)   # 第 0 列是自身
    rho = dist[:, k]
    rho = np.clip(rho, 1e-300, None)
    log_unit_ball = math.log(math.pi ** (d / 2) / math.gamma(d / 2 + 1))
    return float(digamma(N) - digamma(k) + log_unit_ball + d * np.mean(np.log(rho)))


def gauss_entropy_nats(sigma, d):
    return 0.5 * d * math.log(2 * math.pi * math.e * sigma * sigma)


def normalize_inputs(mins, widths):
    return (torch.tensor(mins, dtype=torch.float64), torch.tensor(widths, dtype=torch.float64))


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
    return m


def main():
    # ---------- 1. 先验证估计器：已知高斯 ----------
    rng = np.random.default_rng(0)
    for d, sig in ((3, 0.3), (5, 0.1)):
        Y = rng.normal(0, sig, size=(20000, d))
        est = knn_entropy_nats(Y, k=5)
        true = gauss_entropy_nats(sig, d)
        print(f"校验 d={d} sigma={sig}: 估计 {est:+.4f} nats, 理论 {true:+.4f} nats, "
              f"差 {est-true:+.4f} ({100*(est-true)/abs(true):+.1f}%)")

    # ---------- 2. 训练模型，逐层比较两种 EI ----------
    model = train(4000)
    Ws = [l.weight.detach().cpu().numpy().astype(np.float64) for l in model.layers]
    rng = np.random.default_rng(1)
    N = 20000
    print("\n每层 EI（nats -> bits，除以 n_in 归一化）：")
    print(f"{'层':>3} {'n_in->n_out':>10} {'sigma':>7} {'h(Y) kNN':>10} {'h(eps)':>9} "
          f"{'EI_knn(bits)':>13} {'EI_formula(bits)':>16}")
    # 用 4000 轮时的残差 σ（三组共享）与每层梯度 σ 两种都算
    res_sigma = 0.342368
    layer_sigmas = [0.002199, 0.004409, 0.022825]
    for idx, (W, sig) in enumerate(zip(Ws, layer_sigmas), start=1):
        n_out, n_in = W.shape
        x = rng.uniform(0, 1, (N, n_in))
        y = 1 / (1 + np.exp(-(x @ W.T)))
        noise = rng.normal(0, sig, size=y.shape)
        Y = y + noise
        hY = knn_entropy_nats(Y, k=5)
        he = gauss_entropy_nats(sig, n_out)
        ei_knn = (hY - he) / n_in / math.log(2)
        # 公式法（体积项）
        ap = y * (1 - y)
        J = ap[:, :, None] * W[None, :, :]
        sv = np.linalg.svd(J, compute_uv=False)
        lp = np.mean(np.log(np.clip(sv, 1e-300, None)).sum(axis=1))
        term1 = -0.5 * n_out * math.log(2 * math.pi * math.e * sig * sig)
        ei_f = (term1 + lp) / n_in / math.log(2)
        print(f"L{idx:>2} {n_in:>4}->{n_out:<4} {sig:>7.4f} {hY:>10.4f} {he:>9.4f} "
              f"{ei_knn:>13.3f} {ei_f:>16.3f}")

    print("\n用残差 sigma=0.342（Overall 口径）时逐层 kNN EI：")
    for idx, W in enumerate(Ws, start=1):
        n_out, n_in = W.shape
        x = rng.uniform(0, 1, (N, n_in))
        y = 1 / (1 + np.exp(-(x @ W.T)))
        Y = y + rng.normal(0, res_sigma, size=y.shape)
        hY = knn_entropy_nats(Y, k=5)
        he = gauss_entropy_nats(res_sigma, n_out)
        print(f"  L{idx}: h(Y)={hY:+.3f}  h(eps)={he:+.3f}  EI={(hY-he)/n_in/math.log(2):+.4f} bits")


if __name__ == "__main__":
    main()
