"""端到端验证：让连续版 Overall EI 转正。
配置 A: sigmoid, 注入宽度 w=3, Adam lr=0.01, 10000 轮
配置 B: tanh,   注入宽度 w=2, Adam lr=0.01, 10000 轮

只改训练/测量口径，EI 公式不变（仍是 eicore.continuous 那套）。
输出：results/updates/overall_positive.png + results/updates/overall_positive.csv
"""

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from eicore import load_iris_tensors  # noqa: E402
from eicore.sigma import GradientSigmaAccumulator, ResidualSigmaAccumulator  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYER_COLORS = {"L1": "#1f77b4", "L2": "#ff7f0e", "L3": "#2ca02c"}
TOTAL_COLOR = "#222222"


class Net(nn.Module):
    def __init__(self, dims=(4, 5, 5, 3), act="sigmoid"):
        super().__init__()
        self.act = act
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(len(dims) - 1)
        )
        for layer in self.layers:
            limit = 1.0 / math.sqrt(layer.in_features)
            nn.init.uniform_(layer.weight, -limit, limit)

    def fwd(self, x):
        acts = []
        for layer in self.layers:
            z = layer(x)
            x = torch.tanh(z) if self.act == "tanh" else torch.sigmoid(z)
            x.retain_grad()
            acts.append(x)
        return x, acts

    def forward(self, x):
        return self.fwd(x)[0]


def deriv(act, z):
    if act == "tanh":
        t = torch.tanh(z)
        return 1.0 - t * t
    s = torch.sigmoid(z)
    return s * (1.0 - s)


def layer_ei(W, bias, samples, sigma, width, act, eps=1e-12):
    n_out, n_in = W.shape
    z = samples @ W.t() + (bias if bias is not None else 0.0)
    ap = deriv(act, z)
    J = ap[:, :, None] * W[None, :, :]
    sv = torch.linalg.svdvals(J)
    log_pdet = float(torch.mean(torch.sum(torch.log(torch.clamp(sv, min=eps)), dim=1)).item())
    hx = n_in * math.log(width)
    term1 = -0.5 * n_out * math.log(2.0 * math.pi * math.e * sigma * sigma)
    ei = (term1 + hx + log_pdet) / n_in
    return ei / math.log(2.0), term1 / math.log(2.0), log_pdet / math.log(2.0)


def measure(model, width, n_samples, seed, device, layer_sigmas=None, residual_sigma=None):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    rows = []
    for idx, layer in enumerate(model.layers, start=1):
        n_in = layer.in_features
        samples = torch.rand((n_samples, n_in), generator=gen, dtype=torch.float64,
                             device=device) * width
        W = layer.weight.detach()
        bias = layer.bias.detach() if layer.bias is not None else None
        name = f"L{idx}"
        sig = layer_sigmas[name] if layer_sigmas else residual_sigma
        ei, t1, lp = layer_ei(W, bias, samples, sig, width, model.act)
        rows.append({"layer": name, "n_in": n_in, "ei": ei, "term1": t1, "logpdet": lp})
    return rows


def run(act, width, epochs, lr=0.01, record_every=200, ei_samples=2000):
    torch.manual_seed(42)
    np.random.seed(42)
    xtr, ytr, xte, yte, _, _ = load_iris_tensors(42)
    xtr, ytr, xte, yte = xtr.to(DEV), ytr.to(DEV), xte.to(DEV), yte.to(DEV)
    model = Net(act=act).to(device=DEV, dtype=torch.float64)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu"); g.manual_seed(42)
    loader = DataLoader(TensorDataset(xtr.cpu(), ytr.cpu()), batch_size=10, shuffle=True, generator=g)

    hist = []
    for epoch in range(1, epochs + 1):
        model.train()
        names = [f"L{i}" for i in range(1, len(model.layers) + 1)]
        sacc = GradientSigmaAccumulator(layer_names=names)
        racc = ResidualSigmaAccumulator()
        tr_sum, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            out, acts = model.fwd(xb)
            loss = loss_fn(out, yb)
            racc.update(out, yb)
            loss.backward()
            sacc.update({f"L{i}": a.grad for i, a in enumerate(acts, start=1)})
            opt.step()
            tr_sum += float(loss.item()) * xb.shape[0]
            seen += xb.shape[0]
        if epoch == 1 or epoch % record_every == 0 or epoch == epochs:
            ls = sacc.sigmas()
            rows = measure(model, width, ei_samples, 1234 + epoch, DEV, layer_sigmas=ls)
            orows = measure(model, width, ei_samples, 1234 + epoch, DEV,
                            residual_sigma=racc.sigma())
            overall = sum(r["ei"] for r in orows)
            entry = {"epoch": epoch, "train_mse": tr_sum / seen,
                     "test_mse": float(loss_fn(model(xte), yte).item()),
                     "overall_sigma": racc.sigma(), "overall_ei": overall}
            for r in rows:
                entry[r["layer"]] = r["ei"]
                entry[r["layer"] + "_pdet"] = r["logpdet"]
            hist.append(entry)
            print(f"  [{act} w={width:g}] ep {epoch:>5d} MSE={entry['train_mse']:.5f} "
                  f"sigma={entry['overall_sigma']:.4f} OverallEI={overall:+.2f} "
                  f"(L1={entry['L1']:+.2f} L2={entry['L2']:+.2f} L3={entry['L3']:+.2f})",
                  flush=True)
    return hist


def main():
    cfg_a = ("sigmoid", 3.0, 10000)
    cfg_b = ("tanh", 2.0, 10000)
    print("=== A. sigmoid + w=3 + Adam ===", flush=True)
    ha = run(*cfg_a)
    print("\n=== B. tanh + w=2 + Adam ===", flush=True)
    hb = run(*cfg_b)

    outdir = PACKAGE / "results" / "updates"
    with (outdir / "overall_positive.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "epoch", "train_mse", "test_mse", "overall_sigma", "overall_ei",
                    "L1", "L2", "L3"])
        for tag, hist in (("sigmoid_w3_adam", ha), ("tanh_w2_adam", hb)):
            for r in hist:
                w.writerow([tag, r["epoch"], f"{r['train_mse']:.6f}", f"{r['test_mse']:.6f}",
                            f"{r['overall_sigma']:.6f}", f"{r['overall_ei']:.4f}",
                            f"{r['L1']:.4f}", f"{r['L2']:.4f}", f"{r['L3']:.4f}"])

    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 160, "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "lines.linewidth": 2.0, "legend.fontsize": 9,
    })
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.9))
    for ax, hist, title in (
        (axes[0], ha, "A: sigmoid, w=3, Adam"),
        (axes[1], hb, "B: tanh, w=2, Adam"),
    ):
        ep = [r["epoch"] for r in hist]
        for layer in ("L1", "L2", "L3"):
            ax.plot(ep, [r[layer] for r in hist], color=LAYER_COLORS[layer], label=layer)
        ax.plot(ep, [r["overall_ei"] for r in hist], color=TOTAL_COLOR, lw=2.8,
                ls="--", label="Overall EI")
        ax.axhline(0.0, color="#888888", ls=":", lw=1.0)
        ax.set_xlabel("Epoch"); ax.set_ylabel("EI (bits)"); ax.set_title(title)
        ax.legend(loc="best")

    ax = axes[2]
    for hist, label in ((ha, "A sigmoid w=3"), (hb, "B tanh w=2")):
        ax.plot([r["epoch"] for r in hist], [r["train_mse"] for r in hist], label=f"{label} train MSE")
        ax.plot([r["epoch"] for r in hist], [r["overall_ei"] for r in hist], ls="--",
                label=f"{label} Overall EI")
    ax.axhline(0.0, color="#888888", ls=":", lw=1.0)
    ax.set_xlabel("Epoch"); ax.set_title("MSE vs Overall EI"); ax.legend(loc="best", fontsize=8)

    fig.suptitle("Continuous Overall EI: can it go positive? (sigmoid w=3 vs tanh w=2, Adam)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outdir / "overall_positive.png")
    plt.close(fig)

    for tag, hist in (("A sigmoid w=3", ha), ("B tanh w=2", hb)):
        r = hist[-1]
        print(f"\n{tag}: final MSE={r['train_mse']:.5f} sigma={r['overall_sigma']:.4f} "
              f"Overall EI={r['overall_ei']:+.2f} (L1 {r['L1']:+.2f} L2 {r['L2']:+.2f} L3 {r['L3']:+.2f})")
        first_pos = next((x for x in hist if x["overall_ei"] > 0), None)
        print("  越过 0 的 epoch:", first_pos["epoch"] if first_pos else "未越过")


if __name__ == "__main__":
    main()
