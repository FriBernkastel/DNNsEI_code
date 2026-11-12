"""诊断（严格用 IrisSigmoidNet）：MSE 的构成，以及哪些训练配置能压到 0.05 以下。"""

import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from eicore import IrisSigmoidNet, load_iris_tensors  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XT, YT, XE, YE, _, _ = load_iris_tensors(42)
XT, YT, XE, YE = XT.to(DEV), YT.to(DEV), XE.to(DEV), YE.to(DEV)


def run(epochs=4000, lr=0.01, opt_name="sgd", batch=10):
    torch.manual_seed(42)
    np.random.seed(42)
    model = IrisSigmoidNet().to(device=DEV, dtype=torch.float64)
    loss_fn = nn.MSELoss()
    opt = (torch.optim.Adam if opt_name == "adam" else torch.optim.SGD)(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu"); g.manual_seed(42)
    loader = DataLoader(TensorDataset(XT.cpu(), YT.cpu()), batch_size=batch,
                        shuffle=True, generator=g)
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        if ep in (1000, 2000, 4000, 8000, 20000) and ep <= epochs:
            with torch.no_grad():
                s = float(loss_fn(model(XT), YT).item())
            print(f"    ep {ep:>5d}: train MSE {s:.4f}", flush=True)
    with torch.no_grad():
        out_tr, out_te = model(XT), model(XE)
        tr_mse = float(loss_fn(out_tr, YT).item())
        te_mse = float(loss_fn(out_te, YE).item())
        acc_tr = float((out_tr.argmax(1) == YT.argmax(1)).float().mean().item())
        acc_te = float((out_te.argmax(1) == YE.argmax(1)).float().mean().item())
        per = ((out_tr - YT) ** 2).mean(dim=1).cpu().numpy()
        conf = out_tr.max(dim=1).values.cpu().numpy()
    return tr_mse, te_mse, acc_tr, acc_te, per, conf


def stats(name, res):
    tr, te, atr, ate, per, conf = res
    p = np.sort(per)
    print(f"{name:26s} train {tr:.4f}  test {te:.4f}  acc {atr:.3f}/{ate:.3f} "
          f"| sampleMSE med {np.median(p):.4f} p90 {np.percentile(p,90):.4f} max {p[-1]:.4f}"
          f" | 平均置信度 {conf.mean():.3f}", flush=True)


if __name__ == "__main__":
    print("=== A. 原配置（4000 ep, SGD 0.01, 无 bias, 原始输入）===", flush=True)
    a = run(4000)
    stats("baseline", a)

    print("\n=== B. 同一配置训练到 20000 ep ===", flush=True)
    b = run(20000)
    stats("20000 epochs", b)

    print("\n=== C. Adam lr=0.01, 4000 ep ===", flush=True)
    c = run(4000, opt_name="adam")
    stats("Adam", c)
