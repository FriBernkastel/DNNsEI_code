"""诊断脚本：IRIS 上 MSE 为什么是 0.118，哪些改动能把它压到 0.05 以下。

不修改原程序。所有实验都用同一份 train/test 划分（seed=42）。

【注意】本脚本用 nn.Sequential 手工搭网络，初始化的抽取顺序与 IrisSigmoidNet 不同，
baseline 会停在"几乎没学会"的状态（MSE≈0.21、准确率 0.40），因此结论仅作参考。
正式结论请看 diagnose_mse2.py（严格使用 IrisSigmoidNet）。
"""

import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from eicore import IrisSigmoidNet, load_iris_tensors  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XT, YT, XE, YE, MINS, WIDTHS = load_iris_tensors(42)
XT, YT, XE, YE = XT.to(DEV), YT.to(DEV), XE.to(DEV), YE.to(DEV)


def build(dims=(4, 5, 5, 3), bias=False):
    layers, prev = [], dims[0]
    for d in dims[1:]:
        layers += [nn.Linear(prev, d, bias=bias), nn.Sigmoid()]
        prev = d
    return nn.Sequential(*layers).to(device=DEV, dtype=torch.float64)


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        out = model(x)
        mse = float(nn.MSELoss()(out, y).item())
        per_sample = ((out - y) ** 2).mean(dim=1).cpu().numpy()
        acc = float((out.argmax(dim=1) == y.argmax(dim=1)).float().mean().item())
    return mse, acc, per_sample


def train(model, loss_fn, epochs=4000, lr=0.01, batch=10, opt_name="sgd", normalize=False):
    x_train, x_test = XT, XE
    if normalize:
        x_train = (XT - MINS.to(DEV)) / WIDTHS.to(DEV)
        x_test = (XE - MINS.to(DEV)) / WIDTHS.to(DEV)
    if opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    loader = DataLoader(
        TensorDataset(x_train.cpu(), YT.cpu()), batch_size=batch, shuffle=True, generator=g
    )
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    tr, tr_acc, per = evaluate(model, x_train, YT)
    te, te_acc, _ = evaluate(model, x_test, YE)
    return tr, te, tr_acc, te_acc, per


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=== 0. baseline: 4-5-5-3, sigmoid, no bias, MSE, SGD lr=0.01, 4000 ep ===")
    m = build()
    tr, te, tra, tea, per = train(m, nn.MSELoss(), epochs=4000)
    print(f"train MSE={tr:.4f}  test MSE={te:.4f}  train acc={tra:.3f}  test acc={tea:.3f}")
    p = np.sort(per)
    print("per-sample MSE: min %.4f  p25 %.4f  median %.4f  p75 %.4f  p90 %.4f  max %.4f"
          % (p[0], np.percentile(p, 25), np.median(p), np.percentile(p, 75),
             np.percentile(p, 90), p[-1]))
    top = np.sort(per)[::-1]
    print("top-10 worst samples carry %.1f%% of the total MSE; bottom 50%% carry %.1f%%"
          % (100 * top[:10].sum() / per.sum(), 100 * np.sort(per)[:60].sum() / per.sum()))

    print("\n=== 1-6: knobs that could lower MSE ===")
    rows = []

    torch.manual_seed(42)
    m = build()
    tr, te, tra, tea, _ = train(m, nn.MSELoss(), epochs=20000)
    rows.append(("4000 -> 20000 epochs", tr, te, tea))

    torch.manual_seed(42)
    m = build()
    tr, te, tra, tea, _ = train(m, nn.MSELoss(), epochs=4000, normalize=True)
    rows.append(("min-max 归一化输入", tr, te, tea))

    torch.manual_seed(42)
    m = build()
    tr, te, tra, tea, _ = train(m, nn.MSELoss(), epochs=4000, opt_name="adam", lr=0.01)
    rows.append(("Adam lr=0.01", tr, te, tea))

    torch.manual_seed(42)
    m = build(bias=True)
    tr, te, tra, tea, _ = train(m, nn.MSELoss(), epochs=4000)
    rows.append(("加 bias", tr, te, tea))

    torch.manual_seed(42)
    m = build(dims=(4, 16, 16, 3))
    tr, te, tra, tea, _ = train(m, nn.MSELoss(), epochs=4000)
    rows.append(("加宽 4-16-16-3", tr, te, tea))

    torch.manual_seed(42)
    m = build()
    tr, te, tra, tea, _ = train(m, nn.MSELoss(), epochs=4000, normalize=True,
                                opt_name="adam", lr=0.01)
    rows.append(("归一化 + Adam", tr, te, tea))

    print(f"{'配置':28s} {'train MSE':>10s} {'test MSE':>9s} {'test acc':>9s}")
    for name, tr, te, tea in rows:
        print(f"{name:28s} {tr:>10.4f} {te:>9.4f} {tea:>9.3f}")

    print("\n=== 7: 交叉熵 + softmax（不再是 MSE 损失，只把 MSE 当指标） ===")
    torch.manual_seed(42)
    net = nn.Sequential(
        nn.Linear(4, 5), nn.Sigmoid(), nn.Linear(5, 5), nn.Sigmoid(),
        nn.Linear(5, 3), nn.Softmax(dim=1),
    ).to(device=DEV, dtype=torch.float64)
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    g = torch.Generator(device="cpu"); g.manual_seed(42)
    loader = DataLoader(TensorDataset(XT.cpu(), YT.cpu()), batch_size=10, shuffle=True, generator=g)
    ce = nn.CrossEntropyLoss()
    for _ in range(4000):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            ce(net(xb), yb.argmax(dim=1)).backward()
            opt.step()
    tr, tra, _ = evaluate(net, XT, YT)
    te, tea, _ = evaluate(net, XE, YE)
    print(f"train MSE={tr:.4f}  test MSE={te:.4f}  train acc={tra:.3f}  test acc={tea:.3f}")


if __name__ == "__main__":
    main()
