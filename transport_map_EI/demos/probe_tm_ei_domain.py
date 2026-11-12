"""Probe: is TM-EI a discrete-path or continuous-path estimator?"""
from __future__ import annotations

import sys

import numpy as np

TM_DIR = __import__("pathlib").Path(__file__).resolve().parents[1]
PACKAGE = TM_DIR.parent
sys.path.insert(0, str(TM_DIR))
sys.path.insert(0, str(PACKAGE))

from transport_map_ei import (  # noqa: E402
    estimate_transport_map_mutual_information as tm_mi,
)

LN2 = np.log(2.0)


def bits(x: float) -> float:
    return x / LN2


def show(tag, mi_nats, truth_bits, note=""):
    print(f"  {tag:46s} TM-EI={bits(mi_nats):8.4f} bit  true={truth_bits:6.3f} bit  {note}")


rng = np.random.default_rng(0)

print("== A. 离散对象：8 状态均匀 cause，effect 为恒等映射（真值 log2 8 = 3 bit）==")
n = 20000
c_idx = rng.integers(0, 8, size=n)
onehot = np.eye(8)[c_idx]
show("one-hot cause -> effect(index)   [8 维]", tm_mi(onehot, c_idx[:, None])["mi"], 3.0)
show("one-hot cause -> effect(one-hot)", tm_mi(onehot, onehot)["mi"], 3.0)
show("整数编码 cause -> effect(index)", tm_mi(c_idx[:, None].astype(float), c_idx[:, None].astype(float))["mi"], 3.0)

print("\n== B. 离散对象：2 状态（真值 1 bit）==")
c2 = rng.integers(0, 2, size=n).astype(float)
show("0/1 cause -> 0/1 effect", tm_mi(c2[:, None], c2[:, None])["mi"], 1.0)

print("\n== C. 离散 + 真实噪声（把离散映射抹平，真值仍是 1 bit）==")
ce = c2 + 0.3 * rng.normal(size=n)
show("0/1 cause -> 0/1 effect (噪声版)", tm_mi(ce[:, None], c2[:, None])["mi"], 1.0)

print("\n== D. 连续对象：连续线性/非线性，TM-EI 有解析对照 ==")
x = rng.uniform(-1.0, 1.0, size=n)
show("高斯线性 rho=0.8", tm_mi(x[:, None], (0.8 * x + 0.6 * rng.normal(size=n))[:, None])["mi"], 0.736965)
show("连续 tanh 映射 x -> tanh(2x)", tm_mi(x[:, None], np.tanh(2 * x)[:, None])["mi"], float("nan"), "(确定性，无真值)")

print("\n== E. 确定性关系下 jitter 的敏感性（x -> 2x 精确线性，理论 MI 发散）==")
xr = rng.normal(size=n)
for jit in (1e-6, 1e-3, 1e-1):
    v = tm_mi(xr[:, None], (2.0 * xr)[:, None], jitter=jit)["mi"]
    print(f"  jitter={jit:8.1e}  TM-EI={bits(v):10.4f} bit")

print("\n== F. 对离散输入的 jitter 敏感性（one-hot，真值 3 bit）==")
for jit in (1e-6, 1e-3, 1e-1):
    v = tm_mi(onehot, onehot, jitter=jit)["mi"]
    print(f"  jitter={jit:8.1e}  TM-EI={bits(v):10.4f} bit")
