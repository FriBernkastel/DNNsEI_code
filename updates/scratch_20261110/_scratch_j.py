import sys, numpy as np
sys.path.insert(0, ".")
from eicore import load_iris_tensors
import torch
LN2 = np.log(2)
d = np.load(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "transport_map_EI" / "data" / "iris_weights_seed42.npz"))
x_tr, *_ = load_iris_tensors(42); x = x_tr.numpy()
rng = np.random.default_rng(0); u = rng.random((20000, 4))
def elogj(samples, W):
    z = samples @ W.T; s = 1/(1+np.exp(-z))
    J = s[:, :, None] * (1 - s)[:, :, None] * W[None, :, :]   # (n, n_out, n_in)
    sv = np.linalg.svd(J, compute_uv=False)
    return np.mean(np.sum(np.log(np.clip(sv, 1e-12, None)), axis=1))
W1 = d["W1"]
ju, jr = elogj(u, W1), elogj(x, W1)
print(f"E log|det J|  uniform = {ju:+.3f} nats   real = {jr:+.3f} nats   diff = {jr-ju:+.3f}")
hx = 0.5*(4*np.log(2*np.pi*np.e) + np.linalg.slogdet(np.cov(x, rowvar=False))[1])
print(f"h(real x) = {hx:+.3f} nats   h(uniform) = 0")
print(f"预测 EI_real - EI_uniform = {(hx + jr - ju)/4/LN2:+.3f} bit ; 实测 = {3.3585 - 4.8122:+.3f} bit")
print(f"\n符号饱和检查: 真实输入 |W1x|>4 的比例 = {(np.abs(x@W1.T)>4).mean()*100:.0f}%  "
      f"vs uniform {(np.abs(u@W1.T)>4).mean()*100:.0f}%")
