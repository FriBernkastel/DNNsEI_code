import sys, numpy as np
sys.path.insert(0, ".")
from eicore import load_iris_tensors
sys.path.insert(0, r"D:\PythonProject")
LN2 = np.log(2)
d = np.load(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "transport_map_EI" / "data" / "iris_weights_seed42.npz"))
x_tr, _, x_te, _, fmin, fwid = load_iris_tensors(42)
W1, s1 = d["W1"], float(d["sigma1"])
def elogj(samples, W):
    z = samples @ W.T; s = 1/(1+np.exp(-z)); sp = s*(1-s)
    J = sp[:,:,None] * W[None,:,:]
    sv = np.linalg.svd(J, compute_uv=False)
    return np.mean(np.sum(np.log(np.clip(sv,1e-12,None)),1))
def ei(S, W, s, Hx):
    return (Hx + elogj(S, W) - 0.5*W.shape[0]*np.log(2*np.pi*np.e*s*s))/W.shape[1]/LN2
rng = np.random.default_rng(0)
mins, wid = fmin.numpy(), fwid.numpy()
u01 = rng.random((20000, 4)); ubox = mins + rng.random((20000, 4))*wid
print(f"特征 mins={mins}  widths={wid}")
for tag, S, Hx in (("U(0,1)^4（仓库现状）", u01, 0.0),
                   ("U(特征范围箱)", ubox, float(np.sum(np.log(wid))))):
    print(f"  {tag:20s} H(X)={Hx:5.2f} nats  E log|J|={elogj(S,W1):7.2f}  -> EI = {ei(S,W1,s1,Hx):6.3f} bit/维")
print(f"  实际样本对（无噪声高斯 MI）               -> EI = 2.728 bit/维")
