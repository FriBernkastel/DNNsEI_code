import sys, numpy as np
sys.path.insert(0, ".")
from eicore import load_iris_tensors
LN2 = np.log(2)
x_tr, *_ = load_iris_tensors(42); x = x_tr.numpy()
def h_gauss(a):
    a = np.atleast_2d(np.asarray(a, float))
    d = a.shape[1]
    S = np.cov(a, rowvar=False) if d > 1 else np.array([[a.var()]])
    return 0.5*(d*np.log(2*np.pi*np.e) + np.linalg.slogdet(S)[1])
print(f"真实输入 x_train: shape={x.shape}, 每维范围={x.max(0)-x.min(0)}")
print(f"h_gauss(real x) = {h_gauss(x):.3f} nats = {h_gauss(x)/LN2:.3f} bit")
print(f"h(U(0,1)^4)     = 0.000 nats = 0.000 bit")
print(f"预测 L1 gap = {0 - h_gauss(x)/LN2:.3f} bit ; 实测 gap = {4.8122 - 3.3585:.3f} bit (epoch 1)")
