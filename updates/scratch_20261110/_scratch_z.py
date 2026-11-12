import sys, numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, r"D:\PythonProject")
from eicore import load_iris_tensors
d = np.load(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "transport_map_EI" / "data" / "iris_weights_seed42.npz"))
x_tr, *_ = load_iris_tensors(42); x = x_tr.numpy()
rng = np.random.default_rng(0); u = rng.random((2000, 4))
for i in (1, 2, 3):
    W = d[f"W{i}"]
    zr = x @ W.T; zu = u @ W.T
    print(f"L{i}: |W|max={np.abs(W).max():.3f} | 真实输入 z=[{zr.min():+.1f},{zr.max():+.1f}] "
          f"| U(0,1) z=[{zu.min():+.1f},{zu.max():+.1f}]")
print(f"\n真实输入范围: {x.min(axis=0)} ~ {x.max(axis=0)}")
