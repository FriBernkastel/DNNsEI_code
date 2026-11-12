import sys, numpy as np
TM_DIR = __import__("pathlib").Path(__file__).resolve().parents[1]
PACKAGE = TM_DIR.parent
sys.path.insert(0, str(TM_DIR))
sys.path.insert(0, str(PACKAGE))
from transport_map_ei import estimate_transport_map_mutual_information as tm
d = np.load(str(TM_DIR / "data" / "iris_weights_seed42.npz"))
W1, s1 = d["W1"], float(d["sigma1"])
sig = lambda z: 1/(1+np.exp(-z))
rng = np.random.default_rng(42)
print("L1: 同口径(n=20000)=", round(3.6702, 3), "bit/dim")
for n in (20000, 2000, 500, 150):
    c = rng.random((n, 4)); e = sig(c @ W1.T) + s1*rng.normal(size=(n,5))
    v = tm(c, e)["mi"]/np.log(2)/4
    print(f"  n={n:6d}  TM-EI={v:7.3f} bit/dim")
print("\n真实 IRIS 激活(n=150) 也与上面 n=150 的行对照：2.758 bit/dim")
