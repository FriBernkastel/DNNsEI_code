import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
TM_DIR = __import__("pathlib").Path(__file__).resolve().parents[1]
PACKAGE = TM_DIR.parent
sys.path.insert(0, str(TM_DIR))
sys.path.insert(0, str(PACKAGE))
from transport_map_ei import estimate_transport_map_mutual_information as tm

LN2 = np.log(2.0)
d = np.load(str(TM_DIR / "data" / "iris_weights_seed42.npz"))
sig = lambda z: 1/(1+np.exp(-z))
REF = {"L1": (4, 5.695192766368154, 2.514700558146253),
       "L2": (5, 2.189728377158652, 2.1044596358131984),
       "L3": (5, 0.9215081678801678, 2.016371029517777)}
rng = np.random.default_rng(42)
n = 20000
scales = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
curves, tms = {}, {}
for i in (1, 2, 3):
    W, s = d[f"W{i}"], float(d[f"sigma{i}"]); n_in = W.shape[1]
    c = rng.random((n, n_in))
    cur = []
    for k in scales:
        e = sig(c @ W.T) + k*s*rng.normal(size=(n, W.shape[0]))
        cur.append(tm(c, e)["mi"]/LN2/n_in)
    curves[f"L{i}"] = cur
    e = sig(c @ W.T) + s*rng.normal(size=(n, W.shape[0]))
    tms[f"L{i}"] = tm(c, e)["mi"]/LN2/n_in

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
layers = ["L1", "L2", "L3"]
x = np.arange(3); w = 0.26
cont = [REF[k][1] for k in layers]; disc = [REF[k][2] for k in layers]
tmv = [tms[k] for k in layers]
axes[0].bar(x-w, cont, w, label="Continuous (Jacobian+sigma)", color="#4c78a8")
axes[0].bar(x, disc, w, label="Discrete (bins=32)", color="#e45756")
axes[0].bar(x+w, tmv, w, label="TM-EI (Gaussian fit)", color="#54a24b")
axes[0].set_xticks(x); axes[0].set_xticklabels(layers)
axes[0].set_ylabel("EI (bits, per input dim)"); axes[0].set_title("IRIS epoch-4000 weights: three EI routes")
axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=.3)
for xi, v in zip(x+w, tmv):
    axes[0].text(xi, v+0.08, f"{v:.2f}", ha="center", fontsize=8)

for k, cval in zip(layers, cont):
    s = REF[k][0]
    axes[1].plot(scales, curves[k], "o-", label=f"{k} TM-EI")
    axes[1].axhline(cval, ls="--", lw=1, color=axes[1].lines[-1].get_color(), alpha=.7)
axes[1].set_xscale("log"); axes[1].set_xlabel(r"noise scale  $\sigma/\sigma_{layer}$")
axes[1].set_ylabel("EI (bits, per input dim)")
axes[1].set_title("TM-EI vs noise level (dashed = continuous EI at $\\sigma_{layer}$)")
axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
fig.tight_layout()
out = str(PACKAGE / "results" / "iris_ei_three_routes.png")
fig.savefig(out, dpi=150)
print("wrote", out)
