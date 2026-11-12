import sys, time, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, r"D:\PythonProject\DNNsEI_code_20261029")
from experiments.transport_map_ei import estimate_transport_map_mutual_information as tm
from eicore.discrete import discrete_layer_ei
LN2 = np.log(2.0); sig = lambda z: 1/(1+np.exp(-z))

def truth_1d(w, s):
    c = np.linspace(0,1,4001); m = sig(w*c)
    e = np.linspace(m.min()-8*s, m.max()+8*s, 8001)
    d = np.exp(-0.5*((e[:,None]-m[None,:])/s)**2).mean(1)/(s*np.sqrt(2*np.pi))
    h_e = -np.trapezoid(np.where(d>0, d*np.log(np.where(d>0,d,1)), 0), e)
    return (h_e - 0.5*np.log(2*np.pi*np.e*s*s))/LN2

def cont_1d(w, s):
    c = np.linspace(0,1,4001); j = np.abs(w*sig(w*c)*(1-sig(w*c)))
    return (np.trapezoid(np.log(j), c) - 0.5*np.log(2*np.pi*np.e*s*s))/LN2

rng = np.random.default_rng(0); n = 20000
sigmas = np.geomspace(0.005, 0.3, 12)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, w in zip(axes, (0.5, 4.0)):
    tr = [truth_1d(w,s) for s in sigmas]
    co = [cont_1d(w,s) for s in sigmas]
    tv = []
    for s in sigmas:
        c = rng.random((n,1)); e = sig(w*c) + s*rng.normal(size=(n,1))
        tv.append(tm(c, e)["mi"]/LN2)
    dv = discrete_layer_ei(np.array([[w]]), n, 32, np.random.default_rng(0))["ei_bits"]
    ax.plot(sigmas, tr, "k-", lw=2.2, label="truth (numerical)")
    ax.plot(sigmas, co, "s--", label="continuous (Jacobian+sigma)")
    ax.plot(sigmas, tv, "o-", label="TM-EI")
    ax.axhline(dv, color="#e45756", ls=":", label=f"discrete bins=32 ({dv:.2f})")
    ax.set_xscale("log"); ax.set_xlabel("noise std sigma"); ax.set_ylabel("EI (bit)")
    ax.set_title(f"1D sigmoid layer w={w}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.axhline(0, color="gray", lw=.6)
fig.tight_layout(); p = r"D:\PythonProject\DNNsEI_code_20261029\results\benchmark_1d_sigmoid.png"
fig.savefig(p, dpi=150); print("wrote", p)

print("\n== TM-EI 收敛性（1D, w=4, sigma=0.05, 真值 1.3888）==")
for nn in (500, 2000, 8000, 32000, 128000):
    vals = []
    for rep in range(5):
        c = np.random.default_rng(rep).random((nn,1)); e = sig(4*c) + 0.05*np.random.default_rng(100+rep).normal(size=(nn,1))
        vals.append(tm(c, e)["mi"]/LN2)
    print(f"  n={nn:7d}  mean={np.mean(vals):7.4f}  std={np.std(vals):.4f}")

print("\n== 单次调用耗时（n=20000, d=1）==")
c = rng.random((n,1)); e = sig(4*c) + 0.05*rng.normal(size=(n,1))
t0=time.perf_counter(); tm(c,e); t1=time.perf_counter(); print(f"  TM-EI      {(t1-t0)*1e3:8.2f} ms")
t0=time.perf_counter(); discrete_layer_ei(np.array([[4.0]]), n, 32, np.random.default_rng(0)); t1=time.perf_counter(); print(f"  discrete   {(t1-t0)*1e3:8.2f} ms")
