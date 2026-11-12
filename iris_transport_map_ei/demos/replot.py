import csv, sys, numpy as np
import sys as _sys
from pathlib import Path as _Path
_PKG = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_PKG))

from pathlib import Path
sys.path.insert(0, ".")
from experiments.iris_transport_map_ei import plot_all
rows = list(csv.DictReader(open("results/latest/iris_transport_map_ei/tm_ei_history.csv", encoding="utf-8")))
hist = []
for r in rows:
    hist.append({"epoch": int(float(r["epoch"])), "layer": r["layer"], "n_in": int(r["n_in"]),
                 "n_out": int(r["n_out"]), "sigma": float(r["sigma"]),
                 "n_pairs": int(float(r["n_pairs"])),
                 "cont_ei_bits": float(r["cont_ei_bits"]), "tm_ei_bits": float(r["tm_ei_bits"]),
                 "tm_ei_real_bits": float(r["tm_ei_real_bits"]),
                 "tm_ei_actual_bits": float(r["tm_ei_actual_bits"]),
                 "ratio": float(r["ratio"]),
                 "train_mse": float(r["train_mse"]), "test_mse": float(r["test_mse"])})
run = Path("results/latest/iris_transport_map_ei")
out = Path("results/updates/tm_ei_replot")
out.mkdir(parents=True, exist_ok=True)
p = plot_all(hist, out)
print("wrote", p)

print("\n== 趋势诊断（401 个记录点）==")
for name in ("L1", "L2", "L3"):
    sub = [r for r in hist if r["layer"] == name]
    s = np.array([r["sigma"] for r in sub]); c = np.array([r["cont_ei_bits"] for r in sub])
    t = np.array([r["tm_ei_bits"] for r in sub]); tr = np.array([r["tm_ei_real_bits"] for r in sub])
    x = -np.log2(s)
    print(f"  {name}: corr(cont, -log2 σ)={np.corrcoef(c,x)[0,1]:+.3f}  corr(TM, -log2 σ)={np.corrcoef(t,x)[0,1]:+.3f}"
          f"  corr(TM, cont)={np.corrcoef(t,c)[0,1]:+.3f}  corr(TM_real, cont)={np.corrcoef(tr,c)[0,1]:+.3f}")
    print(f"      σ: {s[0]:.2e} -> {s[-1]:.2e} (x{s[-1]/s[0]:.1f})   TM/cont: {t[0]/c[0]:.2f} -> {t[-1]/c[-1]:.2f}"
          f"   mean ratio={np.mean(t/c):.2f}")
