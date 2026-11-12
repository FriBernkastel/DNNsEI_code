import csv, numpy as np
import sys as _sys
from pathlib import Path as _Path
_PKG = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_PKG))

from scipy.stats import spearmanr
rows = list(csv.DictReader(open(r"results/latest/iris_transport_map_ei/tm_ei_history.csv", encoding="utf-8")))
for name in ("L1","L2","L3"):
    sub = [r for r in rows if r["layer"]==name]
    c = np.array([float(r["cont_ei_bits"]) for r in sub]); t = np.array([float(r["tm_ei_bits"]) for r in sub])
    tr = np.array([float(r["tm_ei_real_bits"]) for r in sub]); e = np.array([int(float(r["epoch"])) for r in sub])
    m = e >= 200
    print(f"{name}: Spearman(TM,cont) full={spearmanr(t,c).statistic:+.3f}  epoch>=200 {spearmanr(t[m],c[m]).statistic:+.3f}   TM_real vs cont {spearmanr(tr,c).statistic:+.3f}")
