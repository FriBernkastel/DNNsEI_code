import csv, numpy as np
import sys as _sys
from pathlib import Path as _Path
_PKG = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_PKG))

from pathlib import Path
new = Path("results/latest/iris_transport_map_ei/tm_ei_history.csv")
old = Path("results/iris_transport_map_ei_20261110_150843/tm_ei_history.csv")
def load(p):
    d = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        d[(int(float(r["epoch"])), r["layer"])] = r
    return d
a, b = load(old), load(new)
print("epoch  layer  real(recompute)  real(actual pairs)  diff   |  noiseless  | n_pairs")
for ep in (1, 1000, 2500, 4000):
    for L in ("L1","L2","L3"):
        o, n = a[(ep,L)], b[(ep,L)]
        print(f"{ep:5d}  {L}   {float(o['tm_ei_real_bits']):8.4f}         {float(n['tm_ei_real_bits']):8.4f}      "
              f"{float(n['tm_ei_real_bits'])-float(o['tm_ei_real_bits']):+7.4f}  |  {float(n['tm_ei_actual_bits']):8.4f}  | {n['n_pairs']}")
    print()
