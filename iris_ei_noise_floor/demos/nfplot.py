import csv, sys
import sys as _sys
from pathlib import Path as _Path
_PKG = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_PKG))

from pathlib import Path
sys.path.insert(0, ".")
from experiments.iris_ei_noise_floor import plot_comparison
from eicore.report import publish_latest
rows = list(csv.DictReader(open("results/latest/iris_ei_noise_floor/noise_floor_history.csv", encoding="utf-8")))
hist = []
for r in rows:
    d = {}
    for k, v in r.items():
        d[k] = int(float(v)) if k in ("epoch", "n_in", "n_out") else (v if k == "layer" else float(v))
    hist.append(d)
run = sorted(Path("results").glob("iris_ei_noise_floor_*"))[-1]
print(plot_comparison(hist, run))
print("mirrored:", publish_latest(run))
