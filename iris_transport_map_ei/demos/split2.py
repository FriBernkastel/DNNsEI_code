import csv, sys
import sys as _sys
from pathlib import Path as _Path
_PKG = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_PKG))

from pathlib import Path
sys.path.insert(0, ".")
from experiments.iris_transport_map_ei import plot_all
rows = list(csv.DictReader(open("results/latest/iris_transport_map_ei/tm_ei_history.csv", encoding="utf-8")))
hist = [{k: (int(float(v)) if k in ("epoch","n_in","n_out","n_pairs") else
             (float(v) if k not in ("layer",) else v))
         for k, v in r.items()} for r in rows]
run = sorted(Path("results").glob("iris_transport_map_ei_*"))[-1]
for k, p in plot_all(hist, run).items():
    print(k, "->", p)
from eicore.report import publish_latest
print("mirrored ->", publish_latest(run))
