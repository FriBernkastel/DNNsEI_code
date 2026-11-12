import csv, sys
import sys as _sys
from pathlib import Path as _Path
_PKG = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_PKG))

from pathlib import Path
sys.path.insert(0, ".")
from experiments.iris_transport_map_ei import plot_all
rows = list(csv.DictReader(open("results/latest/iris_transport_map_ei/tm_ei_history.csv", encoding="utf-8")))
hist = [{"epoch": int(float(r["epoch"])), "layer": r["layer"], "n_in": int(r["n_in"]),
         "n_out": int(r["n_out"]), "sigma": float(r["sigma"]),
         "cont_ei_bits": float(r["cont_ei_bits"]), "tm_ei_bits": float(r["tm_ei_bits"]),
         "tm_ei_real_bits": float(r["tm_ei_real_bits"]), "ratio": float(r["ratio"]),
         "train_mse": float(r["train_mse"]), "test_mse": float(r["test_mse"])} for r in rows]
run = sorted(Path("results").glob("iris_transport_map_ei_*"))[-1]
paths = plot_all(hist, run)
for k, p in paths.items():
    print(k, "->", p)
