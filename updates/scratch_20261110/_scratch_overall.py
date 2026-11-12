import sys, numpy as np, torch
sys.path.insert(0, ".")
from eicore import IrisSigmoidNet, compute_model_layer_ei
d = np.load(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "transport_map_EI" / "data" / "iris_weights_seed42.npz"))
m = IrisSigmoidNet().to(dtype=torch.float64)
with torch.no_grad():
    for i, L in enumerate(m.layers, start=1):
        L.weight.copy_(torch.tensor(d[f"W{i}"]))
res = float(d["residual_sigma"])
print(f"residual sigma = {res:.6f}")
rows = compute_model_layer_ei(m, sigma=res, n_samples=2000, device=torch.device("cpu"),
                              seed=42+4000, first_layer_mins=None, first_layer_widths=None)
tot_old = tot_new = 0.0
print(f"{'layer':>5} {'n_in':>4} {'n_out':>5} {'log_pdet':>9} {'old':>9} {'new':>7} {'floor_gain':>10} {'tax_fix':>8}")
for r in rows:
    tot_old += r["ei_bits"]; tot_new += r["ei_bits_floored"]
    print(f"{r['layer']:>5} {r['n_in']:4d} {r['n_out']:5d} {r['log_pdet_mean_nats']:9.3f} "
          f"{r['ei_bits']:9.4f} {r['ei_bits_floored']:7.4f} {r['floor_gain_bits']:10.4f} {r['tax_fix_bits']:8.4f}")
print(f"总计: old={tot_old:+.4f} bit   new={tot_new:+.4f} bit")
