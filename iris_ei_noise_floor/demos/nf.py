import csv
rows = list(csv.DictReader(open("results/latest/iris_ei_noise_floor/noise_floor_history.csv", encoding="utf-8")))
sel = [r for r in rows if int(float(r["epoch"])) in (1, 1000, 2500, 4000)]
print(f"{'ep':>5} {'layer':>5} {'sigma':>9} {'old':>9} {'new':>9} {'floor_gain':>10} {'tax_fix':>9}")
for r in sel:
    print(f"{int(float(r['epoch'])):5d} {r['layer']:>5} {float(r['sigma']):9.5f} {float(r['ei_bits']):9.4f} "
          f"{float(r['ei_bits_floored']):9.4f} {float(r['floor_gain_bits']):10.4f} {float(r['tax_fix_bits']):9.4f}")
print()
for ep in (1, 2500, 4000):
    sub = [r for r in rows if int(float(r["epoch"])) == ep]
    o = sum(float(r["ei_bits"]) for r in sub); n = sum(float(r["ei_bits_floored"]) for r in sub)
    ro = float(sub[0]["overall_ei_bits"]); rn = float(sub[0]["overall_ei_bits_floored"]); rs = float(sub[0]["overall_sigma"])
    print(f"ep={ep:4d}  sum(layer sigma) old={o:+8.3f} new={n:+8.3f} | residual sigma={rs:.4f} old={ro:+8.3f} new={rn:+8.3f}")
