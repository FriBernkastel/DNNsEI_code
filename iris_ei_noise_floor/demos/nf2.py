import csv
rows = list(csv.DictReader(open("results/latest/iris_ei_noise_floor/noise_floor_history.csv", encoding="utf-8")))
print("L2 在两个尖谷附近：")
for r in rows:
    if r["layer"] == "L2" and int(float(r["epoch"])) in (170, 180, 185, 190, 200, 580, 590, 595, 600, 610):
        print(f"  ep={int(float(r['epoch'])):4d}  old={float(r['ei_bits']):7.4f}  new={float(r['ei_bits_floored']):7.4f}  floor_gain={float(r['floor_gain_bits']):6.4f}")
print("\nfloor_gain>0 的 epoch 数 / 总数：")
for L in ("L1","L2","L3"):
    sub = [r for r in rows if r["layer"] == L]
    act = sum(1 for r in sub if float(r["floor_gain_bits"]) > 1e-12)
    print(f"  {L}: {act}/{len(sub)}")
print("\n最终（ep=4000）逐层：")
for r in rows:
    if int(float(r["epoch"])) == 4000:
        print(f"  {r['layer']}: sigma={float(r['sigma']):.5f}  old={float(r['ei_bits']):7.4f}  new={float(r['ei_bits_floored']):7.4f}  "
              f"floor_gain={float(r['floor_gain_bits']):6.4f}  tax_fix={float(r['tax_fix_bits']):7.4f}")
