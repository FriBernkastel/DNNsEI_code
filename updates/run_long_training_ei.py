"""训练 20000 轮（其余同实验一），只在 epoch 1 和最后一轮测 EI，看 MSE 压下去后 Overall EI 变成多少。"""

import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import experiments.iris_continuous_ei as ice  # noqa: E402

if __name__ == "__main__":
    torch.manual_seed(42)
    config = ice.Config(epochs=20000, record_every=20000, ei_samples=2000)
    history = ice.train_and_measure(config)
    print("\n=== summary ===")
    for epoch in (1, 20000):
        rows = [r for r in history if int(r["epoch"]) == epoch]
        if not rows:
            continue
        print(f"epoch {epoch}: train_mse={float(rows[0]['train_mse']):.4f} "
              f"test_mse={float(rows[0]['test_mse']):.4f} "
              f"overall_sigma={float(rows[0]['overall_sigma']):.4f} "
              f"OverallEI={float(rows[0]['overall_ei_bits']):+.3f}")
        for r in rows:
            print(f"   {r['layer']}: EI={float(r['ei_bits']):+.3f} "
                  f"Sens={float(r['sensitivity_bits']):+.3f} "
                  f"term1={float(r['term1_bits']):+.3f} "
                  f"logpdet_bits={float(r['log_pdet_mean_nats'])/0.693147:.3f}")
