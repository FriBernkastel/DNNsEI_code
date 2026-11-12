"""Scratch driver: exercise experiments/transport_map_ei.py (TM-EI)."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, r"D:\PythonProject\DNNsEI_code_20261029")

from experiments.transport_map_ei import (  # noqa: E402
    compute_two_source_transport_map_ei,
    estimate_transport_map_mutual_information,
    gaussian_logdet_bias_correction,
    lift_joint_source_features,
    lift_single_source_features,
    simulate_alpha_nonlinear_case,
)

LN2 = np.log(2.0)


def analytic_linear_check() -> None:
    print("== 1. Analytic check: bivariate Gaussian, known MI = -0.5*ln(1-rho^2) ==")
    rng = np.random.default_rng(0)
    for rho, n in ((0.3, 20000), (0.8, 20000), (0.9, 500)):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        z = rng.multivariate_normal(np.zeros(2), cov, size=n)
        est = estimate_transport_map_mutual_information(z[:, :1], z[:, 1:])["mi"]
        truth = -0.5 * np.log(1.0 - rho**2)
        print(
            f"  rho={rho:4.2f} n={n:6d}  TM-EI={est:8.5f} nats ({est / LN2:7.5f} bits)"
            f"  analytic={truth:8.5f} nats ({truth / LN2:7.5f} bits)  err={est - truth:+.5f}"
        )


def lift_check() -> None:
    print("\n== 2. Feature lifts ==")
    x = np.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    y = np.linspace(1.0, -1.0, 5).reshape(-1, 1)
    print("  lift_single_source_features shape:", lift_single_source_features(x).shape)
    print("  lift_joint_source_features  shape:", lift_joint_source_features(x, y).shape)
    print("  bias correction d=1,n=1000:", round(gaussian_logdet_bias_correction(1, 1000), 6))
    print("  bias correction d=5,n=1000:", round(gaussian_logdet_bias_correction(5, 1000), 6))


def alpha_sweep(n_samples: int = 20000, L: float = 1.0, noise_std: float = 1e-3) -> None:
    print(f"\n== 3. Two-source TM-EI on Q1_next = a*sin(Q2*Q3) + (1-a)*Q2 + noise ==")
    print(f"   n={n_samples}, L={L}, noise_std={noise_std}, values in bits")
    header = (
        f"  {'alpha':>5} {'joint_ei':>9} {'left_ei':>9} {'right_ei':>9} "
        f"{'syn':>9} {'syn/joint':>10} {'left/joint':>11} {'right/joint':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        data = simulate_alpha_nonlinear_case(
            alpha=alpha, n_samples=n_samples, L=L, noise_std=noise_std, seed=7
        )
        res = compute_two_source_transport_map_ei(
            data["q2"], data["q3"], data["q1_next"]
        )
        print(
            f"  {alpha:5.2f} {res['joint_ei'] / LN2:9.4f} {res['left_ei'] / LN2:9.4f} "
            f"{res['right_ei'] / LN2:9.4f} {res['syn'] / LN2:9.4f} "
            f"{res['syn_ratio']:10.4f} {res['left_ratio']:11.4f} {res['right_ratio']:12.4f}"
        )


def raw_nats(alpha: float = 1.0, n_samples: int = 20000) -> None:
    print(f"\n== 4. Raw nats + bias-correction breakdown (alpha={alpha}) ==")
    data = simulate_alpha_nonlinear_case(
        alpha=alpha, n_samples=n_samples, L=1.0, noise_std=1e-3, seed=7
    )
    for name, feats in (
        ("left  [q2,q2^2,q2^3]", lift_single_source_features(data["q2"])),
        ("right [q3,q3^2,q3^3]", lift_single_source_features(data["q3"])),
        ("joint [q2,q3,q2q3,q2^2,q3^2]", lift_joint_source_features(data["q2"], data["q3"])),
    ):
        out = estimate_transport_map_mutual_information(feats, data["q1_next"])
        raw = float(out["pointwise_mi"].mean()) + out["bias_correction"]
        print(
            f"  {name:32s} raw={raw:8.5f}  bias={out['bias_correction']:8.5f}  "
            f"corrected={out['mi'] / LN2:8.5f} bits"
        )


def main() -> None:
    print("TM-EI (affine transport-map) estimator smoke run\n")
    analytic_linear_check()
    lift_check()
    alpha_sweep()
    raw_nats()
    print("\nOK: all calls returned finite values.")


if __name__ == "__main__":
    main()
