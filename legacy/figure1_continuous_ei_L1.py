import csv
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


EPS = 1e-12


@dataclass(frozen=True)
class Config:
    L: float = 1.0
    sigma_list: tuple = (0.01, 0.05, 0.1)
    seeds: tuple = (0, 1, 2, 3, 4)
    single_samples: int = 2000
    manifold_samples: int = 1000
    single_weights: tuple = tuple(np.linspace(0.0, 27.0, 91))
    manifold_weights: tuple = tuple(np.linspace(-9.0, 9.0, 41))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_act(x):
    return np.tanh(x)


def tanh_prime(x):
    t = np.tanh(x)
    return 1.0 - t * t


def relu(x):
    return np.maximum(0.0, x)


def relu_prime(x):
    return (x > 0.0).astype(np.float64)


def apply_output_clip_to_jacobian(act_name, z, jac):
    if act_name != "relu":
        return jac
    return np.where(z <= 0.0, 0.0, np.where(z < 1.0, jac, 1.0))


def ensure_positive(v):
    return np.maximum(v, EPS)


def get_sampling_bounds(act_name):
    if act_name == "tanh":
        return -1.0, 1.0, -1.0, 1.0
    return 0.0, 1.0, 0.0, 1.0


def sample_input_points(act_name, single_samples, manifold_samples, seed):
    rng = np.random.default_rng(seed)
    x_low, x_high, _, _ = get_sampling_bounds(act_name)
    single_x = rng.uniform(x_low, x_high, size=single_samples)
    manifold_a = rng.uniform(x_low, x_high, size=manifold_samples)
    manifold_b = rng.uniform(x_low, x_high, size=manifold_samples)
    return {
        "x_low": x_low,
        "x_high": x_high,
        "single_x": single_x,
        "manifold_a": manifold_a,
        "manifold_b": manifold_b,
    }


def decompose_ei(log_pdet_mean, n_in, n_out, input_width, sigma):
    """Return (term1_nats, term2_nats, ei_nats) decomposed into determinism & non-degeneracy."""
    sigma2 = sigma * sigma
    # term1: determinism — certainty of the mapping given input
    # -H(Y|X) = -(1/2) n_out log(2π e σ²)
    term1 = -0.5 * n_out * (1.0 + math.log(2.0 * math.pi) + math.log(sigma2))
    # term2: non-degeneracy — volume change from input space through the mapping
    # H(X) + E[log|det J|], where H(X) = n_in log(input_width)
    term2 = n_in * math.log(input_width) + log_pdet_mean
    return term1, term2, term1 + term2


def to_bits(nats):
    return nats / math.log(2.0)


def print_progress(prefix, current, total, last_percent, step_percent=5):
    pct = int((current * 100) / max(total, 1))
    if pct >= last_percent + step_percent or current == total:
        print(f"[{prefix}] {pct:3d}% ({current}/{total})", flush=True)
        return pct
    return last_percent


def run_single_edge(config, act_name, act_prime_fn):
    rows = []
    summary_rows = []
    total_steps = len(config.sigma_list) * len(config.seeds) * len(config.single_weights)
    step = 0
    last_pct = -1
    x_low, x_high, _, _ = get_sampling_bounds(act_name)
    input_width = x_high - x_low

    for sigma in config.sigma_list:
        for seed in config.seeds:
            rng = np.random.default_rng(seed)
            x = rng.uniform(x_low, x_high, size=config.single_samples)

            curve_vals = []
            for w in config.single_weights:
                z = w * x
                jac = w * act_prime_fn(z)
                jac = apply_output_clip_to_jacobian(act_name, z, jac)
                log_pdet_mean = np.mean(np.log(ensure_positive(np.abs(jac))))
                term1_nats, term2_nats, ei_nats = decompose_ei(
                    log_pdet_mean, n_in=1, n_out=1, input_width=input_width, sigma=sigma
                )
                ei_bits = to_bits(ei_nats)
                term1_bits = to_bits(term1_nats)
                term2_bits = to_bits(term2_nats)
                curve_vals.append(ei_bits)
                rows.append(
                    {
                        "panel": f"single_{act_name}",
                        "activation": act_name,
                        "sigma": float(sigma),
                        "seed": int(seed),
                        "weight": float(w),
                        "ei_bits": float(ei_bits),
                        "term1_bits": float(term1_bits),
                        "term2_bits": float(term2_bits),
                    }
                )
                step += 1
                last_pct = print_progress(f"single-{act_name}", step, total_steps, last_pct)

            peak_idx = int(np.argmax(curve_vals))
            summary_rows.append(
                {
                    "activation": act_name,
                    "sigma": float(sigma),
                    "seed": int(seed),
                    "peak_weight": float(config.single_weights[peak_idx]),
                    "peak_ei_bits": float(curve_vals[peak_idx]),
                }
            )

    return rows, summary_rows


def run_two_to_one(config, act_name, act_prime_fn):
    rows = []
    total_steps = (
        len(config.sigma_list)
        * len(config.seeds)
        * len(config.manifold_weights)
        * len(config.manifold_weights)
    )
    step = 0
    last_pct = -1
    x_low, x_high, _, _ = get_sampling_bounds(act_name)
    input_width = x_high - x_low

    for sigma in config.sigma_list:
        for seed in config.seeds:
            rng = np.random.default_rng(seed)
            a = rng.uniform(x_low, x_high, size=config.manifold_samples)
            b = rng.uniform(x_low, x_high, size=config.manifold_samples)

            for w_ac in config.manifold_weights:
                for w_bc in config.manifold_weights:
                    z = w_ac * a + w_bc * b
                    sp = act_prime_fn(z)
                    j1 = w_ac * sp
                    j2 = w_bc * sp
                    j1 = apply_output_clip_to_jacobian(act_name, z, j1)
                    j2 = apply_output_clip_to_jacobian(act_name, z, j2)
                    pseudo_det = np.sqrt(j1 * j1 + j2 * j2)
                    log_pdet_mean = np.mean(np.log(ensure_positive(pseudo_det)))
                    term1_nats, term2_nats, ei_nats = decompose_ei(
                        log_pdet_mean,
                        n_in=2,
                        n_out=1,
                        input_width=input_width,
                        sigma=sigma,
                    )
                    ei_bits = to_bits(ei_nats)
                    term1_bits = to_bits(term1_nats)
                    term2_bits = to_bits(term2_nats)
                    rows.append(
                        {
                            "panel": f"two_to_one_{act_name}",
                            "activation": act_name,
                            "sigma": float(sigma),
                            "seed": int(seed),
                            "weight_a_to_c": float(w_ac),
                            "weight_b_to_c": float(w_bc),
                            "ei_bits": float(ei_bits),
                            "term1_bits": float(term1_bits),
                            "term2_bits": float(term2_bits),
                        }
                    )
                    step += 1
                    last_pct = print_progress(f"manifold-{act_name}", step, total_steps, last_pct)

    return rows


def aggregate_single(rows):
    # key: (activation, sigma, weight)
    stats = {}
    for r in rows:
        key = (r["activation"], r["sigma"], r["weight"])
        stats.setdefault(key, {
            "ei_bits": [],
            "term1_bits": [],
            "term2_bits": [],
        })
        stats[key]["ei_bits"].append(r["ei_bits"])
        stats[key]["term1_bits"].append(r["term1_bits"])
        stats[key]["term2_bits"].append(r["term2_bits"])

    out = []
    for (activation, sigma, weight), v in sorted(stats.items()):
        ei_arr = np.asarray(v["ei_bits"], dtype=np.float64)
        t1_arr = np.asarray(v["term1_bits"], dtype=np.float64)
        t2_arr = np.asarray(v["term2_bits"], dtype=np.float64)
        out.append(
            {
                "activation": activation,
                "sigma": sigma,
                "weight": weight,
                "mean_ei_bits": float(ei_arr.mean()),
                "std_ei_bits": float(ei_arr.std(ddof=0)),
                "mean_term1_bits": float(t1_arr.mean()),
                "std_term1_bits": float(t1_arr.std(ddof=0)),
                "mean_term2_bits": float(t2_arr.mean()),
                "std_term2_bits": float(t2_arr.std(ddof=0)),
            }
        )
    return out


def aggregate_manifold(rows):
    # key: (activation, sigma, w_ac, w_bc)
    stats = {}
    for r in rows:
        key = (r["activation"], r["sigma"], r["weight_a_to_c"], r["weight_b_to_c"])
        stats.setdefault(key, {
            "ei_bits": [],
            "term1_bits": [],
            "term2_bits": [],
        })
        stats[key]["ei_bits"].append(r["ei_bits"])
        stats[key]["term1_bits"].append(r["term1_bits"])
        stats[key]["term2_bits"].append(r["term2_bits"])

    out = []
    for (activation, sigma, w_ac, w_bc), v in sorted(stats.items()):
        ei_arr = np.asarray(v["ei_bits"], dtype=np.float64)
        t1_arr = np.asarray(v["term1_bits"], dtype=np.float64)
        t2_arr = np.asarray(v["term2_bits"], dtype=np.float64)
        out.append(
            {
                "activation": activation,
                "sigma": sigma,
                "weight_a_to_c": w_ac,
                "weight_b_to_c": w_bc,
                "mean_ei_bits": float(ei_arr.mean()),
                "std_ei_bits": float(ei_arr.std(ddof=0)),
                "mean_term1_bits": float(t1_arr.mean()),
                "std_term1_bits": float(t1_arr.std(ddof=0)),
                "mean_term2_bits": float(t2_arr.mean()),
                "std_term2_bits": float(t2_arr.std(ddof=0)),
            }
        )
    return out


def save_csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


FIELD_LABELS = {
    "ei": ("EI (bits)", "Continuous EI"),
    "term1": ("Term1 — Determinism (bits)", "Continuous Term1 — Determinism"),
    "term2": ("Term2 — Non-degeneracy (bits)", "Continuous Term2 — Non-degeneracy"),
}


def plot_single_field(summary_rows, out_png, field="ei"):
    mean_key = f"mean_{field}_bits"
    std_key = f"std_{field}_bits"
    ylabel, suptitle_prefix = FIELD_LABELS[field]
    activations = ["sigmoid", "tanh", "relu"]
    sigma_colors = {0.01: "#1f77b4", 0.05: "#ff7f0e", 0.1: "#2ca02c"}

    fig = plt.figure(figsize=(16, 5))
    for i, act in enumerate(activations):
        ax = fig.add_subplot(1, 3, i + 1)
        for sigma in sorted(sigma_colors):
            rows = [r for r in summary_rows if r["activation"] == act and abs(r["sigma"] - sigma) < 1e-12]
            rows = sorted(rows, key=lambda x: x["weight"])
            x = np.array([r["weight"] for r in rows], dtype=np.float64)
            y = np.array([r[mean_key] for r in rows], dtype=np.float64)
            s = np.array([r[std_key] for r in rows], dtype=np.float64)
            ax.plot(x, y, color=sigma_colors[sigma], lw=1.8, label=f"sigma={sigma}")
            ax.fill_between(x, y - s, y + s, color=sigma_colors[sigma], alpha=0.15)

        if field == "ei":
            if act == "sigmoid":
                x_star, note = math.e, "e"
            elif act == "tanh":
                x_star, note = 1.0 / math.tanh(1.0), "coth(1)"
            else:
                x_star, note = 1.0, "1"
            ax.axvline(x_star, color="black", lw=1.0, ls="--")
            ax.text(x_star, ax.get_ylim()[1] * 0.95, f" {note}", va="top")

        ax.set_title(f"{act.title()}")
        ax.set_xlabel("A -> B Weight")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(f"{suptitle_prefix} Figure 1 (a-c), multi-sigma", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_sampling_distributions(config, out_png):
    activations = ["sigmoid", "tanh", "relu"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for col, act_name in enumerate(activations):
        sampled = sample_input_points(
            act_name,
            single_samples=config.single_samples,
            manifold_samples=config.manifold_samples,
            seed=config.seeds[0],
        )

        ax = axes[0, col]
        ax.hist(
            sampled["single_x"],
            bins=40,
            color="#4C72B0",
            alpha=0.85,
            edgecolor="white",
        )
        ax.set_title(f"{act_name.title()}: single-edge input")
        ax.set_xlabel("Input value")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)

        ax = axes[1, col]
        ax.scatter(
            sampled["manifold_a"],
            sampled["manifold_b"],
            s=12,
            alpha=0.22,
            color="#DD8452",
            linewidths=0,
        )
        ax.set_title(f"{act_name.title()}: two-input samples")
        ax.set_xlabel("A")
        ax.set_ylabel("B")
        ax.set_xlim(sampled["x_low"], sampled["x_high"])
        ax.set_ylim(sampled["x_low"], sampled["x_high"])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

    fig.suptitle("Continuous EI Sampling Distributions", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def aggregate_manifold_over_sigma(summary_rows):
    # Build a single manifold per activation by averaging over sigma values.
    stats = {}
    for r in summary_rows:
        key = (r["activation"], r["weight_a_to_c"], r["weight_b_to_c"])
        stats.setdefault(key, {
            "mean_ei_bits": [],
            "mean_term1_bits": [],
            "mean_term2_bits": [],
        })
        stats[key]["mean_ei_bits"].append(r["mean_ei_bits"])
        stats[key]["mean_term1_bits"].append(r["mean_term1_bits"])
        stats[key]["mean_term2_bits"].append(r["mean_term2_bits"])

    out = []
    for (activation, w_ac, w_bc), v in sorted(stats.items()):
        ei_arr = np.asarray(v["mean_ei_bits"], dtype=np.float64)
        t1_arr = np.asarray(v["mean_term1_bits"], dtype=np.float64)
        t2_arr = np.asarray(v["mean_term2_bits"], dtype=np.float64)
        out.append(
            {
                "activation": activation,
                "weight_a_to_c": w_ac,
                "weight_b_to_c": w_bc,
                "mean_ei_bits": float(ei_arr.mean()),
                "mean_term1_bits": float(t1_arr.mean()),
                "mean_term2_bits": float(t2_arr.mean()),
            }
        )
    return out


def plot_manifold_2d_and_3d(summary_rows, out_dir, field="ei"):
    mean_key = f"mean_{field}_bits"
    zlabel, suptitle_prefix = FIELD_LABELS[field]
    activations = ["sigmoid", "tanh", "relu"]
    sigmas = [0.01, 0.05, 0.1]
    merged = aggregate_manifold_over_sigma(summary_rows)

    # 2D contour figure: rows=activation, cols=sigma
    fig2d = plt.figure(figsize=(16, 12))
    for r_idx, act in enumerate(activations):
        for c_idx, sigma in enumerate(sigmas):
            ax = fig2d.add_subplot(3, 3, r_idx * 3 + c_idx + 1)
            rows = [
                r for r in summary_rows
                if r["activation"] == act and abs(r["sigma"] - sigma) < 1e-12
            ]
            w_a_vals = sorted({r["weight_a_to_c"] for r in rows})
            w_b_vals = sorted({r["weight_b_to_c"] for r in rows})
            z = np.zeros((len(w_b_vals), len(w_a_vals)), dtype=np.float64)
            idx_a = {v: k for k, v in enumerate(w_a_vals)}
            idx_b = {v: k for k, v in enumerate(w_b_vals)}
            for row in rows:
                z[idx_b[row["weight_b_to_c"]], idx_a[row["weight_a_to_c"]]] = row[mean_key]

            c = ax.contourf(w_a_vals, w_b_vals, z, levels=24, cmap="viridis")
            fig2d.colorbar(c, ax=ax)
            ax.set_title(f"{act.title()}, sigma={sigma}")
            ax.set_xlabel("A -> C Weight")
            ax.set_ylabel("B -> C Weight")

    fig2d.suptitle(f"{suptitle_prefix} Figure 1 (d-f) 2D Contours, no sigma fusion", fontsize=13)
    fig2d.tight_layout(rect=[0, 0, 1, 0.92])
    fig2d.savefig(out_dir / f"continuous_figure1_manifold_2d_{field}.png", dpi=170)
    plt.close(fig2d)

    # 3D surface figure: one panel per activation, sigma-averaged
    fig3d = plt.figure(figsize=(16, 4.8))
    for i, act in enumerate(activations):
        ax = fig3d.add_subplot(1, 3, i + 1, projection="3d")
        rows = [r for r in merged if r["activation"] == act]
        w_a_vals = sorted({r["weight_a_to_c"] for r in rows})
        w_b_vals = sorted({r["weight_b_to_c"] for r in rows})
        z = np.zeros((len(w_b_vals), len(w_a_vals)), dtype=np.float64)
        idx_a = {v: k for k, v in enumerate(w_a_vals)}
        idx_b = {v: k for k, v in enumerate(w_b_vals)}
        for r in rows:
            z[idx_b[r["weight_b_to_c"]], idx_a[r["weight_a_to_c"]]] = r[mean_key]

        w_a_grid, w_b_grid = np.meshgrid(w_a_vals, w_b_vals)
        surf = ax.plot_surface(w_a_grid, w_b_grid, z, cmap="viridis", linewidth=0, antialiased=True)
        fig3d.colorbar(surf, ax=ax, shrink=0.7, pad=0.08)
        ax.set_title(act.title())
        ax.set_xlabel("A -> C Weight")
        ax.set_ylabel("B -> C Weight")
        ax.set_zlabel(zlabel)
        ax.view_init(elev=27, azim=-135)

    fig3d.suptitle(f"{suptitle_prefix} Figure 1 (d-f) 3D Surfaces, sigma-averaged", fontsize=13)
    fig3d.tight_layout(rect=[0, 0, 1, 0.92])
    fig3d.savefig(out_dir / f"continuous_figure1_manifold_3d_{field}.png", dpi=170)
    plt.close(fig3d)


def main():
    config = Config()
    t0 = time.time()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / f"figure1_continuous_L1_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    activations = [
        ("sigmoid", sigmoid_prime),
        ("tanh", tanh_prime),
        ("relu", relu_prime),
    ]

    all_single_raw = []
    all_single_peaks = []
    all_manifold_raw = []

    print("[Stage] Running single-edge continuous EI (a-c)", flush=True)
    for act_name, act_prime in activations:
        print(f"[Start] single-edge {act_name}", flush=True)
        single_raw, single_peaks = run_single_edge(config, act_name, act_prime)
        all_single_raw.extend(single_raw)
        all_single_peaks.extend(single_peaks)
        print(f"[Done] single-edge {act_name}", flush=True)

    print("[Stage] Running two-to-one continuous EI manifolds (d-f)", flush=True)
    for act_name, act_prime in activations:
        print(f"[Start] manifold {act_name}", flush=True)
        man_raw = run_two_to_one(config, act_name, act_prime)
        all_manifold_raw.extend(man_raw)
        print(f"[Done] manifold {act_name}", flush=True)

    single_summary = aggregate_single(all_single_raw)
    manifold_summary = aggregate_manifold(all_manifold_raw)

    save_csv(all_single_raw, out_dir / "single_edge_raw.csv")
    save_csv(single_summary, out_dir / "single_edge_summary.csv")
    save_csv(all_single_peaks, out_dir / "single_edge_peak_by_seed.csv")
    save_csv(all_manifold_raw, out_dir / "manifold_raw.csv")
    save_csv(manifold_summary, out_dir / "manifold_summary.csv")

    plot_sampling_distributions(config, out_dir / "continuous_sampling_distributions.png")

    for field in ("ei", "term1", "term2"):
        plot_single_field(single_summary, out_dir / f"continuous_figure1_single_{field}.png", field=field)
        plot_manifold_2d_and_3d(manifold_summary, out_dir, field=field)

    elapsed = time.time() - t0
    print(f"Saved outputs to: {out_dir}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
