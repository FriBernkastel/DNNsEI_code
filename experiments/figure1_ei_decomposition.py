import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eicore.common import EPS
from eicore.discrete import causal_metrics_from_encoded_states, discretize_uniform, encode_states
from eicore.plotting import TOTAL_COLOR, apply_style, finalize_figure
from eicore.report import publish_latest

LOG2 = math.log(2.0)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


ACTS = {
    "sigmoid": {
        "fn": _sigmoid,
        "deriv": lambda z: _sigmoid(z) * (1.0 - _sigmoid(z)),
        "x_range": (0.0, 1.0),
        "y_range": (0.0, 1.0),
    },
    "tanh": {
        "fn": np.tanh,
        "deriv": lambda z: 1.0 - np.tanh(z) ** 2,
        "x_range": (-1.0, 1.0),
        "y_range": (-1.0, 1.0),
    },
    "relu": {
        "fn": lambda z: np.maximum(z, 0.0),
        "deriv": lambda z: (z > 0.0).astype(np.float64),
        "x_range": (0.0, 1.0),
        "y_range": (0.0, 1.0),
    },
}


@dataclass(frozen=True)
class Figure1Config:
    n_weight_points: int = 91
    max_weight: float = 27.0
    single_samples: int = 30000
    bins_list: tuple = (8, 16, 32, 64)
    grid_size: int = 41
    grid_max_weight: float = 9.0
    grid_samples: int = 100000
    grid_bins: int = 32
    sigma: float = 0.1
    seed: int = 42
    outdir: str = "results"
    act_names: tuple = ("sigmoid", "tanh", "relu")


def single_edge_discrete(act_name, weights, bins_list, n_samples, rng):
    spec = ACTS[act_name]
    x_low, x_high = spec["x_range"]
    y_low, y_high = spec["y_range"]
    a = rng.uniform(x_low, x_high, size=n_samples).astype(np.float64)

    rows = []
    for b in bins_list:
        x_state = encode_states(discretize_uniform(a, x_low, x_high, b), b)
        for w in weights:
            y = spec["fn"](w * a)
            y_state = encode_states(discretize_uniform(y, y_low, y_high, b), b)
            metrics = causal_metrics_from_encoded_states(x_state, y_state, n_cause_states=b)
            ei_bits = metrics["ei_bits"]
            rows.append(
                {
                    "act": act_name,
                    "method": "discrete",
                    "bins": b,
                    "weight": float(w),
                    "ei_bits": ei_bits,
                    "sensitivity_bits": ei_bits,
                    "degeneracy_bits": 0.0,
                }
            )
    return rows


def single_edge_continuous(act_name, weights, sigma, n_samples, rng):
    spec = ACTS[act_name]
    x_low, x_high = spec["x_range"]
    a = rng.uniform(x_low, x_high, size=n_samples).astype(np.float64)
    width = x_high - x_low

    term1_nats = -0.5 * math.log(2.0 * math.pi * math.e * sigma * sigma)
    rows = []
    for w in weights:
        d = spec["deriv"](w * a)
        log_j = np.log(np.clip(np.abs(w * d), EPS, None))
        term2_nats = math.log(width) + float(np.mean(log_j))
        ei_bits = (term1_nats + term2_nats) / LOG2
        rows.append(
            {
                "act": act_name,
                "method": "continuous",
                "bins": 0,
                "weight": float(w),
                "ei_bits": ei_bits,
                "term1_bits": term1_nats / LOG2,
                "term2_bits": term2_nats / LOG2,
                "sensitivity_bits": ei_bits,
                "degeneracy_bits": 0.0,
            }
        )
    return rows


def two_to_one_discrete(act_name, grid, bins, n_samples, rng):
    spec = ACTS[act_name]
    x_low, x_high = spec["x_range"]
    y_low, y_high = spec["y_range"]
    a = rng.uniform(x_low, x_high, size=n_samples).astype(np.float64)
    b = rng.uniform(x_low, x_high, size=n_samples).astype(np.float64)

    ab_codes = np.stack(
        [discretize_uniform(a, x_low, x_high, bins), discretize_uniform(b, x_low, x_high, bins)],
        axis=1,
    )
    c_state = encode_states(ab_codes, bins)
    n_cause_states = bins * bins
    a_state = encode_states(discretize_uniform(a, x_low, x_high, bins), bins)
    b_state = encode_states(discretize_uniform(b, x_low, x_high, bins), bins)

    size = grid.shape[0]
    ei = np.zeros((size, size))
    sens = np.zeros((size, size))
    deg = np.zeros((size, size))
    for i, w_ac in enumerate(grid):
        y_from_a = spec["fn"](w_ac * a)
        ea_state = encode_states(discretize_uniform(y_from_a, y_low, y_high, bins), bins)
        mi_a = causal_metrics_from_encoded_states(a_state, ea_state, n_cause_states=bins)["ei_bits"]
        for j, w_bc in enumerate(grid):
            y = spec["fn"](w_ac * a + w_bc * b)
            e_state = encode_states(discretize_uniform(y, y_low, y_high, bins), bins)
            ei[i, j] = causal_metrics_from_encoded_states(c_state, e_state, n_cause_states)["ei_bits"]
            y_from_b = spec["fn"](w_bc * b)
            eb_state = encode_states(discretize_uniform(y_from_b, y_low, y_high, bins), bins)
            mi_b = causal_metrics_from_encoded_states(b_state, eb_state, n_cause_states=bins)["ei_bits"]
            sens[i, j] = mi_a + mi_b
            deg[i, j] = sens[i, j] - ei[i, j]
    return {"act": act_name, "method": "discrete", "grid": grid, "ei": ei, "sensitivity": sens, "degeneracy": deg}


def two_to_one_continuous(act_name, grid, sigma, n_samples, rng):
    spec = ACTS[act_name]
    x_low, x_high = spec["x_range"]
    a = rng.uniform(x_low, x_high, size=n_samples).astype(np.float64)
    b = rng.uniform(x_low, x_high, size=n_samples).astype(np.float64)
    width = x_high - x_low

    term1_nats = -0.5 * math.log(2.0 * math.pi * math.e * sigma * sigma)
    size = grid.shape[0]
    ei = np.zeros((size, size))
    sens = np.zeros((size, size))
    deg = np.zeros((size, size))
    for i, w_ac in enumerate(grid):
        for j, w_bc in enumerate(grid):
            d = spec["deriv"](w_ac * a + w_bc * b)
            j_a = w_ac * d
            j_b = w_bc * d
            log_pdet = float(np.mean(np.log(np.clip(np.sqrt(j_a * j_a + j_b * j_b), EPS, None))))
            log_ja = float(np.mean(np.log(np.clip(np.abs(j_a), EPS, None))))
            log_jb = float(np.mean(np.log(np.clip(np.abs(j_b), EPS, None))))
            ei_nats = (term1_nats + 2.0 * math.log(width) + log_pdet) / 2.0
            sens_nats = (2.0 * term1_nats + 2.0 * math.log(width) + log_ja + log_jb) / 2.0
            ei[i, j] = ei_nats / LOG2
            sens[i, j] = sens_nats / LOG2
            deg[i, j] = sens[i, j] - ei[i, j]
    return {"act": act_name, "method": "continuous", "grid": grid, "ei": ei, "sensitivity": sens, "degeneracy": deg}


BINS_COLORS = {8: "#9ecae1", 16: "#6baed6", 32: "#3182bd", 64: "#08519c"}


def plot_single_edge(single_rows, weights, act_names, bins_list, out_path):
    apply_style()
    fig, axes = plt.subplots(1, len(act_names), figsize=(6.2 * len(act_names), 4.6), sharey=False)
    if len(act_names) == 1:
        axes = [axes]
    for ax, act_name in zip(axes, act_names):
        for b in bins_list:
            vals = [row["ei_bits"] for row in single_rows if row["act"] == act_name and row["method"] == "discrete" and row["bins"] == b]
            ax.plot(weights, vals, color=BINS_COLORS.get(b), linewidth=1.8, label=f"discrete bins={b}")
        cont = [row["ei_bits"] for row in single_rows if row["act"] == act_name and row["method"] == "continuous"]
        ax.plot(weights, cont, color=TOTAL_COLOR, linewidth=2.4, linestyle="--", label="continuous")
        ax.set_xlabel("A → B Weight")
        ax.set_ylabel("EI (bits)")
        ax.set_title(act_name.capitalize())
        ax.legend(loc="best")
    finalize_figure(fig, out_path, suptitle="Single Edge A → B: Discrete vs Continuous EI (sensitivity = EI, degeneracy = 0)")
    return out_path


def plot_two_to_one_heatmaps(discrete_grid, continuous_grid, out_path):
    apply_style()
    grid = discrete_grid["grid"]
    extent = (grid[0], grid[-1], grid[0], grid[-1])
    panels = [
        ("ei", "EI (bits)"),
        ("sensitivity", "Sensitivity (bits)"),
        ("degeneracy", "Degeneracy (bits)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for col, (key, label) in enumerate(panels):
        vmin = min(float(np.min(discrete_grid[key])), float(np.min(continuous_grid[key])))
        vmax = max(float(np.max(discrete_grid[key])), float(np.max(continuous_grid[key])))
        for row, (data, method) in enumerate(((discrete_grid, "Discrete"), (continuous_grid, "Continuous"))):
            ax = axes[row][col]
            im = ax.imshow(
                data[key].T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("B → C Weight")
            ax.set_ylabel("A → C Weight")
            ax.set_title(f"{method} {label}")
            fig.colorbar(im, ax=ax, shrink=0.85)
    finalize_figure(fig, out_path, suptitle=f"Two-to-One A, B → C ({discrete_grid['act'].capitalize()}): Discrete vs Continuous")
    return out_path


def plot_two_to_one_diagonal(discrete_grids, continuous_grids, act_names, out_path):
    apply_style()
    panels = [
        ("ei", "EI (bits)"),
        ("sensitivity", "Sensitivity (bits)"),
        ("degeneracy", "Degeneracy (bits)"),
    ]
    fig, axes = plt.subplots(len(panels), len(act_names), figsize=(6.2 * len(act_names), 4.2 * len(panels)))
    for col, act_name in enumerate(act_names):
        disc = discrete_grids[act_name]
        cont = continuous_grids[act_name]
        grid = disc["grid"]
        diag_idx = np.arange(grid.shape[0])
        for row, (key, label) in enumerate(panels):
            ax = axes[row][col]
            ax.plot(grid, disc[key][diag_idx, diag_idx], color="#3182bd", linewidth=2.0, label="discrete")
            ax.plot(grid, cont[key][diag_idx, diag_idx], color="#222222", linewidth=2.0, linestyle="--", label="continuous")
            ax.set_xlabel("Weight (A → C = B → C)")
            ax.set_ylabel(label)
            ax.set_title(f"{act_name.capitalize()} {label.split(' ')[0]}")
            ax.legend(loc="best")
    finalize_figure(fig, out_path, suptitle="Two-to-One Diagonal Slice: Discrete vs Continuous")
    return out_path


def write_single_edge_csv(rows, path):
    fieldnames = ["act", "method", "bins", "weight", "ei_bits", "sensitivity_bits", "degeneracy_bits", "term1_bits", "term2_bits"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_grid_csv(grids, path):
    fieldnames = ["act", "method", "w_ac", "w_bc", "ei_bits", "sensitivity_bits", "degeneracy_bits"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for data in grids:
            grid = data["grid"]
            for i, w_ac in enumerate(grid):
                for j, w_bc in enumerate(grid):
                    writer.writerow(
                        {
                            "act": data["act"],
                            "method": data["method"],
                            "w_ac": float(w_ac),
                            "w_bc": float(w_bc),
                            "ei_bits": float(data["ei"][i, j]),
                            "sensitivity_bits": float(data["sensitivity"][i, j]),
                            "degeneracy_bits": float(data["degeneracy"][i, j]),
                        }
                    )


def run_experiment(config, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    weights = np.linspace(0.0, config.max_weight, config.n_weight_points)
    grid = np.linspace(-config.grid_max_weight, config.grid_max_weight, config.grid_size)

    single_rows = []
    for act_name in config.act_names:
        single_rows.extend(single_edge_discrete(act_name, weights, config.bins_list, config.single_samples, rng))
        single_rows.extend(single_edge_continuous(act_name, weights, config.sigma, config.single_samples, rng))
        print(f"[single-edge] {act_name} done", flush=True)

    outputs = {}
    single_csv = run_dir / "single_edge_history.csv"
    write_single_edge_csv(single_rows, single_csv)
    outputs["single_csv"] = single_csv
    outputs["single_fig"] = plot_single_edge(
        single_rows, weights, config.act_names, config.bins_list, run_dir / "single_edge_ei.png"
    )

    discrete_grids = {}
    continuous_grids = {}
    for act_name in config.act_names:
        discrete_grids[act_name] = two_to_one_discrete(act_name, grid, config.grid_bins, config.grid_samples, rng)
        print(f"[two-to-one] {act_name} discrete done", flush=True)
        continuous_grids[act_name] = two_to_one_continuous(act_name, grid, config.sigma, config.grid_samples, rng)
        print(f"[two-to-one] {act_name} continuous done", flush=True)
        outputs[f"heatmap_{act_name}"] = plot_two_to_one_heatmaps(
            discrete_grids[act_name],
            continuous_grids[act_name],
            run_dir / f"two_to_one_{act_name}.png",
        )

    grid_csv = run_dir / "two_to_one_grid.csv"
    write_grid_csv(list(discrete_grids.values()) + list(continuous_grids.values()), grid_csv)
    outputs["grid_csv"] = grid_csv
    outputs["diagonal_fig"] = plot_two_to_one_diagonal(
        discrete_grids, continuous_grids, config.act_names, run_dir / "two_to_one_diagonal.png"
    )
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Paper Figure 1 simple systems: discrete vs continuous EI/sensitivity/degeneracy.")
    parser.add_argument("--single-samples", type=int, default=Figure1Config.single_samples)
    parser.add_argument("--grid-samples", type=int, default=Figure1Config.grid_samples)
    parser.add_argument("--grid-size", type=int, default=Figure1Config.grid_size)
    parser.add_argument("--grid-bins", type=int, default=Figure1Config.grid_bins)
    parser.add_argument("--sigma", type=float, default=Figure1Config.sigma)
    parser.add_argument("--seed", type=int, default=Figure1Config.seed)
    parser.add_argument("--outdir", type=str, default=Figure1Config.outdir)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Figure1Config(
        single_samples=args.single_samples,
        grid_samples=args.grid_samples,
        grid_size=args.grid_size,
        grid_bins=args.grid_bins,
        sigma=args.sigma,
        seed=args.seed,
        outdir=args.outdir,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.outdir) / f"figure1_ei_decomposition_{timestamp}"
    outputs = run_experiment(config, run_dir)
    outputs["latest_dir"] = publish_latest(run_dir, config.outdir)
    for path in outputs.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
