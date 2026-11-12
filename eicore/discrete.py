import math

import numpy as np


def discretize_uniform(values, low, high, bins, clip=True):
    scaled = (values - low) / (high - low)
    idx = np.floor(scaled * bins).astype(np.int64)
    if clip:
        np.clip(idx, 0, bins - 1, out=idx)
    return idx


def encode_states(codes, bins):
    if codes.ndim == 1:
        return codes.astype(np.int64)
    powers = (bins ** np.arange(codes.shape[1], dtype=np.int64)).reshape(1, -1)
    return (codes * powers).sum(axis=1, dtype=np.int64)


def causal_metrics_from_encoded_states(c_state, e_state, n_cause_states):
    n_samples = c_state.shape[0]
    c_u, c_inv = np.unique(c_state, return_inverse=True)
    e_u, e_inv = np.unique(e_state, return_inverse=True)

    pairs = np.stack([c_inv, e_inv], axis=1)
    pair_u, pair_counts = np.unique(pairs, axis=0, return_counts=True)

    pc = np.bincount(pair_u[:, 0], weights=pair_counts, minlength=c_u.shape[0]).astype(np.float64)
    pe = np.bincount(pair_u[:, 1], weights=pair_counts, minlength=e_u.shape[0]).astype(np.float64)

    pce = pair_counts.astype(np.float64) / n_samples
    pc = pc / n_samples
    pe = pe / n_samples

    eps = 1e-30
    log_pc = np.log2(pc[pair_u[:, 0]] + eps)
    log_pe = np.log2(pe[pair_u[:, 1]] + eps)
    log_pce = np.log2(pce + eps)

    h_e_bits = -np.sum(pe * np.log2(pe + eps))
    h_e_given_c_bits = -np.sum(pce * (log_pce - log_pc))
    ei_bits = np.sum(pce * (log_pce - log_pc - log_pe))

    max_cause_bits = math.log2(n_cause_states)
    determinism = 1.0 - (h_e_given_c_bits / max_cause_bits)
    degeneracy = 1.0 - (h_e_bits / max_cause_bits)
    non_degeneracy = -degeneracy
    specificity = 1.0 - non_degeneracy
    causal_power = determinism - non_degeneracy

    metrics = {
        "ei_bits": float(ei_bits),
        "h_e_bits": float(h_e_bits),
        "h_e_given_c_bits": float(h_e_given_c_bits),
        "determinism": float(determinism),
        "non_degeneracy": float(non_degeneracy),
        "specificity": float(specificity),
        "causal_power": float(causal_power),
    }
    for key, value in metrics.items():
        if abs(value) < 1e-12:
            metrics[key] = 0.0
    return metrics


def sigmoid_np(z):
    return 1.0 / (1.0 + np.exp(-z))


def discrete_layer_ei(weight, n_samples, bins, rng):
    weight = np.asarray(weight, dtype=np.float64)
    n_out, n_in = weight.shape

    x = rng.uniform(0.0, 1.0, size=(n_samples, n_in))
    y = sigmoid_np(x @ weight.T)
    c_state = encode_states(discretize_uniform(x, 0.0, 1.0, bins), bins)
    e_state = encode_states(discretize_uniform(y, 0.0, 1.0, bins), bins)
    metrics = causal_metrics_from_encoded_states(c_state, e_state, bins ** n_in)
    ei_bits = metrics["ei_bits"]

    sensitivity_bits = 0.0
    for i in range(n_in):
        xs = np.zeros((n_samples, n_in), dtype=np.float64)
        xs[:, i] = rng.uniform(0.0, 1.0, size=n_samples)
        ys = sigmoid_np(xs @ weight.T)
        xi_code = discretize_uniform(xs[:, i], 0.0, 1.0, bins)
        for j in range(n_out):
            yj_code = discretize_uniform(ys[:, j], 0.0, 1.0, bins)
            sensitivity_bits += causal_metrics_from_encoded_states(xi_code, yj_code, bins)["ei_bits"]

    degeneracy_bits = sensitivity_bits - ei_bits
    return {
        "ei_bits": ei_bits,
        "h_e_bits": metrics["h_e_bits"],
        "h_e_given_c_bits": metrics["h_e_given_c_bits"],
        "sensitivity_bits": sensitivity_bits,
        "degeneracy_bits": degeneracy_bits,
        "ei_norm_bits": ei_bits / max(n_in, 1),
        "sensitivity_norm_bits": sensitivity_bits / max(n_in, 1),
        "degeneracy_norm_bits": degeneracy_bits / max(n_in, 1),
    }


def compute_model_layer_ei_discrete(model, n_samples, bins, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for idx, layer in enumerate(model.layers, start=1):
        weight = layer.weight.detach().cpu().numpy().astype(np.float64)
        metrics = discrete_layer_ei(weight, n_samples, bins, rng)
        rows.append(
            {
                "layer": f"L{idx}",
                "n_in": layer.in_features,
                "n_out": layer.out_features,
                **metrics,
            }
        )
    return rows
