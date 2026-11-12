import math

import numpy as np
import torch

from eicore.common import EPS


def sigmoid_layer_jacobian(weight, bias, samples):
    weight = weight.to(dtype=samples.dtype, device=samples.device)
    z = samples @ weight.t()
    if bias is not None:
        bias = bias.to(dtype=samples.dtype, device=samples.device)
        z = z + bias
    s = torch.sigmoid(z)
    sigmoid_prime = s * (1.0 - s)
    return sigmoid_prime[:, :, None] * weight[None, :, :]


def log_pdet_from_jacobian(jacobians, eps=EPS):
    singular_values = torch.linalg.svdvals(jacobians)
    return torch.sum(torch.log(torch.clamp(singular_values, min=eps)), dim=1)


def _input_entropy_nats(input_width, n_in):
    widths = np.asarray(input_width, dtype=np.float64)
    if np.any(widths <= 0):
        raise ValueError("input_width must be positive.")
    if widths.ndim == 0:
        return n_in * math.log(float(widths))
    if widths.shape[0] != n_in:
        raise ValueError("input_width vector length must match layer input dimension.")
    return float(np.sum(np.log(widths)))


def continuous_layer_ei(weight, bias, samples, sigma, input_width=1.0, eps=EPS):
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    n_out, n_in = weight.shape
    input_entropy_nats = _input_entropy_nats(input_width, n_in)

    jacobians = sigmoid_layer_jacobian(weight, bias, samples)
    singular_values = torch.linalg.svdvals(jacobians)
    log_sv = torch.log(torch.clamp(singular_values, min=eps))
    log_pdet_mean = float(torch.mean(torch.sum(log_sv, dim=1)).item())

    term1_nats = -0.5 * n_out * math.log(2.0 * math.pi * math.e * sigma * sigma)
    term2_nats = input_entropy_nats + log_pdet_mean
    ei_nats = (term1_nats + term2_nats) / max(n_in, 1)

    # --- 噪声下限口径 -------------------------------------------------------
    # 每个奇异方向 k 的净贡献是 ln s_k - h_noise（nats）。当该方向信号弱于
    # 噪声底（ln s_k < h_noise）时，它其实什么也传不了，按定义应为 0 而不是
    # 负数，因此逐方向截断到 0，并且只对真实存在的 min(n_out, n_in) 个方向
    # 计噪声成本（n_out > n_in 时多出的输出维没有对应信号，不再收噪声）。
    log2 = math.log(2.0)
    h_noise = 0.5 * math.log(2.0 * math.pi * math.e * sigma * sigma)
    net_nats = log_sv - h_noise
    floored_nats = torch.clamp(net_nats, min=0.0)
    floor_gain = float(torch.sum(torch.clamp(-net_nats, min=0.0), dim=1).mean().item())
    ei_floored_nats = (input_entropy_nats + float(torch.sum(floored_nats, dim=1).mean().item())) / max(n_in, 1)
    tax_fix_nats = (n_out - int(log_sv.shape[1])) * h_noise / max(n_in, 1)

    return {
        "ei_bits": ei_nats / log2,
        "ei_bits_floored": ei_floored_nats / log2,
        "floor_gain_bits": floor_gain / log2,
        "tax_fix_bits": tax_fix_nats / log2,
        "term1_bits": term1_nats / log2,
        "term2_bits": term2_nats / log2,
        "log_pdet_mean_nats": log_pdet_mean,
        "sigma": float(sigma),
    }


def continuous_layer_sensitivity(weight, bias, samples, neuron_sigmas, input_width=1.0, eps=EPS):
    neuron_sigmas = np.asarray(neuron_sigmas, dtype=np.float64)
    n_out, n_in = weight.shape
    if neuron_sigmas.ndim == 0:
        neuron_sigmas = np.full(n_out, float(neuron_sigmas), dtype=np.float64)
    if neuron_sigmas.shape[0] != n_out:
        raise ValueError("neuron_sigmas length must match layer output dimension.")
    if np.any(neuron_sigmas <= 0):
        raise ValueError("neuron_sigmas must be positive.")

    input_entropy_nats = _input_entropy_nats(input_width, n_in)

    jacobians = sigmoid_layer_jacobian(weight, bias, samples)
    log_abs_edges = torch.log(torch.clamp(torch.abs(jacobians), min=eps))
    transmission_nats = float(torch.sum(torch.mean(log_abs_edges, dim=0)).item())

    term1_nats = -0.5 * float(np.sum(np.log(2.0 * math.pi * math.e * neuron_sigmas * neuron_sigmas)))
    sensitivity_nats = (term1_nats + input_entropy_nats + transmission_nats) / max(n_in, 1)
    log2 = math.log(2.0)
    return {
        "sensitivity_bits": sensitivity_nats / log2,
        "sens_term1_bits": term1_nats / log2,
        "sens_transmission_bits": transmission_nats / log2,
    }


def compute_model_layer_ei(model, sigma, n_samples, device, seed, first_layer_mins, first_layer_widths, neuron_sigmas=None):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    rows = []
    for idx, layer in enumerate(model.layers, start=1):
        n_in = layer.in_features
        unit_samples = torch.rand((n_samples, n_in), generator=generator, dtype=torch.float64, device=device)
        samples = unit_samples
        input_width = 1.0
        layer_name = f"L{idx}"
        layer_sigma = sigma[layer_name] if isinstance(sigma, dict) else sigma
        weight = layer.weight.detach()
        bias = layer.bias.detach() if layer.bias is not None else None
        metrics = continuous_layer_ei(
            weight,
            bias,
            samples,
            sigma=layer_sigma,
            input_width=input_width,
        )
        if neuron_sigmas is not None:
            layer_neuron_sigmas = neuron_sigmas[layer_name] if isinstance(neuron_sigmas, dict) else neuron_sigmas
            sens_metrics = continuous_layer_sensitivity(
                weight,
                bias,
                samples,
                neuron_sigmas=layer_neuron_sigmas,
                input_width=input_width,
            )
            sens_metrics["degeneracy_bits"] = sens_metrics["sensitivity_bits"] - metrics["ei_bits"]
            metrics.update(sens_metrics)
        rows.append(
            {
                "layer": layer_name,
                "n_in": n_in,
                "n_out": layer.out_features,
                "input_width": input_width,
                **metrics,
            }
        )
    return rows
