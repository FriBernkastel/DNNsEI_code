import math

import torch

from eicore.common import EPS


class GradientSigmaAccumulator:
    def __init__(self, layer_names, eps=EPS):
        self.eps = float(eps)
        self.stats = {
            name: {
                "sum": 0.0,
                "sum_sq": 0.0,
                "count": 0,
                "neuron_sum": None,
                "neuron_sum_sq": None,
                "neuron_count": 0,
            }
            for name in layer_names
        }

    def update(self, layer_grads):
        for name, grad in layer_grads.items():
            if name not in self.stats:
                continue
            grad = grad.detach().to(torch.float64)
            stat = self.stats[name]
            stat["sum"] += float(torch.sum(grad).item())
            stat["sum_sq"] += float(torch.sum(grad * grad).item())
            stat["count"] += int(grad.numel())
            flat = grad.reshape(-1, grad.shape[-1])
            if stat["neuron_sum"] is None:
                stat["neuron_sum"] = torch.zeros(flat.shape[1], dtype=torch.float64, device=flat.device)
                stat["neuron_sum_sq"] = torch.zeros(flat.shape[1], dtype=torch.float64, device=flat.device)
            stat["neuron_sum"] += torch.sum(flat, dim=0)
            stat["neuron_sum_sq"] += torch.sum(flat * flat, dim=0)
            stat["neuron_count"] += int(flat.shape[0])

    def sigmas(self):
        out = {}
        for name, stat in self.stats.items():
            if stat["count"] == 0:
                raise ValueError(f"Cannot compute sigma for {name} before gradients are accumulated.")
            mean = stat["sum"] / stat["count"]
            var = max(stat["sum_sq"] / stat["count"] - mean * mean, 0.0)
            out[name] = math.sqrt(var + self.eps)
        return out

    def per_neuron_sigmas(self):
        out = {}
        for name, stat in self.stats.items():
            if stat["neuron_count"] == 0:
                raise ValueError(f"Cannot compute per-neuron sigma for {name} before gradients are accumulated.")
            mean = stat["neuron_sum"] / stat["neuron_count"]
            var = torch.clamp(stat["neuron_sum_sq"] / stat["neuron_count"] - mean * mean, min=0.0)
            out[name] = torch.sqrt(var + self.eps).cpu().numpy()
        return out


class ResidualSigmaAccumulator:
    def __init__(self, eps=EPS):
        self.eps = float(eps)
        self.sum = 0.0
        self.sum_sq = 0.0
        self.count = 0

    def update(self, y_pred, y_target):
        residual = (y_pred.detach() - y_target.detach()).to(torch.float64)
        self.sum += float(torch.sum(residual).item())
        self.sum_sq += float(torch.sum(residual * residual).item())
        self.count += int(residual.numel())

    def sigma(self):
        if self.count == 0:
            raise ValueError("Cannot compute residual sigma before residuals are accumulated.")
        mean = self.sum / self.count
        var = max(self.sum_sq / self.count - mean * mean, 0.0)
        return math.sqrt(var + self.eps)
