import math

import torch
import torch.nn as nn


class IrisSigmoidNet(nn.Module):
    def __init__(self, dims=(4, 5, 5, 3)):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(len(dims) - 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            limit = 1.0 / math.sqrt(layer.in_features)
            nn.init.uniform_(layer.weight, -limit, limit)

    def forward(self, x):
        for layer in self.layers:
            x = torch.sigmoid(layer(x))
        return x

    def forward_with_activations(self, x):
        activations = []
        for layer in self.layers:
            x = torch.sigmoid(layer(x))
            x.retain_grad()
            activations.append(x)
        return x, activations


def evaluate_mse(model, x, y, loss_fn):
    model.eval()
    with torch.no_grad():
        return float(loss_fn(model(x), y).item())
