"""
Mapper module: linear and MLP mappers for cross-model embedding space alignment.
Maps local encoder embeddings to server encoder embedding space.
"""

import torch
import torch.nn as nn


class LinearMapper(nn.Module):
    """Linear mapping: hat_e_s = W @ e_l + b"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPMapper(nn.Module):
    """Non-linear MLP mapping with configurable hidden layers."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: list = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 1024]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_mapper(cfg: dict, input_dim: int, output_dim: int) -> nn.Module:
    """Build mapper from config."""
    mapper_cfg = cfg["mapper"]
    mapper_type = mapper_cfg["type"]

    if mapper_type == "linear":
        return LinearMapper(input_dim, output_dim)
    elif mapper_type == "mlp":
        return MLPMapper(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=mapper_cfg.get("hidden_dims", [1024, 1024]),
            dropout=mapper_cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown mapper type: {mapper_type}")
