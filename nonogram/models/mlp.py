"""
MLP Q-Network — 기존 QNetwork와 동일한 구조.

기존 체크포인트(qnet_N5.pt, qnet_N10.pt)와 호환됨.
"""

import torch.nn as nn

from nonogram.env.state import hint_length
from nonogram.models.registry import register_model


@register_model("mlp")
class MLPQNetwork(nn.Module):
    """MLP Q-network. Input: state (k + N), Output: Q-values over 2N actions.

    기존 utils.py의 QNetwork와 동일한 아키텍처.
    """

    def __init__(self, N: int, hidden_dim: int = 128, num_layers: int = 3, **kwargs):
        super().__init__()
        k = hint_length(N)
        input_dim = k + N
        output_dim = 2 * N

        layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
