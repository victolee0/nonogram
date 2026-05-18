"""
Dueling DQN Architecture.

Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

Value stream과 Advantage stream을 분리하여,
state의 절대 가치와 action 간 상대적 이점을 독립적으로 학습.
"""

import torch
import torch.nn as nn

from nonogram.env.state import hint_length
from nonogram.models.registry import register_model


@register_model("dueling")
class DuelingQNetwork(nn.Module):
    """Dueling DQN Q-network.

    공유 feature extractor 후 Value/Advantage 두 stream으로 분기.
    """

    def __init__(self, N: int, hidden_dim: int = 128, num_layers: int = 3, **kwargs):
        super().__init__()
        k = hint_length(N)
        input_dim = k + N
        output_dim = 2 * N

        # 공유 feature extractor
        shared_layers = []
        prev_dim = input_dim
        for _ in range(num_layers - 1):
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.shared = nn.Sequential(*shared_layers)

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        features = self.shared(x)
        value = self.value_stream(features)           # (batch, 1)
        advantage = self.advantage_stream(features)   # (batch, 2N)
        # Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
