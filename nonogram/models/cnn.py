"""
1D CNN Q-Network — 줄의 공간적 패턴을 Conv1d로 캡처.

Hint 부분과 Block 부분을 별도로 처리한 후 결합하여
local spatial patterns를 효과적으로 학습.
"""

import torch
import torch.nn as nn

from nonogram.env.state import hint_length
from nonogram.models.registry import register_model


@register_model("cnn")
class CNNQNetwork(nn.Module):
    """1D CNN Q-network.

    Block 부분에 Conv1d를 적용하여 local spatial patterns를 캡처하고,
    Hint 부분은 FC로 처리 후 결합.
    """

    def __init__(self, N: int, hidden_dim: int = 128, num_layers: int = 3, **kwargs):
        super().__init__()
        k = hint_length(N)
        self.N = N
        self.k = k
        output_dim = 2 * N

        # Hint encoder (FC)
        self.hint_encoder = nn.Sequential(
            nn.Linear(k, hidden_dim // 2),
            nn.ReLU(),
        )

        # Block encoder (Conv1d)
        # blocks: (batch, N) → (batch, 1, N) → Conv1d → (batch, channels, N')
        self.block_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        block_out_dim = 64 * N

        # Combiner
        combined_dim = hidden_dim // 2 + block_out_dim
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # Split hint and blocks
        hint = x[:, :self.k]                                   # (batch, k)
        blocks = x[:, self.k:].unsqueeze(1)                    # (batch, 1, N)

        hint_features = self.hint_encoder(hint)                 # (batch, hidden/2)
        block_features = self.block_encoder(blocks)             # (batch, 64, N)
        block_features = block_features.flatten(start_dim=1)    # (batch, 64*N)

        combined = torch.cat([hint_features, block_features], dim=1)
        return self.combiner(combined)
