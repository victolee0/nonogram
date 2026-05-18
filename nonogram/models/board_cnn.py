"""
2D Board CNN — 전체 NxN 보드를 입력으로 받는 CNN Q-network.

BoardEnv의 multi-channel state (5, N, N)을 입력으로,
2*N*N 크기의 action Q-values를 출력.
"""

import torch
import torch.nn as nn

from nonogram.models.registry import register_model


@register_model("board_cnn")
class BoardCNNQNetwork(nn.Module):
    """2D CNN Q-network for board-level RL.

    Input: (batch, 5, N, N) — [row_hint, col_hint, filled, empty, unknown]
    Output: (batch, 2*N*N) — Q-values for each (cell, action) pair
    """

    def __init__(self, N: int, hidden_dim: int = 128, num_layers: int = 3, **kwargs):
        super().__init__()
        self.N = N
        num_channels = 5
        output_dim = 2 * N * N

        # CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        cnn_out_dim = 64 * N * N

        # FC head
        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        features = self.features(x)                     # (batch, 64, N, N)
        flat = features.flatten(start_dim=1)             # (batch, 64*N*N)
        return self.head(flat)
