"""
2D Board CNN — 전체 NxN 보드를 입력으로 받는 CNN Q-network.

BoardEnv의 multi-channel state (5, N, N)을 입력으로,
2*N*N 크기의 action Q-values를 출력.
"""

import torch
import torch.nn as nn

from nonogram.models.registry import register_model
from nonogram.env.state import hint_length


@register_model("board_cnn")
class BoardCNNQNetwork(nn.Module):
    """2D CNN Q-network for board-level RL.

    Input: (batch, 5, N, N) — [row_hint, col_hint, filled, empty, unknown]
    Output: (batch, 2*N*N) — Q-values for each (cell, action) pair
    """

    def __init__(self, N: int, hidden_dim: int = 128, num_layers: int = 3, **kwargs):
        super().__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        
        # 힌트 인코더용 Embedding & GRU
        self.hint_dim = hidden_dim // 4
        self.hint_emb = nn.Embedding(N + 1, self.hint_dim, padding_idx=0)
        self.hint_rnn = nn.GRU(self.hint_dim, self.hint_dim, batch_first=True)
        
        # 5개 보드 채널 + 2 * 힌트 채널
        num_channels = 5 + 2 * self.hint_dim
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

    def forward(self, x, row_hints=None, col_hints=None):
        B, C, N, _ = x.shape
        
        # 힌트 텐서가 누락된 경우(레거시/1D 호환용) zero-padding 처리
        if row_hints is None or col_hints is None:
            max_len = hint_length(N)
            row_hints = torch.zeros((B, N, max_len), dtype=torch.long, device=x.device)
            col_hints = torch.zeros((B, N, max_len), dtype=torch.long, device=x.device)
        
        # row_hints, col_hints가 list 등의 형식인 경우 텐서 변환
        if not isinstance(row_hints, torch.Tensor):
            row_hints = torch.tensor(row_hints, dtype=torch.long, device=x.device)
        if not isinstance(col_hints, torch.Tensor):
            col_hints = torch.tensor(col_hints, dtype=torch.long, device=x.device)
            
        # Process Row Hints: (B, N, max_len) -> (B * N, max_len)
        r_hints_flat = row_hints.view(B * self.N, -1)
        r_emb = self.hint_emb(r_hints_flat)
        _, r_hidden = self.hint_rnn(r_emb)
        row_features = r_hidden.squeeze(0).view(B, self.N, 1, self.hint_dim)
        row_features = row_features.expand(-1, -1, self.N, -1)  # (B, N, N, hint_dim)
        row_features = row_features.permute(0, 3, 1, 2)  # (B, hint_dim, N, N)

        # Process Column Hints: (B, N, max_len) -> (B * N, max_len)
        c_hints_flat = col_hints.view(B * self.N, -1)
        c_emb = self.hint_emb(c_hints_flat)
        _, c_hidden = self.hint_rnn(c_emb)
        col_features = c_hidden.squeeze(0).view(B, 1, self.N, self.hint_dim)
        col_features = col_features.expand(-1, self.N, -1, -1)  # (B, N, N, hint_dim)
        col_features = col_features.permute(0, 3, 1, 2)  # (B, hint_dim, N, N)

        # Combine all features for 2D convolution
        combined = torch.cat([x, row_features, col_features], dim=1)  # (B, 5 + 2*hint_dim, N, N)
        
        features = self.features(combined)              # (batch, 64, N, N)
        flat = features.flatten(start_dim=1)             # (batch, 64*N*N)
        return self.head(flat)

