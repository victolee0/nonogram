"""
2D Board Transformer — Axial Attention & Global Attention 기반 DQN Q-network.

models_2d.py의 NonogramTransformerBlock을 활용하여,
BoardEnv의 (5, N, N) 상태 채널을 100% 활용하며 2*N*N의 Q-value를 직접 예측합니다.
"""

import torch
import torch.nn as nn

from nonogram.models.registry import register_model
from models_2d import NonogramTransformerBlock


@register_model("board_transformer")
class BoardTransformerQNetwork(nn.Module):
    """Axial Attention + Global Attention 기반 2D 보드 트랜스포머 Q-network.

    Input: (B, 5, N, N) — [row_hint, col_hint, filled, empty, unknown]
    Output: (B, 2*N*N) — 각 셀의 (paint, X) 착수 가치(Q-value)
    """

    def __init__(self, N: int, hidden_dim: int = 128, num_layers: int = 3, num_heads: int = 4, **kwargs):
        super().__init__()
        self.N = N
        self.hidden_dim = hidden_dim

        # 5개 채널 피처를 임베딩 투영: (B, N, N, 5) -> (B, N, N, hidden_dim)
        self.state_proj = nn.Linear(5, hidden_dim)
        self.proj_norm = nn.LayerNorm(hidden_dim)

        # Axial & Global Attention Transformer 스택
        self.blocks = nn.ModuleList([
            NonogramTransformerBlock(hidden_dim, num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])

        # 각 셀 단위 착수 가치 예측 헤드 (paint, X)
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 5, N, N) 입력 텐서
        """
        B, C, N, _ = x.shape

        # 1. 채널 축을 맨 뒤로 돌림: (B, 5, N, N) -> (B, N, N, 5)
        x_permuted = x.permute(0, 2, 3, 1)

        # 2. 임베딩 공간 투영 및 정규화
        x_emb = self.state_proj(x_permuted)
        x_emb = self.proj_norm(x_emb)

        # 3. Axial 및 Global Attention Transformer 순방향 연산
        for block in self.blocks:
            x_emb = block(x_emb)

        # 4. 각 셀별 액션 로짓 도출: (B, N, N, hidden_dim) -> (B, N, N, 2)
        q_out = self.q_head(x_emb)

        # 5. DQN 평탄화 액션 규격 (B, 2*N*N)으로 조율
        # (B, N, N, 2) -> (B, 2, N, N) -> (B, 2*N*N)
        q_out = q_out.permute(0, 3, 1, 2)
        return q_out.reshape(B, -1)
