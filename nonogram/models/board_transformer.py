"""
2D Board Transformer — Axial Attention & Global Attention 기반 DQN Q-network.

models_2d.py의 NonogramTransformerBlock을 활용하여,
BoardEnv의 (5, N, N) 상태 채널을 100% 활용하며 2*N*N의 Q-value를 직접 예측합니다.
"""

import torch
import torch.nn as nn

from nonogram.models.registry import register_model
from models_2d import NonogramTransformerBlock
from nonogram.env.state import hint_length


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

        # 힌트 인코더용 Embedding & GRU
        self.hint_dim = hidden_dim
        self.hint_emb = nn.Embedding(N + 1, self.hint_dim, padding_idx=0)
        self.hint_rnn = nn.GRU(self.hint_dim, self.hint_dim, batch_first=True)

        # 상태 피처 + Row 힌트 피처 + Col 힌트 피처 융합용 투영 레이어
        self.input_proj = nn.Linear(3 * hidden_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

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

    def forward(self, x: torch.Tensor, row_hints=None, col_hints=None) -> torch.Tensor:
        """
        x: (B, 5, N, N) 입력 텐서
        """
        B, C, N, _ = x.shape

        # 1. 채널 축을 맨 뒤로 돌림: (B, 5, N, N) -> (B, N, N, 5)
        x_permuted = x.permute(0, 2, 3, 1)

        # 2. 임베딩 공간 투영 및 정규화
        x_emb = self.state_proj(x_permuted)
        x_emb = self.proj_norm(x_emb)

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
        row_features = row_features.expand(-1, -1, self.N, -1)  # (B, N, N, hidden_dim)

        # Process Column Hints: (B, N, max_len) -> (B * N, max_len)
        c_hints_flat = col_hints.view(B * self.N, -1)
        c_emb = self.hint_emb(c_hints_flat)
        _, c_hidden = self.hint_rnn(c_emb)
        col_features = c_hidden.squeeze(0).view(B, 1, self.N, self.hint_dim)
        col_features = col_features.expand(-1, self.N, -1, -1)  # (B, N, N, hidden_dim)

        # Combine all features for Axial & Global Attention Transformer
        combined = torch.cat([x_emb, row_features, col_features], dim=-1)  # (B, N, N, 3*hidden_dim)
        x_emb = self.input_norm(self.input_proj(combined))  # (B, N, N, hidden_dim)

        # 3. Axial 및 Global Attention Transformer 순방향 연산
        for block in self.blocks:
            x_emb = block(x_emb)

        # 4. 각 셀별 액션 로짓 도출: (B, N, N, hidden_dim) -> (B, N, N, 2)
        q_out = self.q_head(x_emb)

        # 5. DQN 평탄화 액션 규격 (B, 2*N*N)으로 조율
        # (B, N, N, 2) -> (B, 2, N, N) -> (B, 2*N*N)
        q_out = q_out.permute(0, 3, 1, 2)
        return q_out.reshape(B, -1)

