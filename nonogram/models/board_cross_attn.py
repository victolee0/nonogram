"""
2D Board Cross-Attention Q-network.

Row-Column axial self-attention architecture tailored for learning
nonogram row and column constraint propagation without AC-3.
"""

import torch
import torch.nn as nn

from nonogram.models.registry import register_model
from nonogram.env.state import hint_length


class RowColAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.col_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim)
        )
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, dim, N, N)
        B, D, N, _ = x.shape
        
        # 1. Row Attention (attend across columns)
        r_in = x.permute(0, 2, 3, 1).reshape(B * N, N, D)
        r_out, _ = self.row_attn(r_in, r_in, r_in)
        r_out = self.norm1(r_in + r_out)
        x = r_out.view(B, N, N, D).permute(0, 3, 1, 2)
        
        # 2. Col Attention (attend across rows)
        c_in = x.permute(0, 3, 2, 1).reshape(B * N, N, D)
        c_out, _ = self.col_attn(c_in, c_in, c_in)
        c_out = self.norm2(c_in + c_out)
        x = c_out.view(B, N, N, D).permute(0, 3, 2, 1)
        
        # 3. Feed Forward
        flat = x.permute(0, 2, 3, 1).reshape(B, N * N, D)
        ffn_out = self.ffn(flat)
        flat = self.norm3(flat + ffn_out)
        x = flat.view(B, N, N, D).permute(0, 3, 1, 2)
        
        return x


@register_model("board_cross_attn")
class BoardCrossAttnQNetwork(nn.Module):
    """2D Row-Column Cross-Attention Q-network for board-level RL."""

    def __init__(self, N: int, hidden_dim: int = 128, num_layers: int = 3, num_heads: int = 4, **kwargs):
        super().__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        
        # 힌트 인코더용 Embedding & GRU
        self.hint_dim = hidden_dim // 4
        self.hint_emb = nn.Embedding(N + 1, self.hint_dim, padding_idx=0)
        self.hint_rnn = nn.GRU(self.hint_dim, self.hint_dim, batch_first=True)
        
        # 5개 또는 7개 보드 채널 + 2 * 힌트 채널
        # C는 dynamic하게 처리하기 위해 __init__ 시점에 채널 수를 짐작하지 않고
        # input_proj를 lazy/첫 forward에 생성하거나 적당히 지원 (일반적으로 5 또는 7 채널)
        # 넉넉하게 16채널까지 커버 가능하도록 2D Conv를 유연하게 구성
        self.num_hint_features = 2 * self.hint_dim
        
        # First layer projection
        # 입력 채널이 5(기본) or 7(progress 포함)이므로 넉넉하게 dynamic projection 혹은 7채널 기준
        # 보드 채널 수 (C) + 힌트 임베딩 채널 수
        self.num_channels_estimate = 7 + self.num_hint_features
        self.input_proj = nn.Conv2d(self.num_channels_estimate, hidden_dim, kernel_size=1)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            RowColAttentionBlock(hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
        # Q-value output head (1x1 Conv predicting 2 values per cell)
        self.q_head = nn.Conv2d(hidden_dim, 2, kernel_size=1)

    def forward(self, x, row_hints=None, col_hints=None):
        B, C, N, _ = x.shape
        
        # 힌트 텐서가 누락된 경우 zero-padding 처리
        if row_hints is None or col_hints is None:
            max_len = hint_length(N)
            row_hints = torch.zeros((B, N, max_len), dtype=torch.long, device=x.device)
            col_hints = torch.zeros((B, N, max_len), dtype=torch.long, device=x.device)
        
        # 텐서 변환
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

        # C가 5인 경우, 나머지 2개 채널을 0으로 채워 7채널로 맞춰줌 (progress 채널이 없을 때의 하위 호환)
        if C < 7:
            padding_ch = torch.zeros((B, 7 - C, N, N), dtype=x.dtype, device=x.device)
            x_padded = torch.cat([x, padding_ch], dim=1)
        else:
            x_padded = x[:, :7, :, :]

        # Combine features
        combined = torch.cat([x_padded, row_features, col_features], dim=1)
        
        # Project to hidden_dim
        out = self.input_proj(combined)
        
        # Run RowColAttention layers
        for block in self.blocks:
            out = block(out)
            
        # Q-values per cell
        q_cell = self.q_head(out)  # (B, 2, N, N)
        
        # Flatten matching BoardEnv action indexes
        paint_q = q_cell[:, 0, :, :].flatten(start_dim=1)  # (B, N*N)
        empty_q = q_cell[:, 1, :, :].flatten(start_dim=1)  # (B, N*N)
        
        return torch.cat([paint_q, empty_q], dim=1)  # (B, 2*N*N)
