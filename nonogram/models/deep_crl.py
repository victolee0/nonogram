"""
Deep Nonogram CRL Model — 1000층 스케일링이 가능한 대규모 대비 학습(Contrastive RL) 네트워크.

LayerScale, GroupNorm, GELU 및 잔차 연결을 도입하여 그래디언트를 안정적으로 흐르게 하며,
HER(Hindsight Experience Replay) 모드와 Clue(단서 매칭) 모드를 단일 구조로 완벽히 지원합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nonogram.models.registry import register_model


class LayerScale(nn.Module):
    """Deep Network의 잔차 분기 출력을 동적으로 조절하는 스케일 파라미터."""
    def __init__(self, dim: int, init_value: float = 1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim, 1, 1) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class DeepResNet2DBlock(nn.Module):
    """GroupNorm + GELU + LayerScale 기반의 초안정성 2D Residual block."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        # Pre-LN/Pre-GN 구조로 기울기 폭발/소실 방지
        self.norm1 = nn.GroupNorm(num_groups=max(1, dim // 16), num_channels=dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=max(1, dim // 16), num_channels=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        
        self.scale = LayerScale(dim, init_value=1e-3)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim, N, N)
        residual = x
        
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = self.dropout(x)
        
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.conv2(x)
        
        return residual + self.scale(x)


@register_model("deep_crl")
class DeepNonogramCRLNet(nn.Module):
    """대비 학습용 초심층 2D 보드 인코더 + Goal 매칭 네트워크."""

    def __init__(self, N: int, hidden_dim: int = 128, num_layers: int = 20, 
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.N = N
        self.dim = hidden_dim

        # --- 1. State Embeddings ---
        self.cell_emb = nn.Embedding(3, hidden_dim)
        self.hint_emb = nn.Embedding(N + 1, hidden_dim, padding_idx=0)
        self.hint_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        self.input_proj = nn.Linear(3 * hidden_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # --- 2. Deep ResNet Backbone ---
        self.blocks = nn.ModuleList([
            DeepResNet2DBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # --- 3. Action-conditioned State Representation Head ---
        # 2D 보드 표현을 2개 채널(paint, X)로 확장하여 행동별 표현 획득
        self.phi_head = nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size=1)

        # --- 4. Goal Encoders ---
        # (A) HER 모드용 Goal Encoder: 미래 보드 상태(B, N, N)를 임베딩 벡터(B, dim)로 변환
        self.her_goal_proj = nn.Linear(hidden_dim, hidden_dim)
        self.her_goal_blocks = nn.ModuleList([
            DeepResNet2DBlock(hidden_dim, dropout=dropout) for _ in range(3)
        ])
        
        # (B) Clue 모드용 Goal Encoder: 단서(Row, Col)를 임베딩 벡터(B, dim)로 변환
        self.clue_goal_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """기울기 전파 안정화를 위해 Orthogonal 및 Kaiming 초기화 수행."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def encode_state(self, board: torch.Tensor, row_hints: torch.Tensor, 
                     col_hints: torch.Tensor) -> torch.Tensor:
        """현재 상태 s를 인코딩하여 행동별 표현 Phi(s, a) 획득.
        
        Returns:
            phi_s_a: (B, 2, N, N, dim)
        """
        B = board.size(0)
        
        # 1. 셀 상태 임베딩
        board_mapped = board.clone()
        board_mapped[board == -1] = 2
        cell_features = self.cell_emb(board_mapped.long()) # (B, N, N, dim)

        # 2. 행 단서 임베딩
        r_hints_flat = row_hints.view(B * self.N, -1)
        r_emb = self.hint_emb(r_hints_flat)
        _, r_hidden = self.hint_rnn(r_emb)
        row_features = r_hidden.squeeze(0).view(B, self.N, 1, self.dim)
        row_features = row_features.expand(-1, -1, self.N, -1)

        # 3. 열 단서 임베딩
        c_hints_flat = col_hints.view(B * self.N, -1)
        c_emb = self.hint_emb(c_hints_flat)
        _, c_hidden = self.hint_rnn(c_emb)
        col_features = c_hidden.squeeze(0).view(B, 1, self.N, self.dim)
        col_features = col_features.expand(-1, self.N, -1, -1)

        # 4. 결합 및 투영
        combined = torch.cat([cell_features, row_features, col_features], dim=-1)
        x = self.input_norm(self.input_proj(combined)) # (B, N, N, dim)
        
        # Conv2D를 위해 채널 축 이동: (B, N, N, dim) -> (B, dim, N, N)
        x = x.permute(0, 3, 1, 2)

        # 5. Deep ResNet 통과
        for block in self.blocks:
            x = block(x)

        # 6. 행동별 표현으로 확장: (B, dim, N, N) -> (B, 2*dim, N, N)
        phi = self.phi_head(x)
        # (B, 2*dim, N, N) -> (B, 2, dim, N, N) -> (B, 2, N, N, dim)
        phi = phi.view(B, 2, self.dim, self.N, self.N).permute(0, 1, 3, 4, 2)
        return phi

    def encode_goal(self, goal_input: dict, crl_mode: str) -> torch.Tensor:
        """목표 g를 인코딩하여 글로벌 벡터 Psi(g) 획득.
        
        Returns:
            psi_g: (B, dim)
        """
        B = next(iter(goal_input.values())).size(0)

        if crl_mode == "her":
            # HER 모드: goal_input["future_board"] (B, N, N)
            future_board = goal_input["future_board"]
            board_mapped = future_board.clone()
            board_mapped[future_board == -1] = 2
            
            x = self.cell_emb(board_mapped.long()) # (B, N, N, dim)
            x = self.her_goal_proj(x).permute(0, 3, 1, 2) # (B, dim, N, N)
            
            for block in self.her_goal_blocks:
                x = block(x)
                
            # Global Average Pooling으로 벡터화
            psi_g = x.mean(dim=[2, 3]) # (B, dim)
            return psi_g
            
        elif crl_mode == "clue":
            # Clue 모드: goal_input["row_hints"] (B, N, H), goal_input["col_hints"] (B, N, H)
            row_hints = goal_input["row_hints"]
            col_hints = goal_input["col_hints"]
            
            # Row clues
            r_hints_flat = row_hints.view(B * self.N, -1)
            r_emb = self.hint_emb(r_hints_flat)
            _, r_hidden = self.hint_rnn(r_emb)
            r_vec = r_hidden.squeeze(0).view(B, self.N, self.dim).mean(dim=1) # (B, dim)

            # Col clues
            c_hints_flat = col_hints.view(B * self.N, -1)
            c_emb = self.hint_emb(c_hints_flat)
            _, c_hidden = self.hint_rnn(c_emb)
            c_vec = c_hidden.squeeze(0).view(B, self.N, self.dim).mean(dim=1) # (B, dim)
            
            # Concat & project
            combined = torch.cat([r_vec, c_vec], dim=-1) # (B, 2*dim)
            psi_g = self.clue_goal_proj(combined) # (B, dim)
            return psi_g
        else:
            raise ValueError(f"Unknown CRL mode: {crl_mode}")

    def forward(self, board: torch.Tensor, row_hints: torch.Tensor, col_hints: torch.Tensor, 
                goal_input: dict, crl_mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """행동 가치 Q(s, a, g) 및 표상 리턴.
        
        Q(s, a, g) = dot_product(Phi(s, a), Psi(g))
        
        Returns:
            q_values: (B, 2*N*N)
            phi_s_a: (B, 2, N, N, dim)
            psi_g: (B, dim)
        """
        B = board.size(0)
        
        # 1. 상태 표상 추출
        phi_s_a = self.encode_state(board, row_hints, col_hints) # (B, 2, N, N, dim)
        
        # 2. 목표 표상 추출
        psi_g = self.encode_goal(goal_input, crl_mode) # (B, dim)
        
        # 3. 내적 계산: Q(s, a, g) = Phi(s, a) * Psi(g)
        # (B, 2, N, N, dim) * (B, dim) -> (B, 2, N, N)
        q_values = torch.einsum("bijnd,bd->bijn", phi_s_a, psi_g)
        
        # 4. DQN 호환 액션 규격 (B, 2*N*N)으로 변환
        q_values = q_values.reshape(B, -1)
        
        return q_values, phi_s_a, psi_g


class DeepResNet1DBlock(nn.Module):
    """1D MLP 기반의 초안정성 Residual block."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.ones(dim) * 1e-3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim)
        residual = x
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.fc1(x)
        x = self.dropout(x)
        
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return residual + x * self.scale


@register_model("deep_line_crl")
class DeepLineCRLNet(nn.Module):
    """1D Line 환경용 초심층 대비 학습 모델."""

    def __init__(self, N: int, hidden_dim: int = 128, num_layers: int = 20, 
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.N = N
        self.k = (N + 1) // 2
        self.dim = hidden_dim

        # State embeddings
        self.cell_emb = nn.Embedding(3, hidden_dim)
        self.hint_emb = nn.Embedding(N + 1, hidden_dim, padding_idx=0)

        # Projection from (k + N) * dim to dim
        self.state_proj = nn.Linear((self.k + N) * hidden_dim, hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            DeepResNet1DBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Action representations: 2N actions
        self.phi_head = nn.Linear(hidden_dim, 2 * N * hidden_dim)

        # Goal encoders
        # HER: future state (B, k + N) -> only future blocks are goal
        self.her_goal_proj = nn.Linear(N * hidden_dim, hidden_dim)
        self.her_goal_blocks = nn.ModuleList([
            DeepResNet1DBlock(hidden_dim, dropout=dropout) for _ in range(3)
        ])

        # Clue: target hints (B, k)
        self.clue_goal_proj = nn.Linear(self.k * hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        # state: (B, k + N)
        B = state.size(0)
        hints = state[:, :self.k].long()
        blocks = state[:, self.k:].long()

        blocks_mapped = blocks.clone()
        blocks_mapped[blocks == -1] = 2 # -1 -> 2

        h_emb = self.hint_emb(hints).view(B, -1) # (B, k * dim)
        b_emb = self.cell_emb(blocks_mapped).view(B, -1) # (B, N * dim)

        combined = torch.cat([h_emb, b_emb], dim=-1) # (B, (k + N) * dim)
        x = self.state_norm(self.state_proj(combined)) # (B, dim)

        for block in self.blocks:
            x = block(x)

        # Output phi_s_a: (B, 2*N, dim)
        phi = self.phi_head(x).view(B, 2 * self.N, self.dim)
        return phi

    def encode_goal(self, goal_input: dict, crl_mode: str) -> torch.Tensor:
        # goal_input has: "future_state" (B, k + N) or "clues" (B, k)
        B = next(iter(goal_input.values())).size(0)
        
        if crl_mode == "her":
            future_state = goal_input["future_state"]
            future_blocks = future_state[:, self.k:].long()
            future_blocks_mapped = future_blocks.clone()
            future_blocks_mapped[future_blocks == -1] = 2

            b_emb = self.cell_emb(future_blocks_mapped).view(B, -1) # (B, N * dim)
            x = F.gelu(self.her_goal_proj(b_emb)) # (B, dim)
            for block in self.her_goal_blocks:
                x = block(x)
            return x
        elif crl_mode == "clue":
            # In 1D Line, goal is target clues from the state
            # which is just state[:, :k]
            state = goal_input["state"]
            hints = state[:, :self.k].long()
            h_emb = self.hint_emb(hints).view(B, -1) # (B, k * dim)
            psi_g = self.clue_goal_proj(h_emb)
            return psi_g
        else:
            raise ValueError(f"Unknown CRL mode: {crl_mode}")

    def forward(self, state: torch.Tensor, goal_input: dict, crl_mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi_s_a = self.encode_state(state) # (B, 2*N, dim)
        psi_g = self.encode_goal(goal_input, crl_mode) # (B, dim)

        # q_values = dot product between phi_s_a and psi_g
        # (B, 2*N, dim) * (B, dim) -> (B, 2*N)
        q_values = torch.einsum("bad,bd->ba", phi_s_a, psi_g)
        return q_values, phi_s_a, psi_g

