"""
Symmetry-Aware Bipartite GNN — D4 대칭성을 보존하는 노노그램 이분 그래프 신경망.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nonogram.models.registry import register_model


class SymmetryGNNLayer(nn.Module):
    """행과 열 노드 간에 동변성(Equivariant)을 갖는 이분 그래프 레이어."""

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        # 대칭형 에지 갱신 (행/열 정보를 동일한 선형 레이어로 프로젝션 후 더함)
        self.node_proj = nn.Linear(node_dim, edge_dim)
        self.edge_proj = nn.Linear(edge_dim, edge_dim)
        self.edge_update = nn.Sequential(
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        # 공유 노드 갱신 (행과 열 노드가 동일한 파라미터를 공유)
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
        )

    def forward(self, row_nodes: torch.Tensor, col_nodes: torch.Tensor, edge_features: torch.Tensor):
        """
        row_nodes: (B, N, node_dim)
        col_nodes: (B, N, node_dim)
        edge_features: (B, N, N, edge_dim)
        """
        B, N, _ = row_nodes.shape

        # 1. Edge Feature Update
        r_expanded = row_nodes.unsqueeze(2).expand(-1, -1, N, -1)   # (B, N, N, node_dim)
        c_expanded = col_nodes.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, N, node_dim)

        node_contrib = self.node_proj(r_expanded) + self.node_proj(c_expanded)
        edge_contrib = self.edge_proj(edge_features)
        new_edge_features = self.edge_update(node_contrib + edge_contrib)

        # 2. Row Node Update (열 방향 에지 메시지 취합)
        row_msg = new_edge_features.mean(dim=2)                     # (B, N, edge_dim)
        new_row_nodes = self.node_update(torch.cat([row_nodes, row_msg], dim=-1))

        # 3. Col Node Update (행 방향 에지 메시지 취합)
        col_msg = new_edge_features.mean(dim=1)                     # (B, N, edge_dim)
        new_col_nodes = self.node_update(torch.cat([col_nodes, col_msg], dim=-1))

        return new_row_nodes, new_col_nodes, new_edge_features


@register_model("symmetry_gnn")
class SymmetryGNNNetwork(nn.Module):
    """노노그램 보드의 D4 대칭군을 고려한 이분 그래프 신경망 (Symmetry-Aware Bipartite GNN)."""

    def __init__(self, N: int, hidden_dim: int = 128, num_layers: int = 3, **kwargs):
        super().__init__()
        self.N = N
        self.hidden_dim = hidden_dim

        # 힌트 및 에지 임베딩 레이어 공유
        self.node_init = nn.Linear(1, hidden_dim)
        self.edge_init = nn.Linear(3, hidden_dim)

        # GNN 레이어 스택
        self.gnn_layers = nn.ModuleList([
            SymmetryGNNLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # 최종 행별 configurations 출력 헤드
        out_dim = 2 ** N
        self.row_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # 수평 반전(Horizontal Flip) 시 configuration의 bit reverse 인덱스 미리 계산
        bit_rev_indices = []
        for c in range(out_dim):
            rev = 0
            for i in range(N):
                if (c >> i) & 1:
                    rev |= (1 << (N - 1 - i))
            bit_rev_indices.append(rev)
        self.register_buffer("bit_reverse_indices", torch.tensor(bit_rev_indices, dtype=torch.long))

    def forward_base(self, x: torch.Tensor) -> torch.Tensor:
        """Symmetrization을 제외한 기본 순전파."""
        B, C, N, _ = x.shape

        # 1. 노드 및 에지 초기화
        row_hints = x[:, 0, :, 0].unsqueeze(-1)
        row_nodes = self.node_init(row_hints)

        col_hints = x[:, 1, 0, :].unsqueeze(-1)
        col_nodes = self.node_init(col_hints)

        edges_in = x[:, 2:5].permute(0, 2, 3, 1)  # (B, N, N, 3)
        edge_features = self.edge_init(edges_in)

        # 2. GNN 레이어 순회
        for layer in self.gnn_layers:
            row_nodes, col_nodes, edge_features = layer(row_nodes, col_nodes, edge_features)

        # 3. Output logits 산출: (B, N, 2^N)
        logits = self.row_head(row_nodes)
        return logits

    def forward(self, x: torch.Tensor, symmetrize: bool = True) -> torch.Tensor:
        """대칭 변환을 고려한 로짓 산출.

        symmetrize=True 인 경우, 4개의 행 보존 대칭 변환(Identity, Horizontal, Vertical, Rot180)의
        출력을 평균하여 완벽한 동변성(Equivariance)을 보장합니다.
        """
        if not symmetrize:
            return self.forward_base(x)

        B, C, N, _ = x.shape

        # 대칭 변환 생성
        x0 = x
        x1 = torch.flip(x, dims=(-1,))       # Horizontal Flip
        x2 = torch.flip(x, dims=(-2,))       # Vertical Flip
        x3 = torch.flip(x, dims=(-2, -1))    # Rotate 180

        # 배치 형태로 묶어 한 번에 추론
        x_all = torch.stack([x0, x1, x2, x3], dim=1)  # (B, 4, C, N, N)
        x_flat = x_all.view(B * 4, C, N, N)

        logits_flat = self.forward_base(x_flat)       # (B * 4, N, 2^N)
        logits_all = logits_flat.view(B, 4, N, -1)

        L0 = logits_all[:, 0]
        L1 = logits_all[:, 1]
        L2 = logits_all[:, 2]
        L3 = logits_all[:, 3]

        # 역변환 적용
        L1_back = L1[:, :, self.bit_reverse_indices]
        L2_back = torch.flip(L2, dims=(1,))
        L3_back = torch.flip(L3[:, :, self.bit_reverse_indices], dims=(1,))

        # 평균 계산
        logits_avg = (L0 + L1_back + L2_back + L3_back) / 4.0
        return logits_avg
