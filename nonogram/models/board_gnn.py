"""
2D Board GNN — 노노그램을 행과 열의 이분 그래프(Bipartite Graph)로 모델링하는 그래프 신경망.

행(Row) 노드 N개와 열(Col) 노드 N개 사이를 연결하는 N*N개의 셀(Cell)을 에지(Edge)로 정의하고,
행-열 간의 상호 제약 조건을 메시지 패싱(Message Passing)으로 해결합니다.
"""

import torch
import torch.nn as nn

from nonogram.models.registry import register_model


class BipartiteGNNLayer(nn.Module):
    """행과 열 노드 간의 메시지 패싱을 수행하는 이분 그래프 레이어."""

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        # 에지 피처 갱신: (행 노드 + 열 노드 + 이전 에지 피처) -> 새 에지 피처
        self.edge_update = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        # 행 노드 갱신: (이전 행 노드 + 취합된 열 방향 에지 메시지) -> 새 행 노드
        self.row_update = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
        )
        # 열 노드 갱신: (이전 열 노드 + 취합된 행 방향 에지 메시지) -> 새 열 노드
        self.col_update = nn.Sequential(
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

        # 1. Edge Feature Update (각 셀에 행/열 노드 정보 전파)
        r_expanded = row_nodes.unsqueeze(2).expand(-1, -1, N, -1)   # (B, N, N, dim)
        c_expanded = col_nodes.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, N, dim)

        combined_edge = torch.cat([r_expanded, c_expanded, edge_features], dim=-1)
        new_edge_features = self.edge_update(combined_edge)         # (B, N, N, edge_dim)

        # 2. Row Node Update (각 행 i에 대해 모든 열 j의 셀 정보 취합)
        row_msg = new_edge_features.mean(dim=2)                     # (B, N, edge_dim)
        new_row_nodes = self.row_update(torch.cat([row_nodes, row_msg], dim=-1))

        # 3. Col Node Update (각 열 j에 대해 모든 행 i의 셀 정보 취합)
        col_msg = new_edge_features.mean(dim=1)                     # (B, N, edge_dim)
        new_col_nodes = self.col_update(torch.cat([col_nodes, col_msg], dim=-1))

        return new_row_nodes, new_col_nodes, new_edge_features


@register_model("board_gnn")
class BoardGNNNetwork(nn.Module):
    """노노그램 풀이용 2D 이분 그래프 신경망 (Bipartite GNN).

    DQN Q-network 및 PPO 백본 피처 인코더로 공용 활용할 수 있도록 설계되었습니다.
    """

    def __init__(self, N: int, hidden_dim: int = 128, num_layers: int = 3, **kwargs):
        super().__init__()
        self.N = N
        self.hidden_dim = hidden_dim

        # 힌트 및 에지 원핫 상태를 임베딩 공간으로 투영
        self.row_init = nn.Linear(1, hidden_dim)
        self.col_init = nn.Linear(1, hidden_dim)
        self.edge_init = nn.Linear(3, hidden_dim)  # [filled, empty, unknown] 마스크 채널

        # GNN 메시지 패싱 레이어 스택
        self.gnn_layers = nn.ModuleList([
            BipartiteGNNLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # 최종 에지(셀 착수) 별 액션 Q-value / 로짓 출력 헤드
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # [paint (+1), X (-1)] 출력
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 5, N, N) 입력 텐서
        """
        B, C, N, _ = x.shape

        # 1. 이분 그래프 초기화
        # 행 노드: row_hints (B, N, 1) -> (B, N, hidden)
        row_hints = x[:, 0, :, 0].unsqueeze(-1)
        row_nodes = self.row_init(row_hints)

        # 열 노드: col_hints (B, N, 1) -> (B, N, hidden)
        col_hints = x[:, 1, 0, :].unsqueeze(-1)
        col_nodes = self.col_init(col_hints)

        # 에지 피처: [filled, empty, unknown] 원핫 3채널 -> (B, N, N, hidden)
        edges_in = x[:, 2:5].permute(0, 2, 3, 1)  # (B, N, N, 3)
        edge_features = self.edge_init(edges_in)

        # 2. Bipartite GNN Message Passing 순회
        for layer in self.gnn_layers:
            row_nodes, col_nodes, edge_features = layer(row_nodes, col_nodes, edge_features)

        # 3. Output logits 산출
        # edge_features: (B, N, N, hidden) -> (B, N, N, 2)
        logits = self.edge_head(edge_features)

        # Flat action space (2*N*N)에 맞게 형상 변경
        # (B, N, N, 2) -> (B, 2, N, N) -> (B, 2*N*N)
        logits = logits.permute(0, 3, 1, 2)  # (B, 2, N, N)

        # index 0 ~ N*N-1: paint (+1) Q-values
        # index N*N ~ 2*N*N-1: X (-1) Q-values
        return logits.reshape(B, -1)
