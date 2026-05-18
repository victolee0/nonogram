"""
2D 보드 환경 — 전체 NxN 보드를 state로 사용하는 RL 환경.

1D 줄 환경과 달리, 교차 줄 정보를 직접 활용할 수 있어
더 복잡한 추론이 가능하지만 state/action space가 훨씬 크다.
"""

import random
from typing import Optional

import numpy as np

from nonogram.env.state import encode_hint, hint_length


class BoardEnv:
    """2D NxN 노노그램 보드 환경.

    State: (channels, N, N) 텐서
        - ch0: row hint features (각 행의 hint를 행 전체에 broadcast)
        - ch1: col hint features (각 열의 hint를 열 전체에 broadcast)
        - ch2: filled mask (board == 1)
        - ch3: empty/X mask (board == -1)
        - ch4: unknown mask (board == 0)

    Action: flat index in [0, 2*N*N)
        - idx in [0, N*N): cell idx → paint (+1)
        - idx in [N*N, 2*N*N): cell idx → X (-1)
    """

    def __init__(self, N: int, early_stop: bool = True,
                 reward_shaping: bool = False, reward_shaping_weight: float = 0.1):
        self.N = N
        self.k = hint_length(N)
        self.early_stop = early_stop
        self.reward_shaping = reward_shaping
        self.reward_shaping_weight = reward_shaping_weight

        self.board: np.ndarray | None = None
        self.gt_board: np.ndarray | None = None
        self.row_hints: list | None = None
        self.col_hints: list | None = None
        self._row_hint_features: np.ndarray | None = None
        self._col_hint_features: np.ndarray | None = None

    @property
    def num_channels(self) -> int:
        return 5

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return (self.num_channels, self.N, self.N)

    @property
    def action_dim(self) -> int:
        return 2 * self.N * self.N

    def reset(self, gt_board: np.ndarray | None = None,
              row_hints: list | None = None,
              col_hints: list | None = None,
              reveal_ratio: float = 0.0) -> np.ndarray:
        """새 에피소드 시작.

        gt_board가 주어지면 그것에서 hints를 추출.
        row_hints/col_hints가 직접 주어지면 gt_board 없이도 사용 가능.
        아무것도 없으면 랜덤 보드 생성.
        """
        if gt_board is not None:
            self.gt_board = gt_board.copy()
            self.row_hints = self._extract_runs_board(gt_board, axis="row")
            self.col_hints = self._extract_runs_board(gt_board, axis="col")
        elif row_hints is not None and col_hints is not None:
            self.gt_board = None
            self.row_hints = row_hints
            self.col_hints = col_hints
        else:
            self.gt_board = np.random.choice([-1, 1], size=(self.N, self.N)).astype(np.int64)
            self.row_hints = self._extract_runs_board(self.gt_board, axis="row")
            self.col_hints = self._extract_runs_board(self.gt_board, axis="col")

        self.board = np.zeros((self.N, self.N), dtype=np.int64)

        # 역방향 커리큘럼 학습 (Reverse Curriculum): 정답 보드의 일부를 미리 공개
        if reveal_ratio > 0.0 and self.gt_board is not None:
            num_cells = self.N * self.N
            num_reveal = int(num_cells * reveal_ratio)
            if num_reveal > 0:
                # 무작위 셀 선택
                indices = np.random.choice(num_cells, size=num_reveal, replace=False)
                for idx in indices:
                    r = idx // self.N
                    c = idx % self.N
                    self.board[r, c] = self.gt_board[r, c]

        self._build_hint_features()
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """현재 state를 (channels, N, N) 형태로 반환."""
        state = np.zeros((self.num_channels, self.N, self.N), dtype=np.float32)
        state[0] = self._row_hint_features
        state[1] = self._col_hint_features
        state[2] = (self.board == 1).astype(np.float32)
        state[3] = (self.board == -1).astype(np.float32)
        state[4] = (self.board == 0).astype(np.float32)
        return state

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Action 적용.

        Returns:
            (next_state, reward, done)
        """
        cell_idx = action % (self.N * self.N)
        fill_value = 1 if action < self.N * self.N else -1
        row = cell_idx // self.N
        col = cell_idx % self.N

        if self.board[row, col] != 0:
            # 이미 결정된 셀에 action → 패널티 + 무효
            return self.get_state(), -0.1, False

        self.board[row, col] = fill_value

        # Terminal check
        if np.all(self.board != 0):
            reward = self._compute_terminal_reward()
            return self.get_state(), reward, True

        # Reward shaping
        reward = 0.0
        if self.reward_shaping and self.gt_board is not None:
            if self.board[row, col] == self.gt_board[row, col]:
                reward = self.reward_shaping_weight * (1.0 / (self.N * self.N))
            else:
                reward = -self.reward_shaping_weight * (1.0 / (self.N * self.N))

        return self.get_state(), reward, False

    def get_valid_mask(self) -> np.ndarray:
        """유효 action mask (길이 2*N*N)."""
        flat = self.board.flatten()
        mask = np.zeros(2 * self.N * self.N, dtype=np.float32)
        for i in range(self.N * self.N):
            if flat[i] == 0:
                mask[i] = 1.0               # paint
                mask[self.N * self.N + i] = 1.0  # X
        return mask

    def select_towards_action(self) -> int | None:
        """Ground truth를 향한 action 선택."""
        if self.gt_board is None:
            return None
        empty = np.argwhere(self.board == 0)
        if len(empty) == 0:
            return None
        idx = random.randint(0, len(empty) - 1)
        r, c = empty[idx]
        cell_idx = r * self.N + c
        gt_val = self.gt_board[r, c]
        if gt_val == 1:
            return cell_idx
        else:
            return self.N * self.N + cell_idx

    def _compute_terminal_reward(self) -> float:
        """Terminal state에서 모든 행/열 hint 일치 여부로 reward 계산."""
        for i in range(self.N):
            row_runs = self._extract_runs(self.board[i, :])
            if row_runs != self.row_hints[i]:
                return -1.0
        for j in range(self.N):
            col_runs = self._extract_runs(self.board[:, j])
            if col_runs != self.col_hints[j]:
                return -1.0
        return 1.0

    def _extract_runs(self, line: np.ndarray) -> list[int]:
        """줄에서 연속 +1의 run lengths 추출."""
        runs = []
        cur = 0
        for v in line:
            if int(v) == 1:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)
        return runs if runs else [0]

    def _extract_runs_board(self, board: np.ndarray, axis: str) -> list[list[int]]:
        """보드 전체에서 행 또는 열 hint 추출."""
        result = []
        for i in range(self.N):
            if axis == "row":
                line = board[i, :]
            else:
                line = board[:, i]
            result.append(self._extract_runs(line))
        return result

    def _build_hint_features(self) -> None:
        """Hint를 (N, N) feature map으로 변환.

        각 행/열의 hint 합계를 정규화하여 해당 행/열 전체에 broadcast.
        """
        self._row_hint_features = np.zeros((self.N, self.N), dtype=np.float32)
        self._col_hint_features = np.zeros((self.N, self.N), dtype=np.float32)

        for i, hints in enumerate(self.row_hints):
            total = sum(h for h in hints if h > 0)
            self._row_hint_features[i, :] = total / self.N

        for j, hints in enumerate(self.col_hints):
            total = sum(h for h in hints if h > 0)
            self._col_hint_features[:, j] = total / self.N
