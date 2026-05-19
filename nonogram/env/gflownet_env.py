"""
GFlowNet Nonogram Environment — GFlowNet 학습을 위한 생성적 MDP 환경.
"""

import random
import numpy as np

from nonogram.env.state import hint_length
from nonogram.env.feasibility import is_feasible


class GFlowNetNonogramEnv:
    """GFlowNet용 2D NxN 노노그램 생성 환경.

    State: (channels, N, N) 텐서
        - ch0: row hint features (각 행의 hint 합계를 N으로 정규화하여 broadcast)
        - ch1: col hint features (각 열의 hint 합계를 N으로 정규화하여 broadcast)
        - ch2: filled mask (board == 1)
        - ch3: empty/X mask (board == -1)
        - ch4: unknown mask (board == 0)

    Action: flat index in [0, N * 2^N)
        - row_idx = action // 2^N
        - config_idx = action % 2^N (N-bit binary configuration)
    """

    def __init__(self, N: int, reward_temp: float = 1.0, epsilon: float = 1e-4,
                 early_stop: bool = True, pb_type: str = "uniform", pb_epsilon: float = 0.1):
        self.N = N
        self.reward_temp = reward_temp
        self.epsilon = epsilon
        self.early_stop = early_stop
        self.pb_type = pb_type
        self.pb_epsilon = pb_epsilon

        self.board = np.zeros((self.N, self.N), dtype=np.int64)
        self.gt_board = None
        self.row_hints = None
        self.col_hints = None
        self._row_hint_features = None
        self._col_hint_features = None

    @property
    def num_channels(self) -> int:
        return 5

    @property
    def action_dim(self) -> int:
        return self.N * (2 ** self.N)

    def reset(self, gt_board: np.ndarray | None = None,
              row_hints: list | None = None,
              col_hints: list | None = None) -> np.ndarray:
        """새 에피소드 시작. 보드를 비우고 힌트를 세팅."""
        if gt_board is not None:
            self.gt_board = gt_board.copy()
            self.row_hints = self._extract_runs_board(gt_board, axis="row")
            self.col_hints = self._extract_runs_board(gt_board, axis="col")
        elif row_hints is not None and col_hints is not None:
            self.gt_board = None
            self.row_hints = row_hints
            self.col_hints = col_hints
        else:
            # 랜덤 정답 보드를 생성하여 실현 가능한 힌트를 추출
            self.gt_board = np.random.choice([-1, 1], size=(self.N, self.N)).astype(np.int64)
            self.row_hints = self._extract_runs_board(self.gt_board, axis="row")
            self.col_hints = self._extract_runs_board(self.gt_board, axis="col")

        self.board = np.zeros((self.N, self.N), dtype=np.int64)
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
        """Action을 적용하여 행을 채운다.

        Returns:
            (next_state, reward, done)
        """
        row_idx = action // (2 ** self.N)
        config_idx = action % (2 ** self.N)

        if np.any(self.board[row_idx] != 0):
            # 이미 채워진 행에 대입하려고 할 경우 모순 처리 (페널티 보상)
            return self.get_state(), self.epsilon, True

        # N-bit binary config로 디코딩 (0 -> -1, 1 -> 1)
        config = np.array([1 if ((config_idx >> i) & 1) else -1 for i in range(self.N)], dtype=np.int64)
        self.board[row_idx, :] = config

        # 1. 조기 종료 검사 (Early Feasibility Check)
        if self.early_stop:
            if not self.check_trajectory_feasibility():
                # 논리적 모순이 발생하여 더 이상 완성 불가 -> epsilon 보상으로 즉시 종료
                return self.get_state(), self.epsilon, True

        # 2. Terminal Check (모든 행이 다 채워졌는지)
        filled_rows = np.any(self.board != 0, axis=1)
        if np.all(filled_rows):
            # 모든 행렬 만족 체크 및 보상 부여
            reward = self.compute_terminal_reward()
            return self.get_state(), reward, True

        return self.get_state(), 0.0, False

    def check_trajectory_feasibility(self) -> bool:
        """현재 부분 보드가 완성 가능한지 체크."""
        # 열 단위 Feasibility 체크
        for j in range(self.N):
            if not is_feasible(self.board[:, j], self.col_hints[j], self.N):
                return False
        # 채워진 행 단위 Feasibility 체크
        for i in range(self.N):
            if np.any(self.board[i] != 0):
                if not is_feasible(self.board[i], self.row_hints[i], self.N):
                    return False
        return True

    def compute_terminal_reward(self) -> float:
        """Terminal 상태에서 Clue 만족 여부로 보상 계산."""
        satisfied_lines = 0
        # 행 만족도 체크
        for i in range(self.N):
            if self._extract_runs(self.board[i]) == self.row_hints[i]:
                satisfied_lines += 1
        # 열 만족도 체크
        for j in range(self.N):
            if self._extract_runs(self.board[:, j]) == self.col_hints[j]:
                satisfied_lines += 1

        if satisfied_lines == 2 * self.N:
            # 모든 힌트를 만족하는 정답 완성
            return np.exp(self.reward_temp * satisfied_lines)
        else:
            # 모순이 존재하거나 불완전한 상태
            return self.epsilon

    def get_valid_mask(self) -> np.ndarray:
        """유효 action mask (길이 N * 2^N).

        이미 채워진 행에 대한 액션들은 모두 0으로 마스킹 처리.
        """
        mask = np.zeros(self.action_dim, dtype=np.float32)
        filled_rows = np.any(self.board != 0, axis=1)
        for r in range(self.N):
            if not filled_rows[r]:
                start_idx = r * (2 ** self.N)
                end_idx = (r + 1) * (2 ** self.N)
                mask[start_idx:end_idx] = 1.0
        return mask

    def get_backward_prob(self, prev_board: np.ndarray, curr_board: np.ndarray, cleared_row: int) -> float:
        """curr_board -> prev_board로의 Backward Transition 확률 P_B(prev | curr)를 계산."""
        filled_rows = np.any(curr_board != 0, axis=1)
        k = int(np.sum(filled_rows))  # curr_board에 채워진 행의 수
        if k == 0:
            return 1.0

        if self.pb_type == "uniform":
            return 1.0 / k

        # "칠해진 픽셀 수에 반비례"하는 경험적 분포
        # cleared_row의 칠해진 픽셀(1인 셀)의 개수 c_r를 세고, 1 / (c_r + pb_epsilon)로 가중치를 산출.
        weights = []
        target_weight = 0.0
        for r in range(self.N):
            if filled_rows[r]:
                c_r = int(np.sum(curr_board[r] == 1))
                w = 1.0 / (c_r + self.pb_epsilon)
                weights.append(w)
                if r == cleared_row:
                    target_weight = w
        
        sum_w = sum(weights)
        if sum_w == 0:
            return 1.0 / k
        return target_weight / sum_w

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
