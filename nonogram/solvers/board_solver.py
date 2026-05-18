"""
Board Solver — 2D Board-level Q-network를 사용한 풀이기.

전체 보드를 state로 보고, Q값이 가장 높은 셀을 순차적으로 채워나가는 방식.
"""

import numpy as np
import torch

from nonogram.env.board_env import BoardEnv
from nonogram.solvers.registry import register_solver


@register_solver("board")
class BoardSolver:
    """2D Board CNN 기반 풀이기."""

    def __init__(self, config: dict, q_net=None, device=None, **kwargs):
        self.config = config
        self.solver_cfg = config["solver"]
        self.q_net = q_net
        self.device = device or torch.device("cpu")

    def solve(self, N: int, row_hints: list, col_hints: list,
              verbose: bool = False) -> np.ndarray:
        """NxN 노노그램 풀이.

        전체 보드 state에서 Q-network가 가장 확신하는 셀부터 채워나감.
        """
        max_steps = N * N
        env = BoardEnv(N=N)
        env.reset(row_hints=row_hints, col_hints=col_hints)

        for step_i in range(max_steps):
            state = env.get_state()
            mask = env.get_valid_mask()

            if not np.any(mask > 0):
                break

            state_t = torch.tensor(state, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_t).squeeze(0).cpu().numpy()

            # 무효 action 마스킹
            q_values = np.where(mask > 0, q_values, -np.inf)
            action = int(np.argmax(q_values))

            _, _, done = env.step(action)

            if verbose:
                print(f"Step {step_i + 1}: action={action}, Q={q_values[action]:.4f}")

            if done:
                break

        return env.board
