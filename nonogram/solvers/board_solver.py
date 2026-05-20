"""
Board Solver — 2D Board-level Q-network를 사용한 풀이기.

전체 보드를 state로 보고, Q-network의 Q값에 기반한 DFS 백트래킹을 수행하여
교착 상태를 타개하고 정답을 탐색합니다.
"""

import numpy as np
import torch

from nonogram.env.board_env import BoardEnv
from nonogram.env.feasibility import is_feasible
from nonogram.env.state import hint_length
from nonogram.solvers.registry import register_solver


@register_solver("board")
class BoardSolver:
    """2D Board Q-network 기반 풀이기 + 백트래킹 DFS."""

    def __init__(self, config: dict, q_net=None, device=None, **kwargs):
        self.config = config
        self.solver_cfg = config["solver"]
        self.q_net = q_net
        self.device = device or torch.device("cpu")

    def solve(self, N: int, row_hints: list, col_hints: list,
              verbose: bool = False) -> np.ndarray:
        """NxN 노노그램 풀이.

        Q-network의 Q-value가 가장 높은 셀의 결정을 우선 탐색하는 DFS 백트래킹.
        """
        neg_threshold = self.solver_cfg.get("neg_threshold", -0.5)
        max_backtracks = self.solver_cfg.get("max_backtracks", 1000)

        row_hints_clean = [[int(x) for x in h if int(x) > 0] for h in row_hints]
        col_hints_clean = [[int(x) for x in h if int(x) > 0] for h in col_hints]

        model_type = self.config["model"]["type"]
        is_2d_board_model = model_type in ("board_cnn", "board_transformer", "board_cross_attn")

        env = BoardEnv(N=N)
        env.reset(row_hints=row_hints, col_hints=col_hints)

        self.search_calls = 0
        self.backtrack_count = 0
        self.aborted = False
        self.best_feasible_board = env.board.copy()
        self.min_unknowns = N * N

        def search(board, force_count):
            self.search_calls += 1
            if self.backtrack_count >= max_backtracks:
                self.aborted = True
                return None

            local_board = board.copy()
            env.board = local_board.copy()

            # 종료 조건 검사
            if (local_board != 0).all():
                match = True
                for r in range(N):
                    if not is_feasible(local_board[r, :], row_hints_clean[r], N):
                        match = False
                        break
                if match:
                    for c in range(N):
                        if not is_feasible(local_board[:, c], col_hints_clean[c], N):
                            match = False
                            break
                if match:
                    return local_board
                else:
                    self.backtrack_count += 1
                    return None

            state = env.get_state()
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                if is_2d_board_model:
                    from nonogram.agents.dqn import pad_hints_batch
                    k = hint_length(N)
                    padded_rh = torch.tensor(pad_hints_batch([row_hints], k, N), dtype=torch.long, device=self.device)
                    padded_ch = torch.tensor(pad_hints_batch([col_hints], k, N), dtype=torch.long, device=self.device)
                    q_values = self.q_net(state_t, padded_rh, padded_ch).squeeze(0).cpu().numpy()
                else:
                    q_values = self.q_net(state_t).squeeze(0).cpu().numpy()

            # 후보 수집
            candidates = []
            for r in range(N):
                for c in range(N):
                    if local_board[r, c] == 0:
                        idx = r * N + c
                        candidates.append((q_values[idx], r, c, 1))
                        candidates.append((q_values[N * N + idx], r, c, -1))

            if not candidates:
                return None

            # Q-value 기준 내림차순 정렬
            candidates.sort(key=lambda x: x[0], reverse=True)

            # Lazy Feasibility Check
            best_cand = None
            for q_val, r, c, f in candidates:
                local_board[r, c] = f
                row_feas = is_feasible(local_board[r, :], row_hints_clean[r], N)
                col_feas = is_feasible(local_board[:, c], col_hints_clean[c], N) if row_feas else False
                local_board[r, c] = 0

                if row_feas and col_feas:
                    best_cand = (q_val, r, c, f)
                    break

            if best_cand is None:
                self.backtrack_count += 1
                return None

            best_q_val, best_r, best_c, best_f = best_cand

            num_unknowns = int((local_board == 0).sum())
            if num_unknowns < self.min_unknowns:
                self.min_unknowns = num_unknowns
                self.best_feasible_board = local_board.copy()

            if verbose:
                print(f"[DFS Level {force_count}] Guessing cell ({best_r}, {best_c}) = {best_f} (Q={best_q_val:+.4f})")

            # 첫 번째 선택지 탐색
            next_board = local_board.copy()
            next_board[best_r, best_c] = best_f
            res = search(next_board, force_count + 1)
            if res is not None:
                return res

            if self.aborted:
                return None

            # 두 번째 선택지 탐색 (feasible한 경우)
            second_f = -best_f
            local_board[best_r, best_c] = second_f
            second_feasible = is_feasible(local_board[best_r, :], row_hints_clean[best_r], N) and \
                              is_feasible(local_board[:, best_c], col_hints_clean[best_c], N)
            local_board[best_r, best_c] = 0

            if second_feasible:
                if verbose:
                    print(f"[DFS Level {force_count}] Backtracking! Trying cell ({best_r}, {best_c}) = {second_f}")
                next_board2 = local_board.copy()
                next_board2[best_r, best_c] = second_f
                res2 = search(next_board2, force_count + 1)
                return res2
            else:
                if verbose:
                    print(f"[DFS Level {force_count}] Backtracking skipped. Opposite value {second_f} is infeasible.")
                    self.backtrack_count += 1
                return None

        final_board = search(env.board.copy(), 0)

        if verbose:
            print(f"(Search finished. Calls: {self.search_calls}, Backtracks: {self.backtrack_count}, Aborted: {self.aborted})")

        if final_board is None:
            return self.best_feasible_board

        return final_board

