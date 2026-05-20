"""
Line Solver — 기존 solve_nonogram.py의 2D 풀이 알고리즘을 확장.

1D Q-network를 사용하여 행/열을 교대로 처리하는 방식에 더하여,
교착 상태(Deadlock) 발생 시 백트래킹 DFS(Depth-First Search)와
실시간 변경 줄 한정 Feasibility 검사를 결합하여 10x10 등 대형 퍼즐의 정답 달성률을 극대화함.
"""

import numpy as np
import torch

from nonogram.env.state import (
    encode_hint,
    idx_to_action,
    hint_length,
    check_hint_match,
)
from nonogram.env.feasibility import is_feasible
from nonogram.solvers.registry import register_solver


@register_solver("line")
class LineSolver:
    """1D Q-network 기반 줄 교대 풀이기 + 최적화 백트래킹 DFS."""

    def __init__(self, config: dict, q_net=None, device=None, **kwargs):
        self.config = config
        self.solver_cfg = config["solver"]
        self.q_net = q_net
        self.device = device or torch.device("cpu")

    def solve(self, N: int, row_hints: list, col_hints: list,
              verbose: bool = False) -> np.ndarray:
        """NxN 노노그램 풀이.

        Returns:
            board: (N, N) ndarray, 1=filled, -1=X, 0=unknown
        """
        neg_threshold = self.solver_cfg["neg_threshold"]
        max_outer_iters = self.solver_cfg["max_outer_iters"]
        
        # 최적화용 추가 하이퍼파라미터
        min_guess_q = self.solver_cfg.get("min_guess_q", 0.2)
        max_backtracks = self.solver_cfg.get("max_backtracks", 1000)

        # 힌트 인코딩 및 정제 (0 패딩 제외한 실제 힌트 배열 추출)
        row_hints_padded = [encode_hint(h, N) for h in row_hints]
        col_hints_padded = [encode_hint(h, N) for h in col_hints]

        row_hints_clean = [[int(x) for x in h if int(x) > 0] for h in row_hints]
        col_hints_clean = [[int(x) for x in h if int(x) > 0] for h in col_hints]

        initial_board = np.zeros((N, N), dtype=np.int64)

        # 백트래킹을 추적할 내부 상태
        self.search_calls = 0
        self.backtrack_count = 0
        self.aborted = False
        
        # 탐색 과정 중 가장 많이 채워진 feasible 보드를 기록하기 위한 변수
        self.best_feasible_board = initial_board.copy()
        self.min_unknowns = N * N

        def search(board, dirty_rows, dirty_cols, force_count):
            self.search_calls += 1
            if self.backtrack_count >= max_backtracks:
                self.aborted = True
                return None

            local_board = board.copy()
            local_dirty_rows = set(dirty_rows)
            local_dirty_cols = set(dirty_cols)

            outer = 0
            while outer < max_outer_iters:
                outer += 1
                
                # pass 진행 전, 변경 대상 보관
                rows_to_check = set(local_dirty_rows)
                cols_to_check = set(local_dirty_cols)

                changed_row = self._process_pass(
                    "row", local_board, row_hints_padded, local_dirty_rows, local_dirty_cols, N, neg_threshold
                )
                changed_col = self._process_pass(
                    "col", local_board, col_hints_padded, local_dirty_cols, local_dirty_rows, N, neg_threshold
                )

                rows_to_check.update(local_dirty_rows)
                cols_to_check.update(local_dirty_cols)

                # 1. 최적화된 Feasibility 검사: 변경된 줄만 feasibility 검사 수행하여 O(N)으로 최적화
                if not self._check_feasibility_efficient(local_board, row_hints_clean, col_hints_clean, N, rows_to_check, cols_to_check):
                    self.backtrack_count += 1
                    return None

                # feasibility 통과 시점: 가장 많이 채워진 feasible 보드를 업데이트
                num_unknowns = int((local_board == 0).sum())
                if num_unknowns < self.min_unknowns:
                    self.min_unknowns = num_unknowns
                    self.best_feasible_board = local_board.copy()

                if not (changed_row or changed_col):
                    break

            # 2. 종료 조건 검사
            if (local_board != 0).all():
                # 최종 힌트 만족 여부 정교 검증
                if self._check_final_match(local_board, row_hints_clean, col_hints_clean, N):
                    return local_board
                else:
                    self.backtrack_count += 1
                    return None

            # 3. 교착 상태 (Deadlock) → 분기 탐색 (DFS Guessing)
            guess_cand = self._get_best_guess_candidate(
                local_board, row_hints_padded, col_hints_padded, N
            )
            if guess_cand is None:
                return None

            axis, line_idx, j, first_val, second_val, q_val = guess_cand

            # 신뢰도가 임계치보다 낮으면 탐색 헬(Backtracking Hell)에 들어가지 않고, 
            # 현재까지의 확실한 보드 상태(local_board)를 최종 결과로 반환해 안전 장치 확보
            if q_val < min_guess_q:
                if verbose:
                    print(f"[DFS Level {force_count}] Guessing aborted. Best Q ({q_val:+.4f}) < threshold ({min_guess_q})")
                return local_board

            if verbose:
                print(f"[DFS Level {force_count}] Guessing: {axis} {line_idx}, cell {j} = {first_val} (Q={q_val:+.4f})")

            # 첫 번째 값 가정 탐색
            next_board = local_board.copy()
            if axis == "row":
                next_board[line_idx, j] = first_val
                next_dirty_rows = {line_idx}
                next_dirty_cols = {j}
            else:
                next_board[j, line_idx] = first_val
                next_dirty_rows = {j}
                next_dirty_cols = {line_idx}

            res = search(next_board, next_dirty_rows, next_dirty_cols, force_count + 1)
            if res is not None:
                return res

            if self.aborted:
                return None

            # 첫 번째 가정이 실패하면, 두 번째 값(반대 값)으로 백트래킹 탐색
            if verbose:
                print(f"[DFS Level {force_count}] Backtracking! Trying: {axis} {line_idx}, cell {j} = {second_val}")

            next_board2 = local_board.copy()
            if axis == "row":
                next_board2[line_idx, j] = second_val
                next_dirty_rows = {line_idx}
                next_dirty_cols = {j}
            else:
                next_board2[j, line_idx] = second_val
                next_dirty_rows = {j}
                next_dirty_cols = {line_idx}

            res2 = search(next_board2, next_dirty_rows, next_dirty_cols, force_count + 1)
            return res2

        # 최초 탐색 시작
        final_board = search(initial_board, set(range(N)), set(range(N)), 0)

        if verbose:
            print(f"(Search finished. Calls: {self.search_calls}, Backtracks: {self.backtrack_count}, Aborted: {self.aborted})")

        if final_board is None:
            # 백트래킹 실패 시 안전 장치로 가장 많이 채워졌던 feasible 보드 반환
            return self.best_feasible_board

        return final_board

    # --- internal methods ---

    def _get_line(self, board, axis, idx):
        return board[idx, :].copy() if axis == "row" else board[:, idx].copy()

    def _set_cell(self, board, axis, line_idx, j, value, dirty_other):
        if axis == "row":
            r, c = line_idx, j
        else:
            r, c = j, line_idx
        if board[r, c] != 0:
            return False
        board[r, c] = value
        dirty_other.add(c if axis == "row" else r)
        return True

    def _build_state(self, hint_padded, line):
        return np.concatenate([
            np.asarray(hint_padded, dtype=np.float32),
            np.asarray(line, dtype=np.float32),
        ])

    def _valid_mask_from_line(self, line, N):
        mask = np.zeros(2 * N, dtype=np.float32)
        for i in range(N):
            if line[i] == 0:
                mask[i] = 1.0
                mask[N + i] = 1.0
        return mask

    def _process_pass(self, axis, board, hints_padded, dirty_self, dirty_other,
                      N, neg_threshold):
        changed_globally = False
        pending = sorted(dirty_self)
        dirty_self.clear()

        while pending:
            lines = [self._get_line(board, axis, i) for i in pending]
            states = np.stack([
                self._build_state(hints_padded[i], lines[idx])
                for idx, i in enumerate(pending)
            ])
            states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_all = self.q_net(states_t).cpu().numpy()

            next_pending = []
            for idx, line_idx in enumerate(pending):
                line = lines[idx]
                q = q_all[idx]
                mask = self._valid_mask_from_line(line, N)

                bad_actions = np.where((mask > 0) & (q <= neg_threshold))[0]

                line_changed = False
                for action_idx in bad_actions:
                    j, f = idx_to_action(int(action_idx), N)
                    if self._set_cell(board, axis, line_idx, j, -f, dirty_other):
                        line_changed = True
                        changed_globally = True

                if line_changed:
                    new_line = self._get_line(board, axis, line_idx)
                    if not np.all(new_line != 0):
                        next_pending.append(line_idx)

            pending = next_pending

        return changed_globally

    def _check_feasibility_efficient(self, board, row_hints_clean, col_hints_clean, N, rows, cols) -> bool:
        """변경된 행과 열에 대해서만 feasibility를 정밀 체크하여 연산량 O(N)으로 절감."""
        for r in rows:
            if not is_feasible(board[r, :], row_hints_clean[r], N):
                return False
        for c in cols:
            if not is_feasible(board[:, c], col_hints_clean[c], N):
                return False
        return True

    def _check_final_match(self, board, row_hints_clean, col_hints_clean, N) -> bool:
        """최종 완성된 보드가 실제 힌트를 만족하는지 정교하게 검사."""
        for r in range(N):
            if not check_hint_match(board[r, :], row_hints_clean[r], N):
                return False
        for c in range(N):
            if not check_hint_match(board[:, c], col_hints_clean[c], N):
                return False
        return True

    def _get_best_guess_candidate(self, board, row_hints_padded, col_hints_padded, N):
        """교착 상태 시, Q-value가 가장 높은 최적의 이진 분기 후보 셀을 선별."""
        candidates = []
        for r in range(N):
            if not np.all(board[r, :] != 0):
                candidates.append(("row", r))
        for c in range(N):
            if not np.all(board[:, c] != 0):
                candidates.append(("col", c))
        if not candidates:
            return None

        states = []
        for axis, idx in candidates:
            line = self._get_line(board, axis, idx)
            hint = row_hints_padded[idx] if axis == "row" else col_hints_padded[idx]
            states.append(self._build_state(hint, line))

        states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_all = self.q_net(states_t).cpu().numpy()

        best = (-np.inf, None)
        for cand_idx, (axis, idx) in enumerate(candidates):
            line = self._get_line(board, axis, idx)
            mask = self._valid_mask_from_line(line, N)
            q = q_all[cand_idx].copy()
            q[mask == 0] = -np.inf
            a = int(np.argmax(q))
            if q[a] > best[0]:
                best = (float(q[a]), (cand_idx, a))

        if best[1] is None:
            return None

        q_val, (cand_idx, action_idx) = best
        axis, line_idx = candidates[cand_idx]
        j, f = idx_to_action(action_idx, N)
        return axis, line_idx, j, f, -f, q_val
