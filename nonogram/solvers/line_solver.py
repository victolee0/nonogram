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
        min_guess_q = self.solver_cfg.get("min_guess_q", 0.0)
        max_backtracks = self.solver_cfg.get("max_backtracks", 1000)
        method = self.solver_cfg.get("method", "cp")  # "cp" | "rl"

        # 힌트 인코딩 및 정제 (0 패딩 제외한 실제 힌트 배열 추출)
        row_hints_padded = [encode_hint(h, N) for h in row_hints]
        col_hints_padded = [encode_hint(h, N) for h in col_hints]

        row_hints_clean = [[int(x) for x in h if int(x) > 0] for h in row_hints]
        col_hints_clean = [[int(x) for x in h if int(x) > 0] for h in col_hints]

        self.row_hints_clean = row_hints_clean
        self.col_hints_clean = col_hints_clean

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

            # === 풀이 방법 분기 ===
            if method == "cp":
                # --- CP 모드: AC-3 Constraint Propagation (순수 논리) ---
                prop_result = self._propagate_constraints(local_board, N)
                if not prop_result:
                    self.backtrack_count += 1
                    return None
            else:
                # --- RL 모드: Q-value Deduction (순수 RL) ---
                # 1D Q-network의 배제 추론을 반복 실행.
                # 각 배제마다 per-cell feasibility guard가 보호.
                outer = 0
                while outer < max_outer_iters:
                    outer += 1

                    rows_to_check = set(local_dirty_rows)
                    cols_to_check = set(local_dirty_cols)

                    changed_row = self._process_pass(
                        "row", local_board, row_hints_padded,
                        local_dirty_rows, local_dirty_cols, N, neg_threshold
                    )
                    changed_col = self._process_pass(
                        "col", local_board, col_hints_padded,
                        local_dirty_cols, local_dirty_rows, N, neg_threshold
                    )

                    rows_to_check.update(local_dirty_rows)
                    cols_to_check.update(local_dirty_cols)

                    # 변경된 줄만 feasibility 검사
                    if not self._check_feasibility_efficient(
                        local_board, row_hints_clean, col_hints_clean, N,
                        rows_to_check, cols_to_check
                    ):
                        self.backtrack_count += 1
                        return None

                    if not (changed_row or changed_col):
                        break

            # feasibility 통과 시점: 가장 많이 채워진 feasible 보드를 업데이트
            num_unknowns = int((local_board == 0).sum())
            if num_unknowns < self.min_unknowns:
                self.min_unknowns = num_unknowns
                self.best_feasible_board = local_board.copy()

            # === 종료 조건 검사 ===
            if (local_board != 0).all():
                if self._check_final_match(local_board, row_hints_clean, col_hints_clean, N):
                    return local_board
                else:
                    self.backtrack_count += 1
                    return None

            # === 교착 상태 (Deadlock) → Greedy DFS Guessing (Q-value argmax 기반, 백트래킹 가능) ===
            guess_cand = self._get_best_guess_candidate(
                local_board, row_hints_padded, col_hints_padded, N
            )
            if guess_cand is None:
                return None

            axis, best_r, best_c, first_val, second_val, second_feasible, q_val = guess_cand

            if q_val < min_guess_q:
                if verbose:
                    print(f"[DFS Level {force_count}] Guessing aborted. Best Q ({q_val:+.4f}) < threshold ({min_guess_q})")
                return local_board

            if verbose:
                print(f"[DFS Level {force_count}] Guessing: cell ({best_r}, {best_c}) = {first_val} (Q={q_val:+.4f})")

            # 첫 번째 값 가정 탐색 (이미 feasibility가 보장됨)
            next_board = local_board.copy()
            next_board[best_r, best_c] = first_val

            res = search(next_board, set(range(N)), set(range(N)), force_count + 1)
            if res is not None:
                return res

            if self.aborted:
                return None

            # 두 번째 값(반대 값)이 feasible한 경우에만 안전하게 백트래킹 탐색
            if second_feasible:
                if verbose:
                    print(f"[DFS Level {force_count}] Backtracking! Trying: cell ({best_r}, {best_c}) = {second_val}")

                next_board2 = local_board.copy()
                next_board2[best_r, best_c] = second_val

                res2 = search(next_board2, set(range(N)), set(range(N)), force_count + 1)
                return res2
            else:
                if verbose:
                    print(f"[DFS Level {force_count}] Backtracking skipped. Opposite value {second_val} is infeasible.")
                return None

        # 최초 탐색 시작
        final_board = search(initial_board, set(range(N)), set(range(N)), 0)

        if verbose:
            print(f"(Search finished. Calls: {self.search_calls}, Backtracks: {self.backtrack_count}, Aborted: {self.aborted})")

        if final_board is None:
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

        # 1. 안전장치 탑재 Deduction (배제) 단계
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
                if len(bad_actions) > 0:
                    # Q-value가 가장 낮은 (배제 신뢰도가 가장 강력한) 단 1개의 행동만 선별
                    best_bad_action = min(bad_actions, key=lambda a: q[a])
                    j, f = idx_to_action(int(best_bad_action), N)
                    
                    if axis == "row":
                        r, c = line_idx, j
                    else:
                        r, c = j, line_idx
                        
                    temp_board = board.copy()
                    temp_board[r, c] = -f
                    
                    # 2D feasibility 검증
                    if is_feasible(temp_board[r, :], self.row_hints_clean[r], N) and \
                       is_feasible(temp_board[:, c], self.col_hints_clean[c], N):
                        if self._set_cell(board, axis, line_idx, j, -f, dirty_other):
                            line_changed = True
                            changed_globally = True

                if line_changed:
                    new_line = self._get_line(board, axis, line_idx)
                    if not np.all(new_line != 0):
                        next_pending.append(line_idx)

            pending = next_pending

        # Greedy Step 제거 — DFS Guessing 단계에서 Q-value argmax로
        # 백트래킹 가능하게 처리하여 2D 교차 모순 해결
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

    def _propagate_constraints(self, board, N) -> bool:
        """AC-3 스타일 제약 전파: 각 unknown 셀에 대해 유일한 feasible 값이 존재하면 즉시 확정.

        양방향(+1, -1) feasibility를 모두 검사하여:
        - 하나만 가능 → 논리적으로 확정 (백트래킹 불필요)
        - 둘 다 불가능 → 모순 발견 (False 반환)
        - 둘 다 가능 → 아직 확정 불가 (DFS Guessing에서 처리)
        """
        to_check = set()
        for r in range(N):
            for c in range(N):
                if board[r, c] == 0:
                    to_check.add((r, c))

        while to_check:
            next_check = set()
            for (r, c) in list(to_check):
                if board[r, c] != 0:
                    continue

                # 값 +1 가능 여부 (in-place 검사 후 원복)
                board[r, c] = 1
                feas_pos = is_feasible(board[r, :], self.row_hints_clean[r], N) and \
                           is_feasible(board[:, c], self.col_hints_clean[c], N)

                # 값 -1 가능 여부
                board[r, c] = -1
                feas_neg = is_feasible(board[r, :], self.row_hints_clean[r], N) and \
                           is_feasible(board[:, c], self.col_hints_clean[c], N)

                # 원복
                board[r, c] = 0

                if feas_pos and not feas_neg:
                    board[r, c] = 1
                    # 같은 행/열의 다른 unknown 셀을 재검사 대상에 추가
                    for j in range(N):
                        if board[r, j] == 0 and j != c:
                            next_check.add((r, j))
                        if board[j, c] == 0 and j != r:
                            next_check.add((j, c))
                elif feas_neg and not feas_pos:
                    board[r, c] = -1
                    for j in range(N):
                        if board[r, j] == 0 and j != c:
                            next_check.add((r, j))
                        if board[j, c] == 0 and j != r:
                            next_check.add((j, c))
                elif not feas_pos and not feas_neg:
                    return False  # 모순 발견: 어떤 값도 불가능
                # else: 양쪽 다 feasible → 아직 확정 불가

            to_check = next_check

        return True

    def _get_best_guess_candidate(self, board, row_hints_padded, col_hints_padded, N):
        """배치 Q-value 연산 + Lazy Feasibility Check로 최적의 분기 셀 선별.

        모든 행의 Q-value를 한 번에 배치 연산하고, Q-value 내림차순으로
        feasibility를 순차 검증하여 최초 통과 후보를 즉시 채택합니다.
        """
        # 1. 모든 행에 대한 Q-value 배치 연산 (N개 forward pass → 1개 배치)
        states = []
        for r in range(N):
            line = self._get_line(board, "row", r)
            hint = row_hints_padded[r]
            state = self._build_state(hint, line)
            states.append(state)

        states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_all = self.q_net(states_t).cpu().numpy()  # shape: (N, 2*N)

        # 2. 모든 unknown 셀의 양방향 Q-value 수집
        candidates = []
        for r in range(N):
            for c in range(N):
                if board[r, c] == 0:
                    candidates.append((q_all[r, c], r, c, 1))       # fill (+1)
                    candidates.append((q_all[r, N + c], r, c, -1))  # empty (-1)

        if not candidates:
            return None

        # 3. Q-value 기준 내림차순 정렬
        candidates.sort(key=lambda x: x[0], reverse=True)

        # 4. Lazy Feasibility Check: 상위 후보부터 최초 통과 후보 즉시 채택
        best_cand = None
        for q_val, r, c, f in candidates:
            board[r, c] = f
            row_feas = is_feasible(board[r, :], self.row_hints_clean[r], N)
            col_feas = is_feasible(board[:, c], self.col_hints_clean[c], N) if row_feas else False
            board[r, c] = 0  # 원복

            if row_feas and col_feas:
                best_cand = (q_val, r, c, f)
                break

        if best_cand is None:
            return None

        best_q_val, best_r, best_c, best_f = best_cand

        # 5. 반대 값(second_val)도 feasible한지 검증
        second_f = -best_f
        board[best_r, best_c] = second_f
        second_feasible = is_feasible(board[best_r, :], self.row_hints_clean[best_r], N) and \
                          is_feasible(board[:, best_c], self.col_hints_clean[best_c], N)
        board[best_r, best_c] = 0  # 원복

        return "row", best_r, best_c, best_f, second_f, second_feasible, best_q_val
