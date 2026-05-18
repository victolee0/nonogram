"""
Line Solver — 기존 solve_nonogram.py의 2D 풀이 알고리즘.

1D Q-network를 사용하여 행/열을 교대로 처리하는 방식.
Q <= neg_threshold인 action의 반대를 적용하고,
deadlock 시 force step으로 한 셀을 추측.
"""

import numpy as np
import torch

from nonogram.env.state import (
    encode_hint,
    idx_to_action,
    hint_length,
)
from nonogram.solvers.registry import register_solver


@register_solver("line")
class LineSolver:
    """1D Q-network 기반 줄 교대 풀이기."""

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
        max_forces = self.solver_cfg.get("max_forces")
        if max_forces is None:
            max_forces = N * N

        board = np.zeros((N, N), dtype=np.int64)

        row_hints_padded = [encode_hint(h, N) for h in row_hints]
        col_hints_padded = [encode_hint(h, N) for h in col_hints]

        dirty_rows = set(range(N))
        dirty_cols = set(range(N))

        force_count = 0
        outer = 0
        while outer < max_outer_iters:
            outer += 1
            changed_row = self._process_pass(
                "row", board, row_hints_padded, dirty_rows, dirty_cols, N, neg_threshold
            )
            changed_col = self._process_pass(
                "col", board, col_hints_padded, dirty_cols, dirty_rows, N, neg_threshold
            )

            if verbose:
                print(f"--- after outer pass {outer} "
                      f"(row_changed={changed_row}, col_changed={changed_col}) ---")
                from nonogram.utils.visualization import render_board
                print(render_board(board, row_hints, col_hints))
                print()

            if changed_row or changed_col:
                continue

            if (board != 0).all():
                break

            if force_count >= max_forces:
                if verbose:
                    print(f"(reached max_forces={max_forces}; stopping with "
                          f"{(board == 0).sum()} unknown cells)")
                break

            forced = self._force_one_cell(
                board, row_hints_padded, col_hints_padded,
                dirty_rows, dirty_cols, N
            )
            if not forced:
                break
            force_count += 1
            if verbose:
                print(f"*** forced guess #{force_count}: "
                      f"{forced['axis']} {forced['line_idx']}, cell {forced['j']} = "
                      f"{'+1' if forced['f'] == 1 else '-1'} "
                      f"(Q = {forced['q']:+.4f}) ***")
                print()

        if verbose:
            print(f"(stopped after {outer} outer pass(es), {force_count} forced guess(es))")
        return board

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

    def _force_one_cell(self, board, row_hints_padded, col_hints_padded,
                        dirty_rows, dirty_cols, N):
        candidates = []
        for r in range(N):
            if not np.all(board[r, :] != 0):
                candidates.append(("row", r))
        for c in range(N):
            if not np.all(board[:, c] != 0):
                candidates.append(("col", c))
        if not candidates:
            return False

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
            return False

        q_val, (cand_idx, action_idx) = best
        axis, line_idx = candidates[cand_idx]
        j, f = idx_to_action(action_idx, N)
        other_dirty = dirty_cols if axis == "row" else dirty_rows
        self._set_cell(board, axis, line_idx, j, f, other_dirty)
        (dirty_rows if axis == "row" else dirty_cols).add(line_idx)
        return {"axis": axis, "line_idx": line_idx, "j": j, "f": f, "q": q_val}
