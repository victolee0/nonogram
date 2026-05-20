"""
NxN nonogram solver built on top of the 1D DQN Q-network.

Algorithm (a slightly more efficient version of the user's pseudocode):

    board <- all zeros (0 = unknown, +1 = filled, -1 = X)
    mark every row and column as "dirty"
    repeat:
        process all dirty rows (one batched forward per pass):
            for each dirty row, find ALL valid actions a=(j, f) whose Q(s, a) <= -threshold.
            For each such action, write the OPPOSITE (j, -f) into the board.
            Any column whose cell changed becomes dirty.
            If a row changed (and isn't yet fully decided), keep it dirty -- the new
            information may unlock more -1 actions on the next forward.
        process all dirty columns symmetrically.
    until a full pass over rows + columns produced no change.

The key optimisation over the literal pseudocode: actions whose Q is already -1
are mutually consistent (each one is an independent "this cell must be the
other value" statement), so we apply them all at once between forwards instead
of one-at-a-time. We also batch every line of one axis into a single forward.
"""

import argparse

import numpy as np
import torch

from utils import (
    QNetwork,
    action_to_idx,
    encode_hint,
    hint_length,
    idx_to_action,
)
from nonogram.solvers.line_solver import LineSolver


# ===========================================================================
# Puzzle definition -- edit these directly to solve a different board.
# ===========================================================================

ROW_HINTS = [
    [1, 1],
    [3],
    [2],
    [1, 1],
    [1, 2]
]

COL_HINTS = [
    [1, 2],
    [1],
    [2, 1],
    [2, 1],
    [1, 1]
]

# N is inferred from ROW_HINTS/COL_HINTS, so no need to set it manually.
# It must equal the row/column length the 1D Q-network was trained for.

# ---------------------------------------------------------------------------
# Argument parsing (only the model + solver knobs are CLI-tunable now)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='qnet_N10.pt',
                        help='Path to a 1D Q-network checkpoint trained for N == board size.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Only used if the checkpoint omits hidden_dim.')
    parser.add_argument('--neg_threshold', type=float, default=-0.5,
                        help='Action is treated as "Q == -1" when Q <= neg_threshold.')
    parser.add_argument('--max_outer_iters', type=int, default=200,
                        help='Safety cap on outer row/col alternations.')
    parser.add_argument('--max_forces', type=int, default=None,
                        help='Cap on deadlock-breaking forced guesses. Default: N*N.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print the board after every pass.')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# State / line helpers
# ---------------------------------------------------------------------------

def get_line(board, axis, idx):
    """Return the row (axis='row') or column (axis='col') at `idx`."""
    return board[idx, :].copy() if axis == 'row' else board[:, idx].copy()


def set_cell(board, axis, line_idx, j, value, dirty_other):
    """Write `value` to the cell at (line_idx, j) on `axis`, marking the
    perpendicular line dirty. Returns True if the cell was previously unknown."""
    if axis == 'row':
        r, c = line_idx, j
    else:
        r, c = j, line_idx
    if board[r, c] != 0:
        return False
    board[r, c] = value
    dirty_other.add(c if axis == 'row' else r)
    return True


def is_line_complete(line):
    return bool(np.all(line != 0))


def build_state(hint_padded, line):
    """state = [k hint values] + [N block states]."""
    return np.concatenate([np.asarray(hint_padded, dtype=np.float32),
                           np.asarray(line, dtype=np.float32)])


def valid_mask_from_line(line, N):
    mask = np.zeros(2 * N, dtype=np.float32)
    for i in range(N):
        if line[i] == 0:
            mask[i] = 1.0
            mask[N + i] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def process_pass(axis, board, hints_padded, dirty_self, dirty_other,
                 q_net, device, N, neg_threshold):
    """Process every dirty line of `axis` until no more progress on this axis.

    Each inner iteration:
      1. Batch-forward Q for all currently-dirty lines.
      2. For each line, apply EVERY valid action with Q <= neg_threshold by
         writing the opposite value. Lines that changed but aren't fully
         decided stay dirty for the next inner iteration.

    Returns True iff anything on the board changed during this pass.
    """
    changed_globally = False
    pending = sorted(dirty_self)
    dirty_self.clear()

    while pending:
        # ---- batched forward ----
        lines = [get_line(board, axis, i) for i in pending]
        states = np.stack([build_state(hints_padded[i], lines[idx])
                           for idx, i in enumerate(pending)])
        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        with torch.no_grad():
            q_all = q_net(states_t).cpu().numpy()       # shape: (num_lines, 2N)

        next_pending = []
        for idx, line_idx in enumerate(pending):
            line = lines[idx]
            q = q_all[idx]
            mask = valid_mask_from_line(line, N)

            # candidates: valid actions whose Q says "this is the wrong move"
            bad_actions = np.where((mask > 0) & (q <= neg_threshold))[0]

            line_changed = False
            for action_idx in bad_actions:
                j, f = idx_to_action(int(action_idx), N)
                # the model is asserting "(j, f) ends in failure",
                # which is equivalent to "cell j must be -f".
                if set_cell(board, axis, line_idx, j, -f, dirty_other):
                    line_changed = True
                    changed_globally = True

            if line_changed:
                new_line = get_line(board, axis, line_idx)
                if not is_line_complete(new_line):
                    next_pending.append(line_idx)
            # if not changed: this line is stuck for now -- leave it out;
            # perpendicular progress may revive it later via dirty marking.

        pending = next_pending

    return changed_globally


def force_one_cell(board, row_hints_padded, col_hints_padded,
                   dirty_rows, dirty_cols, q_net, device, N):
    """Deadlock-breaking step: among every undecided cell-action on every
    still-undecided line, pick the (axis, line, action) with the highest Q
    over valid actions, and apply it to the board. Returns True if a cell was
    written, False if nothing was undecided (board already full or stuck in
    some other way)."""
    # Collect all undecided lines across both axes.
    candidates = []  # list of (axis, line_idx)
    for r in range(N):
        if not is_line_complete(board[r, :]):
            candidates.append(('row', r))
    for c in range(N):
        if not is_line_complete(board[:, c]):
            candidates.append(('col', c))
    if not candidates:
        return False

    # Build states and a flat (axis, line_idx, line) record for each.
    states = []
    for axis, idx in candidates:
        line = get_line(board, axis, idx)
        hint = row_hints_padded[idx] if axis == 'row' else col_hints_padded[idx]
        states.append(build_state(hint, line))
    states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
    with torch.no_grad():
        q_all = q_net(states_t).cpu().numpy()        # (num_candidates, 2N)

    # Mask out invalid actions per line, then find the global argmax.
    best = (-np.inf, None)  # (q_value, (cand_idx, action_idx))
    for cand_idx, (axis, idx) in enumerate(candidates):
        line = get_line(board, axis, idx)
        mask = valid_mask_from_line(line, N)
        q = q_all[cand_idx].copy()
        q[mask == 0] = -np.inf
        a = int(np.argmax(q))
        if q[a] > best[0]:
            best = (float(q[a]), (cand_idx, a))

    if best[1] is None:
        return False  # shouldn't happen if candidates is non-empty

    q_val, (cand_idx, action_idx) = best
    axis, line_idx = candidates[cand_idx]
    j, f = idx_to_action(action_idx, N)
    # Forced guess: write f (NOT -f) -- argmax is "the action that ends well",
    # i.e. picking cell j to be f leads to +1 reward.
    other_dirty = dirty_cols if axis == 'row' else dirty_rows
    set_cell(board, axis, line_idx, j, f, other_dirty)
    # The forced line itself may unlock more on the next pass.
    (dirty_rows if axis == 'row' else dirty_cols).add(line_idx)
    return {'axis': axis, 'line_idx': line_idx, 'j': j, 'f': f, 'q': q_val}


def solve(N, row_hints, col_hints, q_net, device, neg_threshold,
          max_outer_iters, verbose=False, max_forces=None):
    config = {
        "solver": {
            "neg_threshold": neg_threshold,
            "max_outer_iters": max_outer_iters,
            "min_guess_q": 0.05,
            "max_backtracks": max_forces if max_forces is not None else 1000,
        }
    }
    solver = LineSolver(config=config, q_net=q_net, device=device)
    return solver.solve(N=N, row_hints=row_hints, col_hints=col_hints, verbose=verbose)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

FILLED = '■'
EMPTY = '□'


def render_board(board, row_hints, col_hints):
    """Render the board with row hints on the left and col hints on top."""
    N = board.shape[0]

    # Column hint header (top-aligned by stacking each col's hint vertically).
    # Each board cell renders as "<char> " (2 chars wide), so each header column
    # is also 2 chars wide: one digit + one space.
    max_col_h = max((len(c) for c in col_hints), default=1)
    col_header_rows = []
    for layer in range(max_col_h):
        parts = []
        for c in col_hints:
            pad = max_col_h - len(c)
            if layer < pad:
                parts.append('  ')
            else:
                parts.append(f'{c[layer - pad]:1d} ')
        col_header_rows.append(''.join(parts))

    # Row hint width
    row_hint_strs = [' '.join(str(x) for x in h) for h in row_hints]
    row_hint_width = max((len(s) for s in row_hint_strs), default=0)

    out_lines = []
    indent = ' ' * (row_hint_width + 2)
    for header in col_header_rows:
        out_lines.append(indent + header)

    for r in range(N):
        cells = []
        for c in range(N):
            v = board[r, c]
            cells.append(FILLED if v == 1 else EMPTY)
        out_lines.append(f'{row_hint_strs[r]:>{row_hint_width}}  ' + ' '.join(cells))

    return '\n'.join(out_lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    row_hints = ROW_HINTS
    col_hints = COL_HINTS

    if len(row_hints) != len(col_hints):
        raise ValueError(f"ROW_HINTS has {len(row_hints)} rows but COL_HINTS has "
                         f"{len(col_hints)} columns -- the board must be square.")
    N = len(row_hints)

    ckpt = torch.load(args.load_path, map_location=device)
    ckpt_N = int(ckpt.get('N', N))
    hidden_dim = int(ckpt.get('hidden_dim', args.hidden_dim))

    if ckpt_N != N:
        raise ValueError(
            f"The 1D model in {args.load_path} was trained for N={ckpt_N}, but the "
            f"board has rows/columns of length {N}. Train (or load) a model "
            f"with N == {N} (e.g. `python train.py --N {N} "
            f"--save_path qnet_N{N}.pt`).")

    q_net = QNetwork(N, hidden_dim).to(device)
    q_net.load_state_dict(ckpt['model_state_dict'])
    q_net.eval()

    print(f"Solving {N}x{N} nonogram using {args.load_path} (N={ckpt_N})")
    print()

    board = solve(
        N=N,
        row_hints=row_hints,
        col_hints=col_hints,
        q_net=q_net,
        device=device,
        neg_threshold=args.neg_threshold,
        max_outer_iters=args.max_outer_iters,
        max_forces=args.max_forces,
        verbose=args.verbose,
    )

    print("=== final board ===")
    print(render_board(board, row_hints, col_hints))

    n_filled = int((board == 1).sum())
    n_x = int((board == -1).sum())
    n_unknown = int((board == 0).sum())
    print()
    print(f"filled={n_filled}, X={n_x}, unknown={n_unknown}")
    if n_unknown == 0:
        print("(board fully decided)")
    else:
        print("(some cells remained unknown -- the 1D model could not force them; "
              "either the model is undertrained or the puzzle requires deeper "
              "inference than the per-line Q-network provides)")


if __name__ == '__main__':
    main()