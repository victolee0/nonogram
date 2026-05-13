"""
Evaluation script for `solve_nonogram.solve`.

Generates random ground-truth NxN boards, extracts their hints, hands those
hints to the solver, and reports how often the solver's output is correct.

Two correctness criteria are tracked:

  * hint_satisfied : the solver returned a fully decided board whose runs
                     match the supplied hints. This is the "real" criterion --
                     nonograms can have multiple solutions for the same hints,
                     so any board satisfying the hints is a valid answer.
  * exact_match    : the solver's board equals the ground truth board cell by
                     cell. A stronger condition that only makes sense when the
                     puzzle has a unique solution.

Usage:
    python evaluate.py --N 5  --num_puzzles 200 --load_path qnet_N5.pt
    python evaluate.py --N 10 --num_puzzles 100 --load_path qnet_N10.pt --show_failures 3
"""

import argparse
import time

import numpy as np
import torch

from utils import QNetwork
from solve_nonogram import solve


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=5,
                        help='Board size (NxN). Must match the model checkpoint.')
    parser.add_argument('--num_puzzles', type=int, default=200,
                        help='How many random puzzles to evaluate on.')
    parser.add_argument('--load_path', type=str, default='qnet_N5.pt',
                        help='Path to a 1D Q-network checkpoint.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Used only if the checkpoint omits hidden_dim.')
    parser.add_argument('--neg_threshold', type=float, default=-0.5,
                        help='Action is treated as "Q == -1" when Q <= neg_threshold.')
    parser.add_argument('--max_outer_iters', type=int, default=200,
                        help='Safety cap on row/col alternations inside the solver.')
    parser.add_argument('--max_forces', type=int, default=None,
                        help='Cap on deadlock-breaking forced guesses inside the solver. '
                             'Default (None) is N*N. Set to 0 to disable forcing.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_failures', type=int, default=0,
                        help='Print up to this many failed puzzles in detail.')
    parser.add_argument('--log_every', type=int, default=20,
                        help='Print a running tally every N puzzles.')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Puzzle generation / hint extraction
# ---------------------------------------------------------------------------

def generate_random_board(N, rng):
    """Uniformly random board with each cell in {-1, +1}."""
    return rng.choice([-1, 1], size=(N, N)).astype(np.int64)


def extract_runs(line):
    """Return the list of run-lengths of +1s in `line`. Empty line -> [0]."""
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


def extract_hints_from_board(board):
    N = board.shape[0]
    row_hints = [extract_runs(board[i, :]) for i in range(N)]
    col_hints = [extract_runs(board[:, j]) for j in range(N)]
    return row_hints, col_hints


def _normalize(hint):
    """Drop zero entries so [] and [0] compare equal."""
    return [int(h) for h in hint if int(h) > 0]


def hints_equal(h1, h2):
    return _normalize(h1) == _normalize(h2)


# ---------------------------------------------------------------------------
# Per-puzzle evaluation
# ---------------------------------------------------------------------------

def evaluate_solution(solver_board, row_hints, col_hints, gt_board):
    """Compute correctness statistics for one solved puzzle."""
    N = solver_board.shape[0]
    num_unknown = int((solver_board == 0).sum())
    num_total = N * N

    hint_satisfied = False
    if num_unknown == 0:
        produced_rows, produced_cols = extract_hints_from_board(solver_board)
        hint_satisfied = (
            all(hints_equal(produced_rows[i], row_hints[i]) for i in range(N))
            and all(hints_equal(produced_cols[j], col_hints[j]) for j in range(N))
        )

    exact_match = bool(np.array_equal(solver_board, gt_board))

    # cell-level agreement vs ground truth (treat unknown 0 as "neither")
    agree = int(((solver_board == gt_board) & (solver_board != 0)).sum())

    return {
        'hint_satisfied': hint_satisfied,
        'exact_match': exact_match,
        'num_unknown': num_unknown,
        'num_total': num_total,
        'num_agree': agree,
    }


# ---------------------------------------------------------------------------
# Rendering (own renderer: distinguishes unknown ? from X .)
# ---------------------------------------------------------------------------

def render(board):
    chars = {1: '#', -1: '.', 0: '?'}
    return '\n'.join(' '.join(chars[int(v)] for v in board[r, :])
                     for r in range(board.shape[0]))


def format_hints(hints):
    return [' '.join(str(x) for x in h) for h in hints]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.load_path, map_location=device)
    ckpt_N = int(ckpt.get('N', args.N))
    hidden_dim = int(ckpt.get('hidden_dim', args.hidden_dim))

    if ckpt_N != args.N:
        raise ValueError(
            f"Checkpoint was trained for N={ckpt_N} but --N={args.N}. "
            f"They must match.")

    q_net = QNetwork(args.N, hidden_dim).to(device)
    q_net.load_state_dict(ckpt['model_state_dict'])
    q_net.eval()

    print(f"Evaluating {args.load_path} (N={ckpt_N}) on {args.num_puzzles} "
          f"random {args.N}x{args.N} puzzles\n")

    # Aggregate counters
    n_hint_ok = 0
    n_exact = 0
    n_fully_decided = 0
    total_unknown = 0
    total_agree = 0
    total_cells = 0
    solve_times = []
    failures = []  # list of (idx, gt, solver_board, row_hints, col_hints)

    t_start = time.time()
    for idx in range(1, args.num_puzzles + 1):
        gt = generate_random_board(args.N, rng)
        row_hints, col_hints = extract_hints_from_board(gt)

        t0 = time.time()
        solver_board = solve(
            N=args.N,
            row_hints=row_hints,
            col_hints=col_hints,
            q_net=q_net,
            device=device,
            neg_threshold=args.neg_threshold,
            max_outer_iters=args.max_outer_iters,
            max_forces=args.max_forces,
            verbose=False,
        )
        solve_times.append(time.time() - t0)

        stats = evaluate_solution(solver_board, row_hints, col_hints, gt)

        if stats['hint_satisfied']:
            n_hint_ok += 1
        if stats['exact_match']:
            n_exact += 1
        if stats['num_unknown'] == 0:
            n_fully_decided += 1
        total_unknown += stats['num_unknown']
        total_agree += stats['num_agree']
        total_cells += stats['num_total']

        if (not stats['hint_satisfied']) and len(failures) < args.show_failures:
            failures.append((idx, gt.copy(), solver_board.copy(),
                             row_hints, col_hints))

        if idx % args.log_every == 0 or idx == args.num_puzzles:
            elapsed = time.time() - t_start
            print(f"[{idx:4d}/{args.num_puzzles}]  "
                  f"hint_ok={n_hint_ok/idx:.3f}  "
                  f"exact={n_exact/idx:.3f}  "
                  f"decided={n_fully_decided/idx:.3f}  "
                  f"avg_unknown={total_unknown/idx:.2f}  "
                  f"agree={total_agree/total_cells:.3f}  "
                  f"({elapsed:.1f}s)")

    # ----- final report -----
    print("\n" + "=" * 60)
    print("Final results")
    print("=" * 60)
    n = args.num_puzzles
    print(f"hint-satisfied rate      : {n_hint_ok}/{n}  ({n_hint_ok/n:.3%})")
    print(f"exact-match rate         : {n_exact}/{n}  ({n_exact/n:.3%})")
    print(f"fully decided rate       : {n_fully_decided}/{n}  ({n_fully_decided/n:.3%})")
    print(f"avg unknown cells / board: {total_unknown/n:.2f}  / {args.N*args.N}")
    print(f"cell-level agreement     : {total_agree/total_cells:.3%}  "
          f"({total_agree}/{total_cells})")
    print(f"avg solve time / puzzle  : {np.mean(solve_times)*1000:.1f} ms  "
          f"(total {sum(solve_times):.1f}s)")

    # ----- failure dump -----
    if failures:
        print("\n" + "=" * 60)
        print(f"Showing {len(failures)} failed puzzle(s):")
        print("=" * 60)
        for idx, gt, sol, rh, ch in failures:
            print(f"\n--- puzzle #{idx} ---")
            print("row hints:", format_hints(rh))
            print("col hints:", format_hints(ch))
            print("ground truth:")
            print(render(gt))
            print("solver output (? = unknown, . = X, # = filled):")
            print(render(sol))


if __name__ == '__main__':
    main()