"""
2D 추론 시각화 스크립트 — 행/열 교대 풀이 또는 2D 보드 기반 풀이의 모든 추론 과정을 시각화.

Usage:
    uv run python scripts/inference_2d.py \
        --config runs/double_dqn_dueling_10x10/config.yaml \
        --load_path runs/double_dqn_dueling_10x10/checkpoints/best.pt \
        --row_hints "0;1;3;5;7;9;7;5;3;1" --col_hints "0;1;3;5;7;9;7;5;3;1"
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nonogram.config import load_config, make_base_parser, resolve_device, get_experiment_dir
from nonogram.models.registry import create_model
from nonogram.env.state import encode_hint, idx_to_action, action_to_idx
from nonogram.env.board_env import BoardEnv
from nonogram.utils.visualization import render_board_debug, render_board
from nonogram.utils.puzzle_io import load_puzzle, parse_hints_str


def visualise(blocks):
    chars = []
    for b in blocks:
        if int(b) == 1:
            chars.append('#')
        elif int(b) == -1:
            chars.append('.')
        else:
            chars.append('?')
    return ''.join(chars)


def run_line_inference(N, row_hints, col_hints, q_net, device, neg_threshold, max_outer_iters, max_forces):
    board = np.zeros((N, N), dtype=np.int64)

    row_hints_padded = [encode_hint(h, N) for h in row_hints]
    col_hints_padded = [encode_hint(h, N) for h in col_hints]

    dirty_rows = set(range(N))
    dirty_cols = set(range(N))

    force_count = 0
    outer = 0

    print("=== [초기 보드 상태] ===")
    print(render_board_debug(board))
    print()

    while outer < max_outer_iters:
        outer += 1
        print(f"====================== Outer Pass {outer} ======================")

        # --- Row pass ---
        print("\n--- [Row Pass] ---")
        changed_row = False
        pending_rows = sorted(list(dirty_rows))
        dirty_rows.clear()

        for r in pending_rows:
            line = board[r, :].copy()
            state = np.concatenate([row_hints_padded[r], line])
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = q_net(state_t).squeeze(0).cpu().numpy()

            mask = np.zeros(2 * N, dtype=np.float32)
            for i in range(N):
                if line[i] == 0:
                    mask[i] = 1.0
                    mask[N + i] = 1.0

            print(f"Row {r} (Hint: {row_hints[r]}):")
            print(f"  state: {visualise(line)}")

            bad_actions = np.where((mask > 0) & (q <= neg_threshold))[0]
            if len(bad_actions) > 0:
                print("  Confidently bad actions found:")
                for action_idx in bad_actions:
                    j, f = idx_to_action(int(action_idx), N)
                    opp_val = -f
                    opp_char = '#' if opp_val == 1 else '.'
                    if board[r, j] == 0:
                        board[r, j] = opp_val
                        print(f"    a = ({j}, {f:+d}) -> Q = {q[action_idx]:+.4f} (<= {neg_threshold}); taking opposite ({j}, {opp_val:+d}) -> cell becomes '{opp_char}'")
                        dirty_cols.add(j)
                        changed_row = True
                print(f"  Updated Row {r}: {visualise(board[r, :])}")
            else:
                print("  No confident actions.")

        # --- Col pass ---
        print("\n--- [Col Pass] ---")
        changed_col = False
        pending_cols = sorted(list(dirty_cols))
        dirty_cols.clear()

        for c in pending_cols:
            line = board[:, c].copy()
            state = np.concatenate([col_hints_padded[c], line])
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = q_net(state_t).squeeze(0).cpu().numpy()

            mask = np.zeros(2 * N, dtype=np.float32)
            for i in range(N):
                if line[i] == 0:
                    mask[i] = 1.0
                    mask[N + i] = 1.0

            print(f"Col {c} (Hint: {col_hints[c]}):")
            print(f"  state: {visualise(line)}")

            bad_actions = np.where((mask > 0) & (q <= neg_threshold))[0]
            if len(bad_actions) > 0:
                print("  Confidently bad actions found:")
                for action_idx in bad_actions:
                    j, f = idx_to_action(int(action_idx), N)
                    opp_val = -f
                    opp_char = '#' if opp_val == 1 else '.'
                    if board[j, c] == 0:
                        board[j, c] = opp_val
                        print(f"    a = ({j}, {f:+d}) -> Q = {q[action_idx]:+.4f} (<= {neg_threshold}); taking opposite ({j}, {opp_val:+d}) -> cell becomes '{opp_char}'")
                        dirty_rows.add(j)
                        changed_col = True
                print(f"  Updated Col {c}: {visualise(board[:, c])}")
            else:
                print("  No confident actions.")

        print(f"\n--- [Outer Pass {outer} 결과 보드] ---")
        print(render_board_debug(board))
        print()

        if changed_row or changed_col:
            continue

        if (board != 0).all():
            print("Board is completely resolved!")
            break

        if force_count >= max_forces:
            print(f"(reached max_forces={max_forces}; stopping)")
            break

        # --- Force guess step ---
        print("\n*** Deadlock detected! Forcing one cell with highest Q-value ***")
        candidates = []
        for r in range(N):
            if not np.all(board[r, :] != 0):
                candidates.append(("row", r))
        for c in range(N):
            if not np.all(board[:, c] != 0):
                candidates.append(("col", c))

        if not candidates:
            break

        states = []
        for axis, idx in candidates:
            line = board[idx, :].copy() if axis == "row" else board[:, idx].copy()
            hint = row_hints_padded[idx] if axis == "row" else col_hints_padded[idx]
            states.append(np.concatenate([hint, line]))

        states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
        with torch.no_grad():
            q_all = q_net(states_t).cpu().numpy()

        best = (-np.inf, None)
        print("Candidate best actions on each line:")
        for cand_idx, (axis, idx) in enumerate(candidates):
            line = board[idx, :].copy() if axis == "row" else board[:, idx].copy()
            mask = np.zeros(2 * N, dtype=np.float32)
            for i in range(N):
                if line[i] == 0:
                    mask[i] = 1.0
                    mask[N + i] = 1.0

            q = q_all[cand_idx].copy()
            q[mask == 0] = -np.inf
            a = int(np.argmax(q))
            best_q_cand = q[a]

            j, f = idx_to_action(a, N)
            print(f"  {axis.capitalize()} {idx}: best a = ({j}, {f:+d}) with Q = {best_q_cand:+.4f}")

            if best_q_cand > best[0]:
                best = (float(best_q_cand), (cand_idx, a))

        if best[1] is None:
            print("No valid force candidate found!")
            break

        q_val, (cand_idx, action_idx) = best
        axis, line_idx = candidates[cand_idx]
        j, f = idx_to_action(action_idx, N)

        opp_char = '#' if f == 1 else '.'
        if axis == "row":
            board[line_idx, j] = f
            dirty_cols.add(j)
            dirty_rows.add(line_idx)
        else:
            board[j, line_idx] = f
            dirty_rows.add(j)
            dirty_cols.add(line_idx)

        force_count += 1
        print(f"-> Chosen Force guess #{force_count}: {axis.capitalize()} {line_idx}, cell {j} = {f:+d} (Q = {q_val:+.4f}) -> cell becomes '{opp_char}'")
        print()
        print("Updated Board after Forced Guess:")
        print(render_board_debug(board))
        print()

    return board


def run_board_inference(N, row_hints, col_hints, q_net, device):
    env = BoardEnv(N=N)
    env.reset(row_hints=row_hints, col_hints=col_hints)

    max_steps = N * N
    print("=== [초기 보드 상태] ===")
    print(render_board_debug(env.board))
    print()

    for step_i in range(max_steps):
        state = env.get_state()
        mask = env.get_valid_mask()

        if not np.any(mask > 0):
            print("No more valid actions left.")
            break

        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(state_t).squeeze(0).cpu().numpy()

        # Print top 5 confident actions
        masked_q = np.where(mask > 0, q_values, -np.inf)
        sorted_indices = np.argsort(masked_q)[::-1]

        print(f"=== Step {step_i + 1} ===")
        print("Top 5 best valid actions:")
        count = 0
        for idx in sorted_indices:
            if masked_q[idx] == -np.inf or count >= 5:
                break
            cell_idx = idx % (N * N)
            fill_value = 1 if idx < N * N else -1
            r = cell_idx // N
            c = cell_idx % N
            action_type = "Paint" if fill_value == 1 else "Mark X"
            print(f"  a = ({r}, {c}) -> {action_type} (Q = {masked_q[idx]:.4f})")
            count += 1

        action = int(np.argmax(masked_q))
        cell_idx = action % (N * N)
        fill_value = 1 if action < N * N else -1
        r = cell_idx // N
        c = cell_idx % N
        action_type = "Paint" if fill_value == 1 else "Mark X"

        _, _, done = env.step(action)
        print(f"-> Chosen: ({r}, {c}) to {action_type} (Q = {masked_q[action]:.4f})")
        print()
        print("Current Board:")
        print(render_board_debug(env.board))
        print()

        if done:
            print("Episode terminal reached!")
            break

    return env.board


def main():
    parser = make_base_parser()
    parser.add_argument("--puzzle", type=str, default=None,
                        help="퍼즐 YAML 파일 경로")
    parser.add_argument("--row_hints", type=str, default=None,
                        help="Row hints (세미콜론으로 구분). 예: '1,3;2;1,1'")
    parser.add_argument("--col_hints", type=str, default=None,
                        help="Col hints (세미콜론으로 구분). 예: '4;1;2,1'")
    parser.add_argument("--load_path", type=str, default=None,
                        help="체크포인트 경로")
    known_args, remaining = parser.parse_known_args()

    config = load_config(known_args.config, remaining)
    device = resolve_device(config["device"])

    # 퍼즐 로드
    if known_args.puzzle:
        puzzle = load_puzzle(known_args.puzzle)
        row_hints = puzzle["row_hints"]
        col_hints = puzzle["col_hints"]
        N = puzzle["N"]
    elif known_args.row_hints and known_args.col_hints:
        row_hints = parse_hints_str(known_args.row_hints)
        col_hints = parse_hints_str(known_args.col_hints)
        if len(row_hints) != len(col_hints):
            raise ValueError(f"row_hints({len(row_hints)}) != col_hints({len(col_hints)})")
        N = len(row_hints)
    else:
        raise ValueError("--puzzle 또는 --row_hints/--col_hints를 지정하세요.")

    # Config의 N과 일치 확인
    config_N = config["env"]["N"]
    if config_N != N:
        print(f"[warn] config N={config_N} != puzzle N={N}. Using puzzle N={N}.")
        config["env"]["N"] = N

    # 모델 로드
    load_path = known_args.load_path
    if load_path is None:
        exp_dir = get_experiment_dir(config)
        load_path = str(exp_dir / "checkpoints" / "latest.pt")

    ckpt = torch.load(load_path, map_location=device)
    q_net = create_model(config).to(device)
    q_net.load_state_dict(ckpt["model_state_dict"])
    q_net.eval()

    solver_type = config["solver"]["type"]
    neg_threshold = config["solver"]["neg_threshold"]
    max_outer_iters = config["solver"]["max_outer_iters"]
    max_forces = config["solver"].get("max_forces")
    if max_forces is None:
        max_forces = N * N

    print(f"Running 2D inference visualization using solver: {solver_type}")
    print(f"Loading weights from {load_path}")
    print()

    if solver_type == "line":
        board = run_line_inference(N, row_hints, col_hints, q_net, device, neg_threshold, max_outer_iters, max_forces)
    elif solver_type == "board":
        board = run_board_inference(N, row_hints, col_hints, q_net, device)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    print("====================== Final Solved Board ======================")
    print(render_board(board, row_hints, col_hints))
    print()


if __name__ == "__main__":
    main()
