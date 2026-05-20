"""
평가 스크립트 — config 기반으로 다양한 solver를 평가.

Usage:
    uv run python scripts/evaluate.py --config configs/dqn_10x10.yaml --num_puzzles 200
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nonogram.config import load_config, make_base_parser, resolve_device, get_experiment_dir
from nonogram.models.registry import create_model
from nonogram.solvers.registry import create_solver
from nonogram.utils.visualization import render_board_debug


def extract_runs(line):
    """줄에서 +1의 run lengths 추출."""
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


def hints_equal(h1, h2):
    n1 = [int(h) for h in h1 if int(h) > 0]
    n2 = [int(h) for h in h2 if int(h) > 0]
    return n1 == n2


def evaluate_solution(solver_board, row_hints, col_hints, gt_board):
    N = solver_board.shape[0]
    num_unknown = int((solver_board == 0).sum())

    hint_satisfied = False
    if num_unknown == 0:
        produced_rows, produced_cols = extract_hints_from_board(solver_board)
        hint_satisfied = (
            all(hints_equal(produced_rows[i], row_hints[i]) for i in range(N))
            and all(hints_equal(produced_cols[j], col_hints[j]) for j in range(N))
        )

    exact_match = bool(np.array_equal(solver_board, gt_board))
    agree = int(((solver_board == gt_board) & (solver_board != 0)).sum())

    return {
        "hint_satisfied": hint_satisfied,
        "exact_match": exact_match,
        "num_unknown": num_unknown,
        "num_total": N * N,
        "num_agree": agree,
    }


def main():
    parser = make_base_parser()
    parser.add_argument("--num_puzzles", type=int, default=200)
    parser.add_argument("--show_failures", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_path", type=str, default=None,
                        help="체크포인트 경로. 없으면 실험 디렉터리에서 최신 모델 로드.")
    known_args, remaining = parser.parse_known_args()

    config = load_config(known_args.config, remaining)
    device = resolve_device(config["device"])
    N = config["env"]["N"]

    rng = np.random.default_rng(known_args.seed)
    torch.manual_seed(known_args.seed)

    # 모델 로드
    load_path = known_args.load_path
    if load_path is None:
        exp_dir = get_experiment_dir(config)
        load_path = str(exp_dir / "checkpoints" / "latest.pt")

    ckpt = torch.load(load_path, map_location=device)

    # PPO 에이전트를 Q-net 형식으로 래핑하는 헬퍼
    class PPOToQNetWrapper(torch.nn.Module):
        def __init__(self, ppo_network):
            super().__init__()
            self.ppo_network = ppo_network
        def forward(self, x):
            logits, _ = self.ppo_network(x)
            return logits

    # 모델 로드
    load_path = known_args.load_path
    if load_path is None:
        exp_dir = get_experiment_dir(config)
        load_path = str(exp_dir / "checkpoints" / "latest.pt")

    ckpt = torch.load(load_path, map_location=device)

    # 모델 생성 및 로드
    agent_type = config["agent"]["type"]
    if agent_type == "ppo":
        from nonogram.agents.ppo import ActorCriticNetwork
        env_type = config["env"].get("type", "line")
        model_type = config["model"].get("type", "mlp")
        
        raw_net = ActorCriticNetwork(
            N=N,
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            env_type=env_type,
            model_type=model_type
        ).to(device)
        raw_net.load_state_dict(ckpt["model_state_dict"])
        q_net = PPOToQNetWrapper(raw_net)
    else:
        q_net = create_model(config).to(device)
        q_net.load_state_dict(ckpt["model_state_dict"])
        
    q_net.eval()

    # Solver 생성
    solver = create_solver(config, q_net=q_net, device=device)

    print(f"Evaluating {load_path} ({config['model']['type']}+{config['agent']['type']}) "
          f"on {known_args.num_puzzles} random {N}x{N} puzzles\n")

    # 평가 루프
    n_hint_ok = 0
    n_exact = 0
    n_fully_decided = 0
    total_unknown = 0
    total_agree = 0
    total_cells = 0
    solve_times = []
    failures = []

    t_start = time.time()
    for idx in range(1, known_args.num_puzzles + 1):
        gt = rng.choice([-1, 1], size=(N, N)).astype(np.int64)
        row_hints, col_hints = extract_hints_from_board(gt)

        t0 = time.time()
        solver_board = solver.solve(N=N, row_hints=row_hints, col_hints=col_hints, verbose=True)
        solve_times.append(time.time() - t0)

        stats = evaluate_solution(solver_board, row_hints, col_hints, gt)

        if stats["hint_satisfied"]:
            n_hint_ok += 1
        if stats["exact_match"]:
            n_exact += 1
        if stats["num_unknown"] == 0:
            n_fully_decided += 1
        total_unknown += stats["num_unknown"]
        total_agree += stats["num_agree"]
        total_cells += stats["num_total"]

        if not stats["hint_satisfied"] and len(failures) < known_args.show_failures:
            failures.append((idx, gt.copy(), solver_board.copy(), row_hints, col_hints))

        if idx % known_args.log_every == 0 or idx == known_args.num_puzzles:
            elapsed = time.time() - t_start
            print(f"[{idx:4d}/{known_args.num_puzzles}]  "
                  f"hint_ok={n_hint_ok/idx:.3f}  "
                  f"exact={n_exact/idx:.3f}  "
                  f"decided={n_fully_decided/idx:.3f}  "
                  f"avg_unknown={total_unknown/idx:.2f}  "
                  f"agree={total_agree/total_cells:.3f}  "
                  f"({elapsed:.1f}s)")

    # 최종 보고
    print("\n" + "=" * 60)
    print("Final results")
    print("=" * 60)
    n = known_args.num_puzzles
    print(f"hint-satisfied rate      : {n_hint_ok}/{n}  ({n_hint_ok/n:.3%})")
    print(f"exact-match rate         : {n_exact}/{n}  ({n_exact/n:.3%})")
    print(f"fully decided rate       : {n_fully_decided}/{n}  ({n_fully_decided/n:.3%})")
    print(f"avg unknown cells / board: {total_unknown/n:.2f}  / {N*N}")
    print(f"cell-level agreement     : {total_agree/total_cells:.3%}  "
          f"({total_agree}/{total_cells})")
    print(f"avg solve time / puzzle  : {np.mean(solve_times)*1000:.1f} ms  "
          f"(total {sum(solve_times):.1f}s)")

    if failures:
        print("\n" + "=" * 60)
        print(f"Showing {len(failures)} failed puzzle(s):")
        print("=" * 60)
        for idx, gt, sol, rh, ch in failures:
            print(f"\n--- puzzle #{idx} ---")
            print("row hints:", [' '.join(str(x) for x in h) for h in rh])
            print("col hints:", [' '.join(str(x) for x in h) for h in ch])
            print("ground truth:")
            print(render_board_debug(gt))
            print("solver output (? = unknown, . = X, # = filled):")
            print(render_board_debug(sol))

    # 결과 저장
    try:
        import json
        exp_dir = get_experiment_dir(config)
        result_path = exp_dir / "eval" / f"results_seed{known_args.seed}.json"
        with open(result_path, "w") as f:
            json.dump({
                "hint_satisfied_rate": n_hint_ok / n,
                "exact_match_rate": n_exact / n,
                "fully_decided_rate": n_fully_decided / n,
                "avg_unknown": total_unknown / n,
                "cell_agreement": total_agree / total_cells,
                "avg_solve_time_ms": np.mean(solve_times) * 1000,
                "num_puzzles": n,
                "N": N,
            }, f, indent=2)
        print(f"\nResults saved to {result_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
