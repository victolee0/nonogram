"""
풀이 스크립트 — 특정 퍼즐을 풀이.

Usage:
    uv run python scripts/solve.py --config configs/dqn_10x10.yaml --puzzle puzzles/example.yaml
    uv run python scripts/solve.py --config configs/dqn_10x10.yaml \\
        --row_hints "1,1;3;2;1,1;1,2" --col_hints "1,2;1;2,1;2,1;1,1"
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nonogram.config import load_config, make_base_parser, resolve_device, get_experiment_dir
from nonogram.models.registry import create_model
from nonogram.solvers.registry import create_solver
from nonogram.utils.visualization import render_board
from nonogram.utils.puzzle_io import load_puzzle, parse_hints_str


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
    parser.add_argument("--verbose", action="store_true")
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

    # Solver
    solver = create_solver(config, q_net=q_net, device=device)

    print(f"Solving {N}x{N} nonogram using {load_path}")
    print()

    board = solver.solve(N=N, row_hints=row_hints, col_hints=col_hints,
                         verbose=known_args.verbose)

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
        print("(some cells remained unknown)")


if __name__ == "__main__":
    main()
