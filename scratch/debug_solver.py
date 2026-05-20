import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nonogram.config import load_config
from nonogram.models.registry import create_model
from nonogram.solvers.line_solver import LineSolver
from nonogram.env.feasibility import is_feasible
from scripts.evaluate import extract_hints_from_board, evaluate_solution
from nonogram.utils.visualization import render_board_debug

def main():
    config_path = "configs/dqn_10x10_super_1d.yaml"
    load_path = "runs/double_dqn_10x10_super_1d_double_dqn_dueling_N10/checkpoints/best.pt"
    
    config = load_config(config_path)
    device = torch.device("cpu")
    N = config["env"]["N"]
    
    ckpt = torch.load(load_path, map_location=device)
    q_net = create_model(config).to(device)
    q_net.load_state_dict(ckpt["model_state_dict"])
    q_net.eval()
    
    solver = LineSolver(config, q_net=q_net, device=device)
    
    rng = np.random.default_rng(0)
    gt = rng.choice([-1, 1], size=(N, N)).astype(np.int64)
    row_hints, col_hints = extract_hints_from_board(gt)
    
    print("=== Full solve with verbose ===")
    solver_board = solver.solve(N=N, row_hints=row_hints, col_hints=col_hints, verbose=True)
    
    print(f"\nSearch Calls: {solver.search_calls}")
    print(f"Backtracks: {solver.backtrack_count}")
    print(f"Aborted: {solver.aborted}")
    print(f"Min unknowns seen: {solver.min_unknowns}")
    
    stats = evaluate_solution(solver_board, row_hints, col_hints, gt)
    print(f"Stats: {stats}")
    print("\nSolver output:")
    print(render_board_debug(solver_board))
    print("\nGround truth:")
    print(render_board_debug(gt))
    
    # best_feasible_board 확인
    print(f"\nBest feasible board unknowns: {int((solver.best_feasible_board == 0).sum())}")
    print("Best feasible board:")
    print(render_board_debug(solver.best_feasible_board))

if __name__ == "__main__":
    main()
