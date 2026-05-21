import argparse
import os
import random
import time

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from models_2d import AlphaZeroNonogramNet
from mcts_2d import AlphaZeroMCTS
from utils import extract_hint_from_blocks, check_hint_match, encode_hint, hint_length

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='Path to YAML config file')
    parser.add_argument('--N', type=int, default=5, help='Board size N x N')
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--num_simulations', type=int, default=100, help='MCTS simulations per step')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--buffer_size', type=int, default=50000)
    parser.add_argument('--min_buffer', type=int, default=1000)
    parser.add_argument('--sync_freq', type=int, default=5, help='Sync weights every N tasks')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--save_path', type=str, default='alphazero_N5.pt')
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--episodes_per_task', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    # MCTS parameters
    parser.add_argument('--c_puct', type=float, default=2.5)
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3)
    parser.add_argument('--dirichlet_epsilon', type=float, default=0.25)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--temp_threshold_ratio', type=float, default=0.5)
    parser.add_argument('--action_masking', action='store_true', default=True)
    # Training parameters
    # 1D guided pretraining
    parser.add_argument('--pretrain_1d_model', type=str, default='', help='Path to 1D Q-network model for 1D-guided BC pretraining')
    parser.add_argument('--pretrain_1d_episodes', type=int, default=1000, help='Number of episodes to generate using 1D solver')
    parser.add_argument('--pretrain_1d_lr', type=float, default=0.001)
    parser.add_argument('--pretrain_1d_hidden_dim', type=int, default=128, help='Hidden dim of the 1D model')
    
    parser.add_argument('--value_loss_weight', type=float, default=0.5)
    parser.add_argument('--lql_weight', type=float, default=0.2)
    parser.add_argument('--lql_n_step', type=int, default=3)
    parser.add_argument('--max_grad_updates_per_task', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reward_shaping', action='store_true', default=False)
    parser.add_argument('--pretrain_bc', action='store_true', default=False)
    parser.add_argument('--pretrain_episodes', type=int, default=2000)
    parser.add_argument('--pretrain_lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if 'env' in cfg and 'N' in cfg['env']: args.N = cfg['env']['N']
        if 'model' in cfg:
            args.hidden_dim = cfg['model'].get('hidden_dim', args.hidden_dim)
            args.num_heads = cfg['model'].get('num_heads', args.num_heads)
            args.num_layers = cfg['model'].get('num_layers', args.num_layers)
        if 'agent' in cfg:
            args.num_simulations = cfg['agent'].get('num_simulations', args.num_simulations)
            args.batch_size = cfg['agent'].get('batch_size', args.batch_size)
            args.buffer_size = cfg['agent'].get('buffer_size', args.buffer_size)
            args.min_buffer = cfg['agent'].get('min_buffer', args.min_buffer)
            args.sync_freq = cfg['agent'].get('sync_freq', args.sync_freq)
            args.lr = cfg['agent'].get('lr', args.lr)
            args.weight_decay = cfg['agent'].get('weight_decay', args.weight_decay)
            args.num_workers = cfg['agent'].get('num_workers', args.num_workers)
            args.episodes_per_task = cfg['agent'].get('episodes_per_task', args.episodes_per_task)
            args.max_grad_updates_per_task = cfg['agent'].get('max_grad_updates_per_task', args.max_grad_updates_per_task)
        if 'mcts' in cfg:
            args.c_puct = cfg['mcts'].get('c_puct', args.c_puct)
            args.dirichlet_alpha = cfg['mcts'].get('dirichlet_alpha', args.dirichlet_alpha)
            args.dirichlet_epsilon = cfg['mcts'].get('dirichlet_epsilon', args.dirichlet_epsilon)
            args.temperature = cfg['mcts'].get('temperature', args.temperature)
            args.temp_threshold_ratio = cfg['mcts'].get('temp_threshold_ratio', args.temp_threshold_ratio)
            args.action_masking = cfg['mcts'].get('action_masking', args.action_masking)
        if 'training' in cfg:
            args.num_episodes = cfg['training'].get('num_episodes', args.num_episodes)
            args.seed = cfg['training'].get('seed', args.seed)
            args.log_freq = cfg['training'].get('log_freq', args.log_freq)
            args.value_loss_weight = cfg['training'].get('value_loss_weight', args.value_loss_weight)
            args.lql_weight = cfg['training'].get('lql_weight', args.lql_weight)
            args.lql_n_step = cfg['training'].get('lql_n_step', args.lql_n_step)
            args.reward_shaping = cfg['training'].get('reward_shaping', args.reward_shaping)
            args.pretrain_bc = cfg['training'].get('pretrain_bc', args.pretrain_bc)
            args.pretrain_episodes = cfg['training'].get('pretrain_episodes', args.pretrain_episodes)
            args.pretrain_lr = cfg['training'].get('pretrain_lr', args.pretrain_lr)
            args.pretrain_1d_model = cfg['training'].get('pretrain_1d_model', args.pretrain_1d_model)
            args.pretrain_1d_episodes = cfg['training'].get('pretrain_1d_episodes', args.pretrain_1d_episodes)
            args.pretrain_1d_lr = cfg['training'].get('pretrain_1d_lr', args.pretrain_1d_lr)
            args.pretrain_1d_hidden_dim = cfg['training'].get('pretrain_1d_hidden_dim', args.pretrain_1d_hidden_dim)
        if 'experiment' in cfg:
            if 'save_path' in cfg['experiment']:
                args.save_path = cfg['experiment']['save_path']
            if cfg['experiment'].get('resume', False):
                args.resume = True
                
    return args


def evaluate_greedy_accuracy(network, device, N, num_puzzles=50):
    """MCTS 없이 오직 2D 네트워크의 Greedy Policy만으로 노노그램 퍼즐을 풀 수 있는지 평가."""
    import numpy as np
    import torch
    from utils import is_feasible, check_hint_match
    
    network.eval()
    k = (N + 1) // 2
    success_count = 0
    
    with torch.no_grad():
        for _ in range(num_puzzles):
            gt_board, r_hints, c_hints = generate_2d_game(N)
            board = np.zeros((N, N), dtype=np.int64)
            
            # 패딩 힌트 생성
            padded_rh = []
            for h in r_hints:
                actual = [int(x) for x in h if int(x) > 0]
                pad_len = k - len(actual)
                padded_rh.append([0] * pad_len + actual)
            padded_rh_t = torch.tensor(padded_rh, dtype=torch.long, device=device).unsqueeze(0)
            
            padded_ch = []
            for h in c_hints:
                actual = [int(x) for x in h if int(x) > 0]
                pad_len = k - len(actual)
                padded_ch.append([0] * pad_len + actual)
            padded_ch_t = torch.tensor(padded_ch, dtype=torch.long, device=device).unsqueeze(0)
            
            failed = False
            for step in range(N * N):
                empty_cells = np.argwhere(board == 0)
                if len(empty_cells) == 0:
                    break
                
                # Feasibility 마스킹
                mask = np.zeros(2 * N * N, dtype=np.float32)
                for r, c in empty_cells:
                    # Paint (1)
                    board[r, c] = 1
                    if is_feasible(board[r, :], r_hints[r], N) and is_feasible(board[:, c], c_hints[c], N):
                        mask[r * N + c] = 1.0
                    # X (-1)
                    board[r, c] = -1
                    if is_feasible(board[r, :], r_hints[r], N) and is_feasible(board[:, c], c_hints[c], N):
                        mask[N * N + r * N + c] = 1.0
                    board[r, c] = 0  # Revert
                
                if not np.any(mask > 0):
                    failed = True
                    break
                
                board_t = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
                policy_logits, _ = network(board_t, padded_rh_t, padded_ch_t)
                
                logits_flat = policy_logits.reshape(2 * N * N).cpu().numpy()
                logits_flat = np.where(mask > 0, logits_flat, -np.inf)
                
                action = int(np.argmax(logits_flat))
                cell_idx = action % (N * N)
                fill_value = 1 if action < N * N else -1
                
                r = cell_idx // N
                c = cell_idx % N
                board[r, c] = fill_value
                
            if not failed and np.all(board != 0):
                row_hints_clean = [[int(x) for x in h if int(x) > 0] for h in r_hints]
                col_hints_clean = [[int(x) for x in h if int(x) > 0] for h in c_hints]
                
                solved = True
                for i in range(N):
                    if not check_hint_match(board[i, :], row_hints_clean[i], N):
                        solved = False
                        break
                    if not check_hint_match(board[:, i], col_hints_clean[i], N):
                        solved = False
                        break
                if solved:
                    success_count += 1
                    
    network.train()
    return success_count / num_puzzles


# --- 1D-Guided Trajectory Generation Helper ---
def generate_1d_guided_trajectory(N, q_net, device, row_hints, col_hints, gt_board):
    import numpy as np
    import torch
    
    board = np.zeros((N, N), dtype=np.int64)
    trajectory = []
    
    # 1D Q-net을 위한 힌트 인코딩
    row_hints_padded = [encode_hint(h, N) for h in row_hints]
    col_hints_padded = [encode_hint(h, N) for h in col_hints]
    
    # 채워야 할 총 셀의 수 N*N
    total_steps = N * N
    
    for step in range(total_steps):
        # 현재 미확정 셀 탐색
        undecided = []
        for r in range(N):
            for c in range(N):
                if board[r, c] == 0:
                    undecided.append((r, c))
                    
        if not undecided:
            break
            
        # 모든 undecided 셀에 대해 Q-value 수집
        # 배치 연산을 사용하여 빠르게 계산
        # 1. 모든 행(row)에 대해 1D Q-value 계산
        row_states = []
        for r in range(N):
            line = board[r, :].copy()
            hint = row_hints_padded[r]
            state = np.concatenate([hint, line])
            row_states.append(state)
        row_states_t = torch.tensor(np.stack(row_states), dtype=torch.float32, device=device)
        with torch.no_grad():
            q_rows = q_net(row_states_t).cpu().numpy() # (N, 2*N)
            
        # 2. 모든 열(col)에 대해 1D Q-value 계산
        col_states = []
        for c in range(N):
            line = board[:, c].copy()
            hint = col_hints_padded[c]
            state = np.concatenate([hint, line])
            col_states.append(state)
        col_states_t = torch.tensor(np.stack(col_states), dtype=torch.float32, device=device)
        with torch.no_grad():
            q_cols = q_net(col_states_t).cpu().numpy() # (N, 2*N)
            
        # 각 undecided 셀에 대해 가치 평가
        best_val = -np.inf
        best_cell = None
        
        for r, c in undecided:
            target_v = gt_board[r, c]
            
            # 행 기준 액션 인덱스 (target_v가 1이면 c, -1이면 N+c)
            act_r = c if target_v == 1 else N + c
            q_r = q_rows[r, act_r]
            
            # 열 기준 액션 인덱스 (target_v가 1이면 r, -1이면 N+r)
            act_c = r if target_v == 1 else N + r
            q_c = q_cols[c, act_c]
            
            # 행과 열의 Q-value 중 최대값을 해당 셀의 가치로 삼음
            score = max(q_r, q_c)
            
            if score > best_val:
                best_val = score
                best_cell = (r, c, target_v)
                
        if best_cell is None:
            break
            
        r, c, v = best_cell
        
        # 2D 원핫 Target Policy 구성 (shape: 2, N, N)
        pi_tensor = np.zeros((2, N, N), dtype=np.float32)
        v_idx = 0 if v == 1 else 1
        pi_tensor[v_idx, r, c] = 1.0
        
        # 궤적 추가: (현재 보드 상태, row_hints, col_hints, pi_tensor)
        trajectory.append((board.copy(), row_hints, col_hints, pi_tensor))
        
        # 액션 적용
        board[r, c] = v
        
    return trajectory


# --- Game Generation Helper ---
def generate_2d_game(N):
    # 1. Random Ground Truth
    ground_truth = np.random.choice([-1, 1], size=(N, N)).astype(np.int64)
    
    # 2. Extract Hints
    row_hints = [extract_hint_from_blocks(ground_truth[r, :], N) for r in range(N)]
    col_hints = [extract_hint_from_blocks(ground_truth[:, c], N) for c in range(N)]
    
    return ground_truth, row_hints, col_hints


# --- Ray Worker ---
@ray.remote
class AlphaZeroWorker:
    def __init__(self, N, hidden_dim, num_heads, num_layers, num_simulations,
                 c_puct, dirichlet_alpha, dirichlet_epsilon, action_masking, lql_n_step, seed):
        self.N = N
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.action_masking = action_masking
        self.lql_n_step = lql_n_step
        
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Workers always use CPU to avoid GPU memory contention with the learner
        self.device = torch.device('cpu')
        self.k = (N + 1) // 2
        self.network = AlphaZeroNonogramNet(N, self.k, dim=hidden_dim, num_heads=num_heads, num_layers=num_layers).to(self.device)
        self.network.eval()

    def set_weights(self, weights):
        self.network.load_state_dict(weights)

    def play_games(self, num_episodes, temperature=1.0, temp_threshold_ratio=0.5, reward_shaping=False):
        transitions = []
        episode_successes = []
        steps_collected = 0

        for _ in range(num_episodes):
            gt_board, row_hints, col_hints = generate_2d_game(self.N)
            board = np.zeros((self.N, self.N), dtype=np.int64)
            
            mcts = AlphaZeroMCTS(
                self.network, row_hints, col_hints,
                c_puct=self.c_puct,
                num_simulations=self.num_simulations,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_epsilon=self.dirichlet_epsilon,
                action_masking=self.action_masking,
                device=self.device
            )
            
            episode_history = []
            done = False
            z = -1.0 # default outcome is loss
            total_cells = self.N * self.N
            temp_threshold = int(total_cells * temp_threshold_ratio)
            step = 0
            
            while not done:
                # Temperature scheduling: explore early, exploit late
                if step < temp_threshold:
                    current_temp = temperature
                else:
                    current_temp = 0.0  # greedy

                # Run MCTS
                pi_dict = mcts.search(board, temperature=current_temp)
                
                # Check if MCTS gave up (no valid moves)
                if not pi_dict:
                    break
                    
                # Format target Policy (pi) for the network
                # Shape: (2, N, N). index 0 for v=1, index 1 for v=-1
                pi_tensor = np.zeros((2, self.N, self.N), dtype=np.float32)
                actions = list(pi_dict.keys())
                probs = list(pi_dict.values())
                
                for a, p in zip(actions, probs):
                    r, c, v = a
                    v_idx = 0 if v == 1 else 1
                    pi_tensor[v_idx, r, c] = p
                
                # Store step data: (board, r_hints, c_hints, pi)
                episode_history.append((board.copy(), row_hints, col_hints, pi_tensor))
                
                # Sample action based on Pi (already temperature-adjusted by MCTS)
                chosen_idx = np.random.choice(len(actions), p=probs)
                chosen_action = actions[chosen_idx]
                r, c, v = chosen_action
                
                # Apply action
                board[r, c] = v
                steps_collected += 1
                step += 1
                
                # Check terminal
                if np.all(board != 0):
                    done = True
                    # Check if perfectly solved
                    solved = True
                    satisfied_lines = 0
                    for i in range(self.N):
                        if check_hint_match(board[i, :], row_hints[i], self.N):
                            satisfied_lines += 1
                        else:
                            solved = False
                        if check_hint_match(board[:, i], col_hints[i], self.N):
                            satisfied_lines += 1
                        else:
                            solved = False
                    
                    if reward_shaping:
                        rate = satisfied_lines / (2.0 * self.N)
                        z = 2.0 * rate - 1.0
                    else:
                        z = 1.0 if solved else -1.0
                    episode_successes.append(1.0 if solved else 0.0)

            if not done:
                # Gave up before finishing
                if reward_shaping:
                    satisfied_lines = 0
                    for i in range(self.N):
                        if check_hint_match(board[i, :], row_hints[i], self.N): satisfied_lines += 1
                        if check_hint_match(board[:, i], col_hints[i], self.N): satisfied_lines += 1
                    rate = satisfied_lines / (2.0 * self.N)
                    z = 2.0 * rate - 1.0
                else:
                    z = -1.0
                episode_successes.append(0.0)

            # Assign actual outcome Z and future state for LQL to all states in episode history
            for i in range(len(episode_history)):
                b, r_h, c_h, pi = episode_history[i]
                
                # For LQL: find n-step future state (if exists, else terminal)
                # We'll just store the board of the future state
                n_idx = min(i + self.lql_n_step, len(episode_history) - 1)
                future_board = episode_history[n_idx][0]
                
                transitions.append((b, r_h, c_h, pi, z, future_board))

        return transitions, episode_successes, steps_collected


# --- Learner Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def append(self, item):
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def pad_hints_batch(hints_list, k, N):
    """Pad a batch of hint lists to size (B, N, k)"""
    B = len(hints_list)
    padded_batch = np.zeros((B, N, k), dtype=np.int64)
    
    for b in range(B):
        for i in range(N):
            actual = [int(x) for x in hints_list[b][i] if int(x) > 0]
            pad_len = k - len(actual)
            padded_batch[b, i, pad_len:] = actual
    return padded_batch


# --- Main Learner ---
def main():
    args = parse_args()
    ray.init(ignore_reinit_error=True)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"AlphaZero MCTS Training | Using device: {device} | Board size: {args.N}x{args.N}", flush=True)
    print(f"  MCTS sims={args.num_simulations}, c_puct={args.c_puct}, "
          f"dirichlet=({args.dirichlet_alpha}, {args.dirichlet_epsilon})", flush=True)
    print(f"  temperature={args.temperature}, threshold_ratio={args.temp_threshold_ratio}", flush=True)
    print(f"  lr={args.lr}, batch={args.batch_size}, buffer={args.buffer_size}, "
          f"value_loss_weight={args.value_loss_weight}", flush=True)
    print(f"  workers={args.num_workers}, episodes_per_task={args.episodes_per_task}, "
          f"max_grad_updates={args.max_grad_updates_per_task}", flush=True)
    
    k = (args.N + 1) // 2
    
    # Initialize Network
    network = AlphaZeroNonogramNet(
        N=args.N, hint_max_length=k, dim=args.hidden_dim, 
        num_heads=args.num_heads, num_layers=args.num_layers
    ).to(device)
    
    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler: cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_episodes, eta_min=args.lr * 0.01
    )
    
    buffer = ReplayBuffer(args.buffer_size)
    
    # Resume from checkpoint if requested
    episodes_collected = 0
    gradient_steps = 0
    
    if args.resume and os.path.exists(args.save_path):
        print(f"Resuming from {args.save_path}...", flush=True)
        ckpt = torch.load(args.save_path, map_location=device)
        network.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'episodes_collected' in ckpt:
            episodes_collected = ckpt['episodes_collected']
        if 'gradient_steps' in ckpt:
            gradient_steps = ckpt['gradient_steps']
        print(f"  Resumed at ep={episodes_collected}, grad_steps={gradient_steps}", flush=True)

    # Initialize Workers
    workers = [
        AlphaZeroWorker.remote(
            N=args.N, hidden_dim=args.hidden_dim, num_heads=args.num_heads, 
            num_layers=args.num_layers, num_simulations=args.num_simulations,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_epsilon=args.dirichlet_epsilon,
            action_masking=args.action_masking,
            lql_n_step=args.lql_n_step,
            seed=args.seed + i * 100
        ) for i in range(args.num_workers)
    ]

    # --- Behavior Cloning Pre-training (B) ---
    if args.pretrain_1d_model and episodes_collected == 0:
        print(f"\n[1D-Guided BC Pre-training] Loading 1D Q-network from {args.pretrain_1d_model}...", flush=True)
        from nonogram.solvers.line_solver import LineSolver
        from nonogram.models.registry import create_model
        
        ckpt_1d = torch.load(args.pretrain_1d_model, map_location=device)
        
        # Reconstruct the 1D model config dynamically from checkpoint metadata
        model_type = ckpt_1d.get("model_type", "mlp")
        N_1d = ckpt_1d.get("N", args.N)
        hidden_dim_1d = ckpt_1d.get("hidden_dim", args.pretrain_1d_hidden_dim)
        
        temp_config = {
            "env": {"N": N_1d},
            "model": {
                "type": model_type,
                "hidden_dim": hidden_dim_1d,
            }
        }
        if "config" in ckpt_1d:
            temp_config = ckpt_1d["config"]
            
        print(f"Instantiating 1D model structure: type='{model_type}', N={N_1d}, hidden_dim={hidden_dim_1d}", flush=True)
        q_net_1d = create_model(temp_config).to(device)
        q_net_1d.load_state_dict(ckpt_1d['model_state_dict'])
        q_net_1d.eval()
        
        print(f"Generating {args.pretrain_1d_episodes} 1D-guided trajectories...", flush=True)
        
        # 데모 데이터 수집
        bc_dataset = []
        collected_episodes = 0
        
        while collected_episodes < args.pretrain_1d_episodes:
            # 2D 퍼즐 생성
            gt_board, r_hints, c_hints = generate_2d_game(args.N)
            
            # 힌트 clean화
            row_hints_clean = [[int(x) for x in h if int(x) > 0] for h in r_hints]
            col_hints_clean = [[int(x) for x in h if int(x) > 0] for h in c_hints]
            
            # Generate 1D guided trajectory targeting the ground-truth board directly
            traj = generate_1d_guided_trajectory(args.N, q_net_1d, device, row_hints_clean, col_hints_clean, gt_board)
            bc_dataset.extend(traj)
            collected_episodes += 1
            if collected_episodes % 100 == 0 or collected_episodes == args.pretrain_1d_episodes:
                print(f"  Generated {collected_episodes}/{args.pretrain_1d_episodes} episodes...", flush=True)
                
        print(f"Total collected transitions for BC: {len(bc_dataset)}")
        
        if bc_dataset:
            # 수집된 데모 데이터셋으로 BC 학습 시작
            network.train()
            bc_optimizer = optim.Adam(network.parameters(), lr=args.pretrain_1d_lr, weight_decay=args.weight_decay)
            
            # 에포크 단위 학습
            num_epochs = 20
            batch_size = args.batch_size
            num_batches = len(bc_dataset) // batch_size
            
            print(f"Training 2D network using 1D-guided dataset for {num_epochs} epochs ({num_batches} batches/epoch)...", flush=True)
            
            for epoch in range(num_epochs):
                random.shuffle(bc_dataset)
                epoch_loss = 0.0
                epoch_p_loss = 0.0
                epoch_v_loss = 0.0
                
                for b_idx in range(num_batches):
                    batch = bc_dataset[b_idx * batch_size : (b_idx + 1) * batch_size]
                    b_boards, b_r_hints, b_c_hints, b_pis = zip(*batch)
                    
                    # Target Z는 성공 궤적이므로 모두 1.0
                    b_zs = [1.0] * batch_size
                    
                    b_boards_t = torch.tensor(np.array(b_boards), dtype=torch.float32, device=device)
                    b_rh_t = torch.tensor(pad_hints_batch(b_r_hints, k, args.N), dtype=torch.long, device=device)
                    b_ch_t = torch.tensor(pad_hints_batch(b_c_hints, k, args.N), dtype=torch.long, device=device)
                    b_pis_t = torch.tensor(np.array(b_pis), dtype=torch.float32, device=device)
                    b_zs_t = torch.tensor(b_zs, dtype=torch.float32, device=device).unsqueeze(1)
                    
                    policy_logits, value_pred = network(b_boards_t, b_rh_t, b_ch_t)
                    
                    logits_flat = policy_logits.reshape(batch_size, -1)
                    pi_flat = b_pis_t.reshape(batch_size, -1)
                    log_probs = F.log_softmax(logits_flat, dim=-1)
                    policy_loss = -torch.sum(pi_flat * log_probs, dim=-1).mean()
                    value_loss = F.mse_loss(value_pred, b_zs_t)
                    
                    loss = policy_loss + args.value_loss_weight * value_loss
                    
                    bc_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    bc_optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_p_loss += policy_loss.item()
                    epoch_v_loss += value_loss.item()
                    
                val_acc = evaluate_greedy_accuracy(network, device, args.N, num_puzzles=50)
                print(f"  [Epoch {epoch+1}/{num_epochs}] avg_loss={epoch_loss/num_batches:.4f} (p_loss={epoch_p_loss/num_batches:.4f}, v_loss={epoch_v_loss/num_batches:.4f}) | Val Greedy Success Rate: {val_acc:.2%}", flush=True)
                
            print("  → 1D-Guided BC Pre-training completed. 2D network initialized.\n", flush=True)

    elif args.pretrain_bc and episodes_collected == 0:
        print(f"\n[BC Pre-training] Running {args.pretrain_episodes} BC episodes to initialize network weights...", flush=True)
        network.train()
        bc_optimizer = optim.Adam(network.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
        
        num_batches = args.pretrain_episodes // args.batch_size
        
        for b_idx in range(num_batches):
            b_boards = []
            b_row_hints = []
            b_col_hints = []
            b_pis = []
            b_zs = []
            
            for _ in range(args.batch_size):
                gt_board, r_hints, c_hints = generate_2d_game(args.N)
                board = np.zeros((args.N, args.N), dtype=np.int64)
                
                # 미결정된 셀들을 무작위 순서로 나열
                indices = list(range(args.N * args.N))
                random.shuffle(indices)
                
                # 다양한 보드 충전 상태 학습을 위한 임의 스텝 사전 입력
                fill_len = random.randint(0, args.N * args.N - 1)
                for idx in indices[:fill_len]:
                    r, c = idx // args.N, idx % args.N
                    board[r, c] = gt_board[r, c]
                    
                # 타겟 액션 지정
                next_idx = indices[fill_len]
                next_r, next_c = next_idx // args.N, next_idx % args.N
                next_v = gt_board[next_r, next_c]
                
                # 원핫 Policy
                pi_tensor = np.zeros((2, args.N, args.N), dtype=np.float32)
                v_idx = 0 if next_v == 1 else 1
                pi_tensor[v_idx, next_r, next_c] = 1.0
                
                b_boards.append(board.copy())
                b_row_hints.append(r_hints)
                b_col_hints.append(c_hints)
                b_pis.append(pi_tensor)
                b_zs.append(1.0) # 올바른 궤적이므로 가치 target은 +1.0
                
            b_boards_t = torch.tensor(np.array(b_boards), dtype=torch.float32, device=device)
            b_rh_t = torch.tensor(pad_hints_batch(b_row_hints, k, args.N), dtype=torch.long, device=device)
            b_ch_t = torch.tensor(pad_hints_batch(b_col_hints, k, args.N), dtype=torch.long, device=device)
            b_pis_t = torch.tensor(np.array(b_pis), dtype=torch.float32, device=device)
            b_zs_t = torch.tensor(b_zs, dtype=torch.float32, device=device).unsqueeze(1)
            
            policy_logits, value_pred = network(b_boards_t, b_rh_t, b_ch_t)
            
            logits_flat = policy_logits.reshape(args.batch_size, -1)
            pi_flat = b_pis_t.reshape(args.batch_size, -1)
            log_probs = F.log_softmax(logits_flat, dim=-1)
            policy_loss = -torch.sum(pi_flat * log_probs, dim=-1).mean()
            value_loss = F.mse_loss(value_pred, b_zs_t)
            
            loss = policy_loss + args.value_loss_weight * value_loss
            
            bc_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            bc_optimizer.step()
            
            if (b_idx + 1) % 10 == 0:
                print(f"  [BC Pre-train {b_idx+1:3d}/{num_batches}] loss={loss.item():.4f} (p_loss={policy_loss.item():.4f}, v_loss={value_loss.item():.4f})", flush=True)
                
        print("  → Behavior Cloning Pre-training completed. Initial network weights set successfully.\n", flush=True)

    # Initial Sync
    weights_id = ray.put({key: val.cpu() for key, val in network.state_dict().items()})
    ray.get([w.set_weights.remote(weights_id) for w in workers])

    # Dispatch tasks
    pending_tasks = {
        w: w.play_games.remote(
            args.episodes_per_task, args.temperature, args.temp_threshold_ratio, reward_shaping=args.reward_shaping
        ) for w in workers
    }
    
    tasks_completed = 0
    success_history = []
    last_log_ep = episodes_collected
    best_success_rate = 0.0
    recent_policy_losses = []
    recent_value_losses = []
    recent_lql_losses = []

    start_time = time.time()

    while episodes_collected < args.num_episodes:
        ready_refs, _ = ray.wait(list(pending_tasks.values()), num_returns=1, timeout=None)
        
        for ref in ready_refs:
            worker = next(w for w, r in pending_tasks.items() if r == ref)
            transitions, ep_successes, steps = ray.get(ref)
            
            for t in transitions: buffer.append(t)
            
            episodes_collected += len(ep_successes)
            tasks_completed += 1
            success_history.extend(ep_successes)
            
            # Dispatch next task immediately (non-blocking)
            pending_tasks[worker] = worker.play_games.remote(
                args.episodes_per_task, args.temperature, args.temp_threshold_ratio, reward_shaping=args.reward_shaping
            )
            
            # Sync weights periodically
            if tasks_completed % args.sync_freq == 0:
                weights_id = ray.put({key: val.cpu() for key, val in network.state_dict().items()})
                for w in workers:
                    w.set_weights.remote(weights_id)

            # Train network
            if len(buffer) >= args.min_buffer:
                network.train()
                
                # Capped gradient updates to prevent overfitting
                updates_to_do = min(
                    args.max_grad_updates_per_task,
                    max(1, len(ep_successes) * 2)
                )
                
                for _ in range(updates_to_do):
                    batch = buffer.sample(args.batch_size)
                    b_boards, b_r_hints, b_c_hints, b_pis, b_zs, b_futures = zip(*batch)
                    
                    # Convert to tensors
                    b_boards_t = torch.tensor(np.array(b_boards), dtype=torch.float32, device=device)
                    b_rh_t = torch.tensor(pad_hints_batch(b_r_hints, k, args.N), dtype=torch.long, device=device)
                    b_ch_t = torch.tensor(pad_hints_batch(b_c_hints, k, args.N), dtype=torch.long, device=device)
                    b_pis_t = torch.tensor(np.array(b_pis), dtype=torch.float32, device=device)
                    b_zs_t = torch.tensor(b_zs, dtype=torch.float32, device=device).unsqueeze(1)
                    b_futures_t = torch.tensor(np.array(b_futures), dtype=torch.float32, device=device)
                    
                    # Forward pass for current states
                    policy_logits, value_pred = network(b_boards_t, b_rh_t, b_ch_t)
                    
                    # Forward pass for future states (for LQL loss)
                    with torch.no_grad():
                        _, value_future = network(b_futures_t, b_rh_t, b_ch_t)
                    
                    # Policy Loss (Cross Entropy)
                    logits_flat = policy_logits.reshape(args.batch_size, -1)
                    pi_flat = b_pis_t.reshape(args.batch_size, -1)
                    log_probs = F.log_softmax(logits_flat, dim=-1)
                    policy_loss = -torch.sum(pi_flat * log_probs, dim=-1).mean()
                    
                    # Value Loss (MSE)
                    value_loss = F.mse_loss(value_pred, b_zs_t)
                    
                    # LQL Loss (arXiv:2605.05812): V(s_t) >= V(s_{t+n})
                    # Constraint: value_pred >= value_future
                    lql_loss = torch.mean(F.relu(value_future - value_pred)**2)
                    
                    loss = policy_loss + args.value_loss_weight * value_loss + args.lql_weight * lql_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    optimizer.step()
                    
                    gradient_steps += 1
                    recent_policy_losses.append(policy_loss.item())
                    recent_value_losses.append(value_loss.item())
                    recent_lql_losses.append(lql_loss.item())
                
                # Step LR scheduler
                scheduler.step()
            
            # Logging
            if episodes_collected - last_log_ep >= args.log_freq:
                window = min(args.log_freq, len(success_history))
                avg_s = float(np.mean(success_history[-window:])) if window > 0 else 0.0
                elapsed = time.time() - start_time
                
                # Compute average losses
                avg_p_loss = float(np.mean(recent_policy_losses[-100:])) if recent_policy_losses else 0.0
                avg_v_loss = float(np.mean(recent_value_losses[-100:])) if recent_value_losses else 0.0
                avg_lql_loss = float(np.mean(recent_lql_losses[-100:])) if recent_lql_losses else 0.0
                current_lr = optimizer.param_groups[0]['lr']
                
                val_acc = evaluate_greedy_accuracy(network, device, args.N, num_puzzles=50)
                print(f"[ep {episodes_collected:6d}/{args.num_episodes}] "
                      f"success(mcts)={avg_s:.3f}  success(greedy)={val_acc:.3f}  "
                      f"p_loss={avg_p_loss:.4f}  v_loss={avg_v_loss:.4f}  lql_loss={avg_lql_loss:.4f}  "
                      f"lr={current_lr:.6f}  buffer={len(buffer)}  "
                      f"grad={gradient_steps}  time={elapsed:.1f}s", flush=True)
                last_log_ep = episodes_collected
                
                # Save best model
                if avg_s > best_success_rate and avg_s > 0:
                    best_success_rate = avg_s
                    _save_checkpoint(network, optimizer, args, episodes_collected,
                                     gradient_steps, best_success_rate,
                                     args.save_path.replace('.pt', '_best.pt'))
                    print(f"  → New best model saved! (success_rate={avg_s:.3f})", flush=True)

    # Save final AlphaZero Model
    _save_checkpoint(network, optimizer, args, episodes_collected,
                     gradient_steps, best_success_rate, args.save_path)
    print(f"\nAlphaZero Model saved to {os.path.abspath(args.save_path)}", flush=True)
    ray.shutdown()


def _save_checkpoint(network, optimizer, args, episodes_collected,
                     gradient_steps, best_success_rate, path):
    """Save a complete checkpoint for resumable training."""
    torch.save({
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'N': args.N,
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'episodes_collected': episodes_collected,
        'gradient_steps': gradient_steps,
        'best_success_rate': best_success_rate,
    }, path)


if __name__ == '__main__':
    main()
