"""
Training script for an RL-based 1D nonogram solver (DQN, gamma = 1).

Data generation per episode:
  1. Sample a random ground-truth row (each cell uniformly in {-1, +1}) and
     extract its hint -- this guarantees the hint is realisable.
  2. Run one episode in one of two modes (or a per-step mix of them):
       - "towards": pick a still-empty cell uniformly at random and fill it
                    with the ground-truth value (guaranteed +1 reward path).
       - "random" : pick a uniformly random valid action.
     Episodes alternate or mix based on --mode / --towards_ratio / --mix_prob.

Early stopping:
  After every action we run a feasibility check on the partial row. If the
  remaining empty cells cannot possibly be completed to satisfy the hint, we
  terminate immediately with reward -1. This keeps useless dead-end rollouts
  short and lets the Q-network see -1 transitions earlier.

Usage example:
    python train.py --N 5 --num_episodes 30000 --save_path qnet_N5.pt
"""

import argparse
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import (
    QNetwork,
    apply_action,
    check_hint_match,
    extract_hint_from_blocks,
    get_blocks,
    get_hint_part,
    hint_length,
    idx_to_action,
    is_feasible,
    is_terminal,
    action_to_idx,
    valid_actions_mask,
)


def parse_args():
    parser = argparse.ArgumentParser()
    # Problem size
    parser.add_argument('--N', type=int, default=5, help='Row length.')
    # Training schedule
    parser.add_argument('--num_episodes', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--min_buffer', type=int, default=2000,
                        help='Wait until the replay buffer has this many transitions before training.')
    parser.add_argument('--train_freq', type=int, default=1,
                        help='Run a gradient step every this many env steps.')
    parser.add_argument('--target_update_freq', type=int, default=500,
                        help='Hard-copy q_net -> target_net every this many env steps.')
    # Optimisation
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    # Data-collection policy
    parser.add_argument('--mode', type=str, default='alternate',
                        choices=['alternate', 'mix'],
                        help='alternate: each episode is purely "towards" or purely "random".'
                             ' mix: each step inside an episode is independently towards/random.')
    parser.add_argument('--towards_ratio', type=float, default=0.5,
                        help='[alternate mode] probability an episode uses the "towards" policy.')
    parser.add_argument('--mix_prob', type=float, default=0.5,
                        help='[mix mode] per-step probability of picking the "towards" action.')
    parser.add_argument('--no_early_stop', action='store_true',
                        help='Disable feasibility-based early stopping.')
    # Logging / checkpointing
    parser.add_argument('--save_path', type=str, default='qnet.pt')
    # LQL (arXiv:2605.05812)
    parser.add_argument('--lql_weight', type=float, default=0.2)
    parser.add_argument('--lql_n_step', type=int, default=3)
    return parser.parse_args()


def generate_sample(N):
    """Uniformly random row in {-1, +1}^N."""
    return np.random.choice([-1, 1], size=N).astype(np.int64)


def select_action_towards(state, sample, N):
    """Pick a random still-empty cell and fill it with the ground-truth value."""
    blocks = get_blocks(state, N)
    empty = [i for i in range(N) if blocks[i] == 0]
    if not empty:
        return None
    i = random.choice(empty)
    f = int(sample[i])
    return action_to_idx(i, f, N)


def select_action_random(state, N):
    """Pick a uniformly random legal action."""
    mask = valid_actions_mask(state, N)
    valid = np.where(mask > 0)[0]
    if len(valid) == 0:
        return None
    return int(random.choice(valid))


def step_env(state, action_idx, hint_part, N, use_early_stop):
    """Apply an action; return (next_state, reward, done)."""
    next_state = apply_action(state, action_idx, N)
    next_blocks = get_blocks(next_state, N)

    if is_terminal(next_state, N):
        reward = 1.0 if check_hint_match(next_blocks, hint_part, N) else -1.0
        return next_state, reward, True

    if use_early_stop and not is_feasible(next_blocks, hint_part, N):
        return next_state, -1.0, True

    return next_state, 0.0, False


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    N = args.N
    k = hint_length(N)
    use_early_stop = not args.no_early_stop

    q_net = QNetwork(N, args.hidden_dim).to(device)
    target_net = QNetwork(N, args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    buffer = deque(maxlen=args.buffer_size)

    step_count = 0
    reward_history = []
    success_history = []

    for episode in range(args.num_episodes):
        # 1) generate a random sample and extract the hint
        sample = generate_sample(N)
        hint_padded = extract_hint_from_blocks(sample, N)
        state = np.array(list(hint_padded) + [0] * N, dtype=np.float32)
        hint_part = state[:k]  # immutable throughout the episode

        # 2) decide data-collection policy for this episode
        episode_use_towards = random.random() < args.towards_ratio  # only used in 'alternate'

        total_reward = 0.0
        done = False
        episode_history = []
        while not done:
            # ---- choose action ----
            if args.mode == 'mix':
                if random.random() < args.mix_prob:
                    a = select_action_towards(state, sample, N)
                else:
                    a = select_action_random(state, N)
            else:  # 'alternate'
                if episode_use_towards:
                    a = select_action_towards(state, sample, N)
                else:
                    a = select_action_random(state, N)

            if a is None:
                break

            next_state, reward, done = step_env(state, a, hint_part, N, use_early_stop)
            episode_history.append((state.copy(), a, reward, next_state.copy(), float(done)))
            state = next_state
            total_reward += reward
            step_count += 1

            # ---- training step ----
            if len(buffer) >= args.min_buffer and step_count % args.train_freq == 0:
                batch = random.sample(buffer, args.batch_size)
                b_states, b_actions, b_rewards, b_next_states, b_dones, b_futures = zip(*batch)

                b_states_t = torch.tensor(np.array(b_states), dtype=torch.float32, device=device)
                b_actions_t = torch.tensor(b_actions, dtype=torch.long, device=device)
                b_rewards_t = torch.tensor(b_rewards, dtype=torch.float32, device=device)
                b_next_t = torch.tensor(np.array(b_next_states), dtype=torch.float32, device=device)
                b_dones_t = torch.tensor(b_dones, dtype=torch.float32, device=device)
                b_futures_t = torch.tensor(np.array(b_futures), dtype=torch.float32, device=device)

                q_pred = q_net(b_states_t).gather(1, b_actions_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # Target Q (Standard)
                    next_q = target_net(b_next_t)
                    next_masks = np.stack([valid_actions_mask(s, N) for s in b_next_states])
                    next_masks_t = torch.tensor(next_masks, dtype=torch.float32, device=device)
                    next_q = next_q.masked_fill(next_masks_t == 0, -1e9)
                    next_q_max = next_q.max(dim=1).values
                    target = b_rewards_t + (1.0 - b_dones_t) * next_q_max
                    
                    # Target Q for LQL (Future Value)
                    next_q_fut = target_net(b_futures_t)
                    fut_masks = np.stack([valid_actions_mask(s, N) for s in b_futures])
                    fut_masks_t = torch.tensor(fut_masks, dtype=torch.float32, device=device)
                    next_q_fut = next_q_fut.masked_fill(fut_masks_t == 0, -1e9)
                    v_future = next_q_fut.max(dim=1).values

                # Standard MSE Loss
                mse_loss = nn.functional.mse_loss(q_pred, target)
                
                # LQL Loss: Q(s_t, a_t) >= V(s_{t+n})
                lql_loss = torch.mean(torch.nn.functional.relu(v_future - q_pred)**2)
                
                loss = mse_loss + args.lql_weight * lql_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

        # 에피소드 종료 후 n-step 매칭하여 버퍼에 저장
        for i in range(len(episode_history)):
            s, a, r, ns, d = episode_history[i]
            f_idx = min(i + args.lql_n_step, len(episode_history) - 1)
            future_s = episode_history[f_idx][3] # next_state of future step
            buffer.append((s, a, r, ns, d, future_s))

            # ---- target network sync ----
            if step_count % args.target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())

        reward_history.append(total_reward)
        success_history.append(1.0 if total_reward > 0 else 0.0)

        if (episode + 1) % args.log_freq == 0:
            window = args.log_freq
            avg_r = float(np.mean(reward_history[-window:]))
            avg_s = float(np.mean(success_history[-window:]))
            print(f"[ep {episode+1:6d}/{args.num_episodes}] "
                  f"avg_reward={avg_r:+.3f}  success_rate={avg_s:.3f}  "
                  f"buffer={len(buffer)}  steps={step_count}")

    # save
    torch.save({
        'model_state_dict': q_net.state_dict(),
        'N': N,
        'hidden_dim': args.hidden_dim,
    }, args.save_path)
    print(f"\nModel saved to {os.path.abspath(args.save_path)}")


if __name__ == '__main__':
    main()