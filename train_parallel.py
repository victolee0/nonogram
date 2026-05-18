import argparse
import os
import random
import time

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim

from utils import (
    QNetwork,
    apply_action,
    check_hint_match,
    extract_hint_from_blocks,
    get_blocks,
    hint_length,
    is_feasible,
    is_terminal,
    valid_actions_mask,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=5, help='Row length.')
    parser.add_argument('--num_episodes', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--buffer_size', type=int, default=500000)
    parser.add_argument('--min_buffer', type=int, default=2000)
    parser.add_argument('--updates_per_episode', type=float, default=1.0)
    parser.add_argument('--target_update_freq', type=int, default=1000)
    parser.add_argument('--sync_freq', type=int, default=10, help='Sync weights to workers every N tasks.')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay_episodes', type=int, default=50000)
    parser.add_argument('--no_early_stop', action='store_true')
    parser.add_argument('--save_path', type=str, default='qnet_parallel.pt')
    parser.add_argument('--log_freq', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--episodes_per_task', type=int, default=10)
    return parser.parse_args()


# --- Worker Functions ---
def generate_sample(N):
    return np.random.choice([-1, 1], size=N).astype(np.int64)

def step_env(state, action_idx, hint_part, N, use_early_stop):
    next_state = apply_action(state, action_idx, N)
    next_blocks = get_blocks(next_state, N)

    if is_terminal(next_state, N):
        reward = 1.0 if check_hint_match(next_blocks, hint_part, N) else -1.0
        return next_state, reward, True

    if use_early_stop and not is_feasible(next_blocks, hint_part, N):
        return next_state, -1.0, True

    return next_state, 0.0, False

@ray.remote
class RolloutWorker:
    def __init__(self, N, hidden_dim, use_early_stop, seed):
        self.N = N
        self.k = hint_length(N)
        self.use_early_stop = use_early_stop
        
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Load local Q-network for epsilon-greedy exploration
        self.q_net = QNetwork(N, hidden_dim)
        self.q_net.eval()
        self.epsilon = 1.0

    def set_weights(self, weights, epsilon):
        self.q_net.load_state_dict(weights)
        self.epsilon = epsilon

    def select_action(self, state):
        mask = valid_actions_mask(state, self.N)
        valid = np.where(mask > 0)[0]
        
        if random.random() < self.epsilon:
            return int(random.choice(valid))
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_vals = self.q_net(state_t).squeeze(0)
                mask_t = torch.tensor(mask, dtype=torch.float32)
                q_vals = q_vals.masked_fill(mask_t == 0, -1e9)
                return int(q_vals.argmax().item())

    def collect_episodes(self, num_episodes):
        transitions = []
        episode_returns = []
        episode_successes = []
        steps_collected = 0

        for _ in range(num_episodes):
            sample = generate_sample(self.N)
            hint_padded = extract_hint_from_blocks(sample, self.N)
            state = np.array(list(hint_padded) + [0] * self.N, dtype=np.float32)
            hint_part = state[:self.k]

            total_reward = 0.0
            done = False

            while not done:
                a = self.select_action(state)
                next_state, reward, done = step_env(state, a, hint_part, self.N, self.use_early_stop)
                transitions.append((state.copy(), a, reward, next_state.copy(), float(done)))
                state = next_state
                total_reward += reward
                steps_collected += 1

            episode_returns.append(total_reward)
            episode_successes.append(1.0 if total_reward > 0 else 0.0)

        return transitions, episode_returns, episode_successes, steps_collected


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

def main():
    args = parse_args()
    
    ray.init(ignore_reinit_error=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    N = args.N
    use_early_stop = not args.no_early_stop

    # Initialize Workers
    workers = [
        RolloutWorker.remote(
            N=N, 
            hidden_dim=args.hidden_dim,
            use_early_stop=use_early_stop, 
            seed=args.seed + i * 1000
        ) for i in range(args.num_workers)
    ]
    
    # Initialize Learner Network & Buffer
    q_net = QNetwork(N, args.hidden_dim).to(device)
    target_net = QNetwork(N, args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)

    # Initial Sync
    weights_id = ray.put({k: v.cpu() for k, v in q_net.state_dict().items()})
    ray.get([w.set_weights.remote(weights_id, args.eps_start) for w in workers])

    # Dispatch initial tasks
    pending_tasks = {w: w.collect_episodes.remote(args.episodes_per_task) for w in workers}
    
    episodes_collected = 0
    total_steps_collected = 0
    gradient_steps = 0
    tasks_completed = 0
    reward_history = []
    success_history = []
    last_log_ep = 0

    import time
    start_time = time.time()

    # Main Actor-Learner Loop
    while episodes_collected < args.num_episodes:
        ready_refs, _ = ray.wait(list(pending_tasks.values()), num_returns=1, timeout=None)
        
        for ref in ready_refs:
            worker = next(w for w, r in pending_tasks.items() if r == ref)
            transitions, ep_returns, ep_successes, ep_steps = ray.get(ref)
            tasks_completed += 1
            
            # Store in buffer
            for t in transitions:
                buffer.append(t)
            
            episodes_collected += len(ep_returns)
            total_steps_collected += ep_steps
            reward_history.extend(ep_returns)
            success_history.extend(ep_successes)
            
            # Update epsilon
            epsilon = max(args.eps_end, args.eps_start - (args.eps_start - args.eps_end) * (episodes_collected / args.eps_decay_episodes))
            
            # Sync weights periodically
            if tasks_completed % args.sync_freq == 0:
                weights_id = ray.put({k: v.cpu() for k, v in q_net.state_dict().items()})
                # Don't block learner, use fire-and-forget for setting weights
                for w in workers:
                    w.set_weights.remote(weights_id, epsilon)

            # Dispatch next task to the same worker
            pending_tasks[worker] = worker.collect_episodes.remote(args.episodes_per_task)
            
            # Perform gradient steps proportional to collected episodes
            if len(buffer) >= args.min_buffer:
                updates_to_do = int(len(ep_returns) * args.updates_per_episode)
                
                for _ in range(updates_to_do):
                    batch = buffer.sample(args.batch_size)
                    b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)

                    b_states_t = torch.tensor(np.array(b_states), dtype=torch.float32, device=device)
                    b_actions_t = torch.tensor(b_actions, dtype=torch.long, device=device)
                    b_rewards_t = torch.tensor(b_rewards, dtype=torch.float32, device=device)
                    b_next_t = torch.tensor(np.array(b_next_states), dtype=torch.float32, device=device)
                    b_dones_t = torch.tensor(b_dones, dtype=torch.float32, device=device)

                    q_pred = q_net(b_states_t).gather(1, b_actions_t.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        # Double DQN
                        next_q_online = q_net(b_next_t)
                        next_masks = np.stack([valid_actions_mask(s, N) for s in b_next_states])
                        next_masks_t = torch.tensor(next_masks, dtype=torch.float32, device=device)
                        next_q_online = next_q_online.masked_fill(next_masks_t == 0, -1e9)
                        best_next_actions = next_q_online.argmax(dim=1, keepdim=True)
                        
                        next_q_target = target_net(b_next_t)
                        next_q_max = next_q_target.gather(1, best_next_actions).squeeze(1)
                        target = b_rewards_t + (1.0 - b_dones_t) * next_q_max  # gamma = 1

                    loss = nn.functional.mse_loss(q_pred, target)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                    optimizer.step()
                    
                    gradient_steps += 1

                    if gradient_steps % args.target_update_freq == 0:
                        target_net.load_state_dict(q_net.state_dict())
            
            # Logging
            if episodes_collected - last_log_ep >= args.log_freq:
                window = args.log_freq
                avg_r = float(np.mean(reward_history[-window:]))
                avg_s = float(np.mean(success_history[-window:]))
                elapsed = time.time() - start_time
                print(f"[ep {episodes_collected:6d}/{args.num_episodes}] "
                      f"eps={epsilon:.3f}  avg_reward={avg_r:+.3f}  success_rate={avg_s:.3f}  "
                      f"buffer={len(buffer)}  grad_steps={gradient_steps}  "
                      f"time={elapsed:.1f}s")
                last_log_ep = episodes_collected

    # Save model
    torch.save({
        'model_state_dict': q_net.state_dict(),
        'N': N,
        'hidden_dim': args.hidden_dim,
    }, args.save_path)
    print(f"\nModel saved to {os.path.abspath(args.save_path)}")
    ray.shutdown()

if __name__ == '__main__':
    main()
