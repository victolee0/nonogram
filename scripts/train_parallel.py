"""
Ray 기반 병렬 학습 스크립트 — config 기반으로 다양한 모델/알고리즘 조합을 대규모 병렬 학습.

이 스크립트는 다수의 CPU 롤아웃 워커(RolloutWorker)를 비동기적으로 실행하여 에피소드를 수집하고,
메인 Learner 프로세스(GPU 우선)에서 최신 가중치 업데이트와 가중치 브로드캐스트를 수행합니다.
이를 통해 10x10 등 대형 보드의 2D RL 수렴 속도를 극적으로 끌어올립니다.

Usage:
    # Ray 병렬 기반 10x10 Board DQN 학습
    uv run python scripts/train_parallel.py --config configs/board_dqn_10x10.yaml --num_workers 4
"""

import os
import sys
import time
import random
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import ray

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nonogram.config import load_config, save_config, make_base_parser, resolve_device, get_experiment_dir
from nonogram.env.line_env import LineEnv
from nonogram.env.board_env import BoardEnv
from nonogram.agents.registry import create_agent
from nonogram.models.registry import create_model
from nonogram.env.state import hint_length


def pad_hints_batch(hints_list, k, N):
    """배치 안의 다양한 길이의 단서 리스트를 (B, N, k) 텐서 크기로 제로 패딩."""
    B = len(hints_list)
    padded_batch = np.zeros((B, N, k), dtype=np.int64)
    for b in range(B):
        if hints_list[b] is None:
            continue
        for i in range(N):
            actual = [int(x) for x in hints_list[b][i] if int(x) > 0]
            pad_len = k - len(actual)
            padded_batch[b, i, pad_len:] = actual
    return padded_batch


@ray.remote
class RolloutWorker:
    """Ray 기반 병렬 에피소드 수집 액터."""
    
    def __init__(self, config: dict, worker_id: int, seed: int):
        self.config = config
        self.worker_id = worker_id
        self.N = config["env"]["N"]
        self.env_type = config["env"].get("type", "line")
        
        # 난수 시드 설정
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 워커는 메모리 경합과 다중 인스턴스 효율을 위해 CPU를 강제 사용합니다.
        self.device = torch.device("cpu")
        
        # 로컬 환경 생성
        if self.env_type == "board":
            self.env = BoardEnv(
                N=self.N,
                early_stop=config["env"].get("early_stop", True),
                reward_shaping=config["env"].get("reward_shaping", False),
                reward_shaping_weight=config["env"].get("reward_shaping_weight", 0.1),
            )
        else:
            self.env = LineEnv(
                N=self.N,
                early_stop=config["env"].get("early_stop", True),
                reward_type=config["env"].get("reward_type", "step_correct"),
                reward_shaping=config["env"].get("reward_shaping", False),
                reward_shaping_weight=config["env"].get("reward_shaping_weight", 0.1),
            )
            
        # 로컬 에이전트 인스턴스 생성 (update 기능은 사용하지 않고 select_action만 수행)
        self.agent = create_agent(config, device=self.device)
        
    def set_weights(self, weights: dict, epsilon: float):
        """Learner로부터 최신 가중치와 ε 파라미터 동기화."""
        self.agent.q_net.load_state_dict(weights)
        if hasattr(self.agent, "epsilon"):
            self.agent.epsilon = epsilon
            
    def collect_episodes(self, num_episodes: int, reveal_ratio: float, mode: str, towards_ratio: float):
        """지정된 에피소드만큼 비동기적으로 플레이하여 트랜지션을 수집."""
        transitions = []
        episode_returns = []
        episode_successes = []
        steps_collected = 0
        
        use_chunking = getattr(self.agent, "action_chunking", False)
        lql_n = self.config["agent"].get("lql_n_step", 3)
        
        for _ in range(num_episodes):
            state = self.env.reset(reveal_ratio=reveal_ratio)
            
            # towards 모드 결정
            if mode == "towards":
                use_towards = True
            elif mode == "mixed":
                use_towards = random.random() < towards_ratio
            else:
                use_towards = False
                
            total_reward = 0.0
            done = False
            episode_buffer = []
            
            while not done:
                use_feasible_mask = self.config["env"].get("use_feasible_mask", False)
                if use_feasible_mask and hasattr(self.env, "get_feasible_action_mask"):
                    mask = self.env.get_feasible_action_mask()
                else:
                    mask = self.env.get_valid_mask()
                    
                # Action 선택
                if use_towards:
                    a = self.env.select_towards_action()
                    actions_chunk = [a] if a is not None else []
                else:
                    row_hints = getattr(self.env, "row_hints", None)
                    col_hints = getattr(self.env, "col_hints", None)
                    if use_chunking and not use_towards:
                        actions_chunk = self.agent.select_action_chunk(
                            state, mask, explore=True, row_hints=row_hints, col_hints=col_hints
                        )
                    else:
                        a = self.agent.select_action(
                            state, mask, explore=True, row_hints=row_hints, col_hints=col_hints
                        )
                        actions_chunk = [a]
                        
                if not actions_chunk:
                    break
                    
                if use_chunking and not use_towards:
                    # Action Chunking 실행
                    state_before_chunk = state.copy()
                    chunk_reward = 0.0
                    chunk_done = False
                    executed_actions = []
                    
                    for act in actions_chunk:
                        next_state, reward, done = self.env.step(act)
                        chunk_reward += reward
                        executed_actions.append(act)
                        state = next_state
                        if done:
                            chunk_done = True
                            break
                            
                    for act in executed_actions:
                        episode_buffer.append((state_before_chunk.copy(), act, chunk_reward, state.copy(), chunk_done))
                    total_reward += chunk_reward
                    steps_collected += len(executed_actions)
                else:
                    # 일반 1-step 실행
                    a = actions_chunk[0]
                    next_state, reward, done = self.env.step(a)
                    episode_buffer.append((state.copy(), a, reward, next_state.copy(), done))
                    state = next_state
                    total_reward += reward
                    steps_collected += 1
                    
            # LQL 및 힌트 연동 형식에 부합하도록 transitions 변환 가공
            row_hints = self.env.row_hints if hasattr(self.env, "row_hints") else None
            col_hints = self.env.col_hints if hasattr(self.env, "col_hints") else None
            
            for i in range(len(episode_buffer)):
                s, a, r, ns, d = episode_buffer[i]
                # n-step 뒤의 상태를 future_state로 묶음
                future_idx = min(i + lql_n, len(episode_buffer) - 1)
                future_s = episode_buffer[future_idx][3]
                transitions.append((
                    s.copy(), a, r, ns.copy(), float(d), future_s.copy(), row_hints, col_hints
                ))
                
            episode_returns.append(total_reward)
            if self.env_type == "board":
                last_reward = episode_buffer[-1][2] if episode_buffer else -1.0
                is_success = (last_reward == 1.0)
            else:
                is_success = (total_reward > 0)
            episode_successes.append(1.0 if is_success else 0.0)
            
        return transitions, episode_returns, episode_successes, steps_collected


def main():
    parser = make_base_parser()
    parser.add_argument('--num_workers', type=int, default=4, help='Ray 병렬 워커 개수')
    parser.add_argument('--episodes_per_task', type=int, default=10, help='워커가 한 번에 수집할 에피소드 수')
    parser.add_argument('--sync_freq', type=int, default=10, help='몇 번 태스크가 끝날 때마다 가중치를 동기화할지')
    parser.add_argument('--updates_per_episode', type=float, default=2.0, help='에피소드당 Learner 가중치 업데이트 횟수')
    
    known_args, remaining = parser.parse_known_args()
    config = load_config(known_args.config, remaining)
    
    # Ray 초기화
    ray.init(ignore_reinit_error=True)
    
    # Learner 디바이스 및 시드 할당
    device = resolve_device(config["device"])
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"Using device for Learner: {device}", flush=True)
    
    N = config["env"]["N"]
    env_type = config["env"].get("type", "line")
    training_cfg = config["training"]
    
    # Learner 에이전트 생성
    learner_agent = create_agent(config, device=device)
    
    # 실험 디렉토리 설정 및 구성 저장
    exp_dir = get_experiment_dir(config)
    save_config(config, exp_dir / "config.yaml")
    
    # Ray 워커들 비동기 인스턴스화
    print(f"Starting {known_args.num_workers} parallel Rollout Workers...", flush=True)
    workers = [
        RolloutWorker.remote(
            config=config,
            worker_id=i,
            seed=seed + i * 1000
        ) for i in range(known_args.num_workers)
    ]
    
    # 초기 가중치 동기화
    weights = {k: v.cpu() for k, v in learner_agent.q_net.state_dict().items()}
    weights_ref = ray.put(weights)
    ray.get([w.set_weights.remote(weights_ref, learner_agent.get_epsilon()) for w in workers])
    
    # 역방향 커리큘럼(Reverse Curriculum) 설정 및 상태 초기화
    curriculum_cfg = training_cfg.get("curriculum", {})
    curriculum_enabled = curriculum_cfg.get("enabled", False)
    curriculum_type = curriculum_cfg.get("type", "linear")
    
    start_ratio = 0.0
    end_ratio = 0.0
    if curriculum_enabled:
        start_ratio = curriculum_cfg.get("start_ratio", 0.8)
        if start_ratio == "auto":
            if env_type == "board":
                start_ratio = (N * N - 1) / (N * N)
            else:
                start_ratio = (N - 1) / N
        else:
            start_ratio = float(start_ratio)
        end_ratio = float(curriculum_cfg.get("end_ratio", 0.0))
        
    reveal_ratio = start_ratio
    
    # 비동기 Task 디스패치
    mode = training_cfg.get("mode", "rl")
    towards_ratio = training_cfg.get("towards_ratio", 0.0)
    
    pending_tasks = {
        w: w.collect_episodes.remote(
            known_args.episodes_per_task, reveal_ratio, mode, towards_ratio
        ) for w in workers
    }
    
    episodes_collected = 0
    total_steps_collected = 0
    gradient_steps = 0
    tasks_completed = 0
    reward_history = []
    success_history = []
    best_success_rate = 0.0
    last_log_ep = 0
    
    t_start = time.time()
    
    print(f"Training started (Parallel Ray Mode, {known_args.num_workers} workers)")
    print(f"Experiment Directory: {exp_dir}", flush=True)
    print(flush=True)
    
    # 메인 액터-러너 비동기 루프
    while episodes_collected < training_cfg["num_episodes"]:
        # 완료된 워커가 나올 때까지 대기
        ready_refs, _ = ray.wait(list(pending_tasks.values()), num_returns=1, timeout=None)
        
        for ref in ready_refs:
            # 완료된 워커 객체 찾기
            worker = next(w for w, r in pending_tasks.items() if r == ref)
            transitions, ep_returns, ep_successes, ep_steps = ray.get(ref)
            tasks_completed += 1
            
            # 중앙 리플레이 버퍼에 수집된 데이터 저장
            for t in transitions:
                s, a, r, ns, d, f_s, row_h, col_h = t
                learner_agent.store_transition(
                    s, a, r, ns, d, future_state=f_s, row_hints=row_h, col_hints=col_h
                )
                
            episodes_collected += len(ep_returns)
            total_steps_collected += ep_steps
            reward_history.extend(ep_returns)
            success_history.extend(ep_successes)
            
            # 커리큘럼 비율 어닐링 계산 (에피소드 개수 기준)
            if curriculum_enabled:
                if curriculum_type == "linear":
                    decay_eps = curriculum_cfg.get("decay_episodes", 20000)
                    if episodes_collected < decay_eps:
                        reveal_ratio = start_ratio - (episodes_collected / decay_eps) * (start_ratio - end_ratio)
                    else:
                        reveal_ratio = end_ratio
                elif curriculum_type == "cosine":
                    decay_eps = curriculum_cfg.get("decay_episodes", 20000)
                    if episodes_collected < decay_eps:
                        reveal_ratio = end_ratio + 0.5 * (start_ratio - end_ratio) * (
                            1.0 + math.cos(math.pi * episodes_collected / decay_eps)
                        )
                    else:
                        reveal_ratio = end_ratio
                elif curriculum_type == "adaptive":
                    adaptive_window = curriculum_cfg.get("adaptive_window", 100)
                    if len(success_history) >= adaptive_window:
                        recent_success = np.mean(success_history[-adaptive_window:])
                        adaptive_threshold = curriculum_cfg.get("adaptive_threshold", 0.8)
                        adaptive_backtrack = curriculum_cfg.get("adaptive_backtrack_threshold", 0.2)
                        adaptive_step = curriculum_cfg.get("adaptive_step", 0.02)
                        
                        if recent_success >= adaptive_threshold:
                            reveal_ratio = max(end_ratio, reveal_ratio - adaptive_step)
                        elif recent_success < adaptive_backtrack:
                            reveal_ratio = min(start_ratio, reveal_ratio + adaptive_step)
            
            # 주기적 가중치 동기화 브로드캐스트
            if tasks_completed % known_args.sync_freq == 0:
                weights = {k: v.cpu() for k, v in learner_agent.q_net.state_dict().items()}
                weights_ref = ray.put(weights)
                # 에이전트의 실시간 ε도 동반 브로드캐스트
                for w in workers:
                    w.set_weights.remote(weights_ref, learner_agent.get_epsilon())
                    
            # 해당 워커에 새 수집 Task 디스패치
            pending_tasks[worker] = worker.collect_episodes.remote(
                known_args.episodes_per_task, reveal_ratio, mode, towards_ratio
            )
            
            # 모인 데이터 크기에 비례하여 Learner 신경망 업데이트
            if learner_agent.can_train():
                updates_to_do = int(len(ep_returns) * known_args.updates_per_episode)
                for _ in range(updates_to_do):
                    learner_agent.update()
                    gradient_steps += 1
                    
                    if learner_agent.should_sync_target():
                        learner_agent.sync_target()
                        
            # 로깅 및 체크포인트 보존
            if episodes_collected - last_log_ep >= training_cfg["log_freq"]:
                window = training_cfg["log_freq"]
                avg_r = float(np.mean(reward_history[-window:]))
                avg_s = float(np.mean(success_history[-window:]))
                elapsed = time.time() - t_start
                eps = learner_agent.get_epsilon()
                
                if curriculum_enabled:
                    print(f"[ep {episodes_collected:6d}/{training_cfg['num_episodes']}] "
                          f"avg_reward={avg_r:+.3f}  success_rate={avg_s:.3f}  "
                          f"reveal={reveal_ratio:.3f}  "
                          f"ε={eps:.3f}  buffer={len(learner_agent.buffer)}  "
                          f"grad_steps={gradient_steps}  ({elapsed:.0f}s)", flush=True)
                else:
                    print(f"[ep {episodes_collected:6d}/{training_cfg['num_episodes']}] "
                          f"avg_reward={avg_r:+.3f}  success_rate={avg_s:.3f}  "
                          f"ε={eps:.3f}  buffer={len(learner_agent.buffer)}  "
                          f"grad_steps={gradient_steps}  ({elapsed:.0f}s)", flush=True)
                          
                last_log_ep = episodes_collected
                
                # 최고 성과 모델 저장
                if avg_s > best_success_rate:
                    best_success_rate = avg_s
                    best_path = exp_dir / "checkpoints" / "best.pt"
                    learner_agent.save(best_path)
                    
    # 최종 모델 가중치 저장
    save_path = exp_dir / "checkpoints" / "latest.pt"
    learner_agent.save(save_path)
    print(f"\nTraining completed. Model saved to {save_path}")
    print(f"Best success rate achieved: {best_success_rate:.3f}")
    
    # Ray 리소스 해제
    ray.shutdown()


if __name__ == "__main__":
    main()
