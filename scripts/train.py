"""
학습 스크립트 — config 기반으로 다양한 모델/알고리즘 조합을 학습.

데이터 수집 모드:
  - "rl":        순수 RL. 에이전트의 ε-greedy policy로만 탐색.
                 reward_type="step_correct"와 조합하면 towards 없이 학습 가능.
  - "towards":   towards 모드. ground-truth를 보여주는 expert demonstration.
                 (towards_ratio로 비율 조절)
  - "mixed":     RL + towards 혼합. (towards_ratio로 비율 조절)

Usage:
    # 순수 RL 학습 (towards 없음)
    uv run python scripts/train.py --config configs/dqn_10x10.yaml \\
        --training.mode rl --env.reward_type step_correct

    # 기존 방식 (towards 혼합)
    uv run python scripts/train.py --config configs/dqn_10x10.yaml \\
        --training.mode mixed --training.towards_ratio 0.5
"""

import os
import random
import sys
import time
import math

import numpy as np
import torch

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nonogram.config import load_config, save_config, make_base_parser, resolve_device, get_experiment_dir
from nonogram.env.line_env import LineEnv
from nonogram.env.board_env import BoardEnv
from nonogram.agents.registry import create_agent


def train_dqn_family(config: dict, device: torch.device):
    """DQN 계열 (DQN, Double DQN) 학습 루프."""
    N = config["env"]["N"]
    training_cfg = config["training"]
    env_type = config["env"].get("type", "line")

    # 환경 생성
    if env_type == "board":
        env = BoardEnv(
            N=N,
            early_stop=config["env"].get("early_stop", True),
            reward_shaping=config["env"].get("reward_shaping", False),
            reward_shaping_weight=config["env"].get("reward_shaping_weight", 0.1),
        )
    else:
        env = LineEnv(
            N=N,
            early_stop=config["env"].get("early_stop", True),
            reward_type=config["env"].get("reward_type", "step_correct"),
            reward_shaping=config["env"].get("reward_shaping", False),
            reward_shaping_weight=config["env"].get("reward_shaping_weight", 0.1),
        )

    # 에이전트 생성
    agent = create_agent(config, device=device)

    # 실험 디렉터리
    exp_dir = get_experiment_dir(config)
    save_config(config, exp_dir / "config.yaml")

    mode = training_cfg.get("mode", "rl")
    towards_ratio = training_cfg.get("towards_ratio", 0.0)

    reward_history = []
    success_history = []
    best_success_rate = 0.0

    print(f"Training {config['agent']['type']} + {config['model']['type']} "
          f"(N={N}) on {device}", flush=True)
    if env_type == "board":
        print(f"Mode: {mode}  |  Env: board  |  early_stop: {env.early_stop}", flush=True)
    else:
        print(f"Mode: {mode}  |  Env: line  |  reward_type: {env.reward_type}  |  "
              f"early_stop: {env.early_stop}", flush=True)
    if mode in ("towards", "mixed"):
        print(f"towards_ratio: {towards_ratio}", flush=True)
    print(f"Experiment: {exp_dir}", flush=True)
    print(flush=True)

    # 역방향 커리큘럼(Reverse Curriculum) 설정 및 상태 초기화
    curriculum_cfg = training_cfg.get("curriculum", {})
    curriculum_enabled = curriculum_cfg.get("enabled", False)
    curriculum_type = curriculum_cfg.get("type", "linear")  # "linear" | "cosine" | "adaptive"
    
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

    t_start = time.time()
    for episode in range(training_cfg["num_episodes"]):
        # 역방향 커리큘럼(Reverse Curriculum) 감쇠 계산
        if curriculum_enabled:
            if curriculum_type == "linear":
                decay_eps = curriculum_cfg.get("decay_episodes", 20000)
                if episode < decay_eps:
                    reveal_ratio = start_ratio - (episode / decay_eps) * (start_ratio - end_ratio)
                else:
                    reveal_ratio = end_ratio
            elif curriculum_type == "cosine":
                decay_eps = curriculum_cfg.get("decay_episodes", 20000)
                if episode < decay_eps:
                    reveal_ratio = end_ratio + 0.5 * (start_ratio - end_ratio) * (1.0 + math.cos(math.pi * episode / decay_eps))
                else:
                    reveal_ratio = end_ratio
            elif curriculum_type == "adaptive":
                # 에피소드 말엽에 업데이트된 reveal_ratio를 그대로 유지함
                pass
        else:
            reveal_ratio = 0.0

        state = env.reset(reveal_ratio=reveal_ratio)

        # 모드 결정
        if mode == "towards":
            use_towards = True
        elif mode == "mixed":
            use_towards = random.random() < towards_ratio
        else:  # "rl" (순수 RL)
            use_towards = False

        total_reward = 0.0
        done = False
        episode_buffer = []
        
        # Action Chunking 사용 여부 판단 (Towards 모드가 아닐 때만 사용)
        use_chunking = getattr(agent, "action_chunking", False) and not use_towards

        while not done:
            mask = env.get_valid_mask()

            # Action 선택
            if use_towards:
                a = env.select_towards_action()
                actions_chunk = [a] if a is not None else []
            else:
                if use_chunking:
                    actions_chunk = agent.select_action_chunk(state, mask, explore=True)
                else:
                    a = agent.select_action(state, mask, explore=True)
                    actions_chunk = [a]

            if not actions_chunk:
                break

            if use_chunking:
                # Action Chunking 실행:
                # 척 안의 모든 액션을 순차적으로 실행 (중간 완료 감시)
                state_before_chunk = state.copy()
                chunk_reward = 0.0
                chunk_done = False
                executed_actions = []
                for act in actions_chunk:
                    next_state, reward, done = env.step(act)
                    chunk_reward += reward
                    executed_actions.append(act)
                    state = next_state
                    if done:
                        chunk_done = True
                        break
                
                # 일반 DQN 형식에 맞추되, 각 액션별 next_state와 reward는 척 시점을 기준으로 묶어 n-step 리턴 수행
                for act in executed_actions:
                    episode_buffer.append((state_before_chunk.copy(), act, chunk_reward, state.copy(), chunk_done))
                
                total_reward += chunk_reward
                agent.step_count += len(executed_actions)
            else:
                # 일반 1-step 실행
                a = actions_chunk[0]
                next_state, reward, done = env.step(a)
                episode_buffer.append((state.copy(), a, reward, next_state.copy(), done))
                state = next_state
                total_reward += reward
                agent.step_count += 1

            # 학습 step (버퍼가 충분할 때만)
            if agent.should_train():
                agent.update()

            # Target network 동기화
            if agent.should_sync_target():
                agent.sync_target()

        # 에피소드 종료 후 n-step 미래 상태를 매칭하여 에이전트 버퍼로 전달
        lql_n = config["agent"].get("lql_n_step", 3)
        row_hints = env.row_hints if hasattr(env, "row_hints") else None
        col_hints = env.col_hints if hasattr(env, "col_hints") else None
        for i in range(len(episode_buffer)):
            s, a, r, ns, d = episode_buffer[i]
            # n-step 뒤의 상태 찾기 (없으면 에피소드 마지막 상태)
            future_idx = min(i + lql_n, len(episode_buffer) - 1)
            future_s = episode_buffer[future_idx][3] # next_state of future step
            agent.store_transition(s, a, r, ns, d, future_state=future_s, row_hints=row_hints, col_hints=col_hints)

        reward_history.append(total_reward)
        # success 판정
        if env_type == "board":
            last_reward = episode_buffer[-1][2] if episode_buffer else -1.0
            is_success = (last_reward == 1.0)
        else:
            is_success = (total_reward > 0)
        success_history.append(1.0 if is_success else 0.0)

        # adaptive 커리큘럼 업데이트
        if curriculum_enabled and curriculum_type == "adaptive":
            adaptive_window = curriculum_cfg.get("adaptive_window", 100)
            if len(success_history) >= adaptive_window:
                recent_success = np.mean(success_history[-adaptive_window:])
                adaptive_threshold = curriculum_cfg.get("adaptive_threshold", 0.8)
                adaptive_backtrack = curriculum_cfg.get("adaptive_backtrack_threshold", 0.2)
                adaptive_step = curriculum_cfg.get("adaptive_step", 0.02)
                
                if recent_success >= adaptive_threshold:
                    # 에이전트가 잘하고 있으므로 난이도 상향 (reveal_ratio 감소)
                    reveal_ratio = max(end_ratio, reveal_ratio - adaptive_step)
                elif recent_success < adaptive_backtrack:
                    # 에이전트가 헤매고 있으므로 난이도 하향 (reveal_ratio 증가로 도움을 줌)
                    reveal_ratio = min(start_ratio, reveal_ratio + adaptive_step)

        if (episode + 1) % training_cfg["log_freq"] == 0:
            window = training_cfg["log_freq"]
            avg_r = float(np.mean(reward_history[-window:]))
            avg_s = float(np.mean(success_history[-window:]))
            elapsed = time.time() - t_start
            eps = agent.get_epsilon()
            if curriculum_cfg.get("enabled", False):
                print(f"[ep {episode+1:6d}/{training_cfg['num_episodes']}] "
                      f"avg_reward={avg_r:+.3f}  success_rate={avg_s:.3f}  "
                      f"reveal={reveal_ratio:.3f}  "
                      f"ε={eps:.3f}  buffer={len(agent.buffer)}  "
                      f"steps={agent.step_count}  ({elapsed:.0f}s)", flush=True)
            else:
                print(f"[ep {episode+1:6d}/{training_cfg['num_episodes']}] "
                      f"avg_reward={avg_r:+.3f}  success_rate={avg_s:.3f}  "
                      f"ε={eps:.3f}  buffer={len(agent.buffer)}  "
                      f"steps={agent.step_count}  ({elapsed:.0f}s)", flush=True)

            # Best model 저장
            if avg_s > best_success_rate:
                best_success_rate = avg_s
                best_path = exp_dir / "checkpoints" / "best.pt"
                agent.save(best_path)

    # 최종 모델 저장
    save_path = exp_dir / "checkpoints" / "latest.pt"
    agent.save(save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Best success rate: {best_success_rate:.3f}")


def train_ppo(config: dict, device: torch.device):
    """PPO 학습 루프."""
    N = config["env"]["N"]
    training_cfg = config["training"]
    env_type = config["env"].get("type", "line")

    if env_type == "board":
        env = BoardEnv(
            N=N,
            early_stop=config["env"].get("early_stop", True),
            reward_shaping=config["env"].get("reward_shaping", False),
            reward_shaping_weight=config["env"].get("reward_shaping_weight", 0.1),
        )
    else:
        env = LineEnv(
            N=N,
            early_stop=config["env"].get("early_stop", True),
            reward_type=config["env"].get("reward_type", "step_correct"),
            reward_shaping=config["env"].get("reward_shaping", False),
            reward_shaping_weight=config["env"].get("reward_shaping_weight", 0.1),
        )

    agent = create_agent(config, device=device)

    exp_dir = get_experiment_dir(config)
    save_config(config, exp_dir / "config.yaml")

    reward_history = []
    success_history = []
    episodes_since_update = 0
    best_success_rate = 0.0

    print(f"Training PPO (N={N}) on {device}")
    if env_type == "board":
        print(f"Env: board  |  early_stop: {env.early_stop}")
    else:
        print(f"Env: line  |  reward_type: {env.reward_type}  |  early_stop: {env.early_stop}")
    print(f"Experiment: {exp_dir}")
    print()

    # 역방향 커리큘럼(Reverse Curriculum) 설정 및 상태 초기화
    curriculum_cfg = training_cfg.get("curriculum", {})
    curriculum_enabled = curriculum_cfg.get("enabled", False) and env_type == "board"
    curriculum_type = curriculum_cfg.get("type", "linear")  # "linear" | "cosine" | "adaptive"
    
    start_ratio = 0.0
    end_ratio = 0.0
    if curriculum_enabled:
        start_ratio = curriculum_cfg.get("start_ratio", 0.8)
        if start_ratio == "auto":
            start_ratio = (N * N - 1) / (N * N)
        else:
            start_ratio = float(start_ratio)
        end_ratio = float(curriculum_cfg.get("end_ratio", 0.0))
        
    reveal_ratio = start_ratio

    t_start = time.time()

    for episode in range(training_cfg["num_episodes"]):
        # 역방향 커리큘럼(Reverse Curriculum) 감쇠 계산
        if curriculum_enabled:
            if curriculum_type == "linear":
                decay_eps = curriculum_cfg.get("decay_episodes", 20000)
                if episode < decay_eps:
                    reveal_ratio = start_ratio - (episode / decay_eps) * (start_ratio - end_ratio)
                else:
                    reveal_ratio = end_ratio
            elif curriculum_type == "cosine":
                decay_eps = curriculum_cfg.get("decay_episodes", 20000)
                if episode < decay_eps:
                    reveal_ratio = end_ratio + 0.5 * (start_ratio - end_ratio) * (1.0 + math.cos(math.pi * episode / decay_eps))
                else:
                    reveal_ratio = end_ratio
            elif curriculum_type == "adaptive":
                # 에피소드 말엽에 업데이트된 reveal_ratio를 그대로 유지함
                pass
        else:
            reveal_ratio = 0.0

        if env_type == "board":
            state = env.reset(reveal_ratio=reveal_ratio)
        else:
            state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            mask = env.get_valid_mask()
            a = agent.select_action(state, mask, explore=True)

            if a is None:
                break

            next_state, reward, done = env.step(a)
            agent.store_transition(state, a, reward, next_state, done, mask)
            state = next_state
            total_reward += reward
            agent.step_count += 1

        reward_history.append(total_reward)
        # success 판정
        if env_type == "board":
            is_success = (reward == 1.0)
        else:
            is_success = (total_reward > 0)
        success_history.append(1.0 if is_success else 0.0)
        episodes_since_update += 1

        # adaptive 커리큘럼 업데이트
        if curriculum_enabled and curriculum_type == "adaptive":
            adaptive_window = curriculum_cfg.get("adaptive_window", 100)
            if len(success_history) >= adaptive_window:
                recent_success = np.mean(success_history[-adaptive_window:])
                adaptive_threshold = curriculum_cfg.get("adaptive_threshold", 0.8)
                adaptive_backtrack = curriculum_cfg.get("adaptive_backtrack_threshold", 0.2)
                adaptive_step = curriculum_cfg.get("adaptive_step", 0.02)
                
                if recent_success >= adaptive_threshold:
                    # 에이전트가 잘하고 있으므로 난이도 상향 (reveal_ratio 감소)
                    reveal_ratio = max(end_ratio, reveal_ratio - adaptive_step)
                elif recent_success < adaptive_backtrack:
                    # 에이전트가 헤매고 있으므로 난이도 하향 (reveal_ratio 증가로 도움을 줌)
                    reveal_ratio = min(start_ratio, reveal_ratio + adaptive_step)

        # PPO 업데이트
        if episodes_since_update >= 8:
            agent.update()
            episodes_since_update = 0

        if (episode + 1) % training_cfg["log_freq"] == 0:
            window = training_cfg["log_freq"]
            avg_r = float(np.mean(reward_history[-window:]))
            avg_s = float(np.mean(success_history[-window:]))
            elapsed = time.time() - t_start
            if curriculum_cfg.get("enabled", False) and env_type == "board":
                print(f"[ep {episode+1:6d}/{training_cfg['num_episodes']}] "
                      f"avg_reward={avg_r:+.3f}  success_rate={avg_s:.3f}  "
                      f"reveal={reveal_ratio:.3f}  "
                      f"steps={agent.step_count}  ({elapsed:.0f}s)")
            else:
                print(f"[ep {episode+1:6d}/{training_cfg['num_episodes']}] "
                      f"avg_reward={avg_r:+.3f}  success_rate={avg_s:.3f}  "
                      f"steps={agent.step_count}  ({elapsed:.0f}s)")

            if avg_s > best_success_rate:
                best_success_rate = avg_s
                best_path = exp_dir / "checkpoints" / "best.pt"
                agent.save(best_path)

    save_path = exp_dir / "checkpoints" / "latest.pt"
    agent.save(save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Best success rate: {best_success_rate:.3f}")


def train_gflownet(config: dict, device: torch.device):
    """GFlowNet Trajectory Balance 학습 루프."""
    N = config["env"]["N"]
    training_cfg = config["training"]

    # GFlowNet용 환경 생성
    from nonogram.env.gflownet_env import GFlowNetNonogramEnv
    env = GFlowNetNonogramEnv(
        N=N,
        reward_temp=config["agent"].get("reward_temp", 1.0),
        epsilon=config["agent"].get("epsilon", 1e-4),
        early_stop=config["env"].get("early_stop", True),
        pb_type=config["agent"].get("pb_type", "uniform"),
        pb_epsilon=config["agent"].get("pb_epsilon", 0.1),
    )

    agent = create_agent(config, device=device)

    exp_dir = get_experiment_dir(config)
    save_config(config, exp_dir / "config.yaml")

    # Tensorboard logger 설정
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(exp_dir / "tb"))

    reward_history = []
    success_history = []
    loss_history = []
    log_Z_history = []
    best_success_rate = 0.0

    print(f"Training GFlowNet (N={N}) on {device}", flush=True)
    print(f"Env: gflownet  |  early_stop: {env.early_stop}  |  pb_type: {env.pb_type}", flush=True)
    print(f"Experiment: {exp_dir}", flush=True)
    print(flush=True)

    t_start = time.time()

    temp_start = 1.0
    temp_end = config["agent"].get("reward_temp", 5.0)

    for episode in range(training_cfg["num_episodes"]):
        # 온도 선형 증가 스케줄링
        progress = min(1.0, episode / (training_cfg["num_episodes"] * 0.8)) # 80% 시점에서 최대 온도 도달
        current_temp = temp_start + progress * (temp_end - temp_start)
        env.reward_temp = current_temp  # 환경에 현재 온도 업데이트

        state = env.reset()
        done = False

        while not done:
            mask = env.get_valid_mask()
            a = agent.select_action(state, mask, explore=True)

            if a is None:
                break

            next_state, reward, done = env.step(a)

            # transition 저장
            agent.store_transition(
                state=state,
                action=a,
                reward=reward,
                next_state=next_state,
                done=done,
                mask=mask
            )
            state = next_state

        # GFlowNet 에이전트 가중치 업데이트
        metrics = {}
        if len(agent.buffer) >= agent.batch_size:
            metrics = agent.update()

        # Terminal Reward 기준 성공 여부 체크
        total_reward = reward
        reward_history.append(total_reward)
        
        # 완벽 정답 보상 임계치 계산 (부동소수점 오차 고려)
        perfect_reward_threshold = math.exp(env.reward_temp * 2.0) * 0.99
        is_success = (total_reward >= perfect_reward_threshold)
        success_history.append(1.0 if is_success else 0.0)

        # 메트릭 기록
        if metrics:
            loss_history.append(metrics["loss"])
            log_Z_history.append(metrics["log_Z"])
            writer.add_scalar("Train/Loss", metrics["loss"], episode)
            writer.add_scalar("Train/Log_Z", metrics["log_Z"], episode)

        writer.add_scalar("Train/Reward", total_reward, episode)
        writer.add_scalar("Train/SuccessRate", float(is_success), episode)

        if (episode + 1) % training_cfg["log_freq"] == 0:
            window = training_cfg["log_freq"]
            avg_r = float(np.mean(reward_history[-window:]))
            avg_s = float(np.mean(success_history[-window:]))
            avg_loss = float(np.mean(loss_history[-window:])) if loss_history else 0.0
            log_z_val = agent.log_Z.item()
            elapsed = time.time() - t_start

            print(f"[ep {episode+1:6d}/{training_cfg['num_episodes']}] "
                  f"avg_reward={avg_r:.6f}  success_rate={avg_s:.3f}  "
                  f"loss={avg_loss:.4f}  log_Z={log_z_val:.3f}  "
                  f"buffer={len(agent.buffer)}  ({elapsed:.0f}s)", flush=True)

            if avg_s > best_success_rate:
                best_success_rate = avg_s
                best_path = exp_dir / "checkpoints" / "best.pt"
                agent.save(best_path)

    save_path = exp_dir / "checkpoints" / "latest.pt"
    agent.save(save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Best success rate: {best_success_rate:.3f}")
    writer.close()


def main():
    parser = make_base_parser()
    known_args, remaining = parser.parse_known_args()

    config = load_config(known_args.config, remaining)
    device = resolve_device(config["device"])

    # Seed
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Using device: {device}", flush=True)

    agent_type = config["agent"]["type"]
    if agent_type in ("dqn", "double_dqn", "crl"):
        train_dqn_family(config, device)
    elif agent_type == "ppo":
        train_ppo(config, device)
    elif agent_type == "gflownet":
        train_gflownet(config, device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


if __name__ == "__main__":
    main()
