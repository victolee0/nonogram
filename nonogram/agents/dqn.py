"""
DQN Agent — 기존 train.py의 학습 로직을 에이전트 클래스로 캡슐화.

표준 DQN: target = r + (1-done) * γ * max_a' Q_target(s', a')
"""

import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from nonogram.agents.base import BaseAgent
from nonogram.agents.registry import register_agent
from nonogram.models.registry import create_model
from nonogram.env.state import valid_actions_mask, hint_length, is_terminal


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



class PrioritizedReplayBuffer:
    """우선순위 경험 재생(PER) 버퍼.
    
    SumTree 대신 Numpy 벡터 연산을 활용해 견고하고 안정적이게 구현되었습니다.
    """
    def __init__(self, max_size: int, alpha: float = 0.6):
        self.max_size = max_size
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.pos = 0

    def push(self, transition, priority):
        """새 전이 상태 저장 및 초기 우선순위 부여."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = np.clip(priority, 1e-5, 100.0)
        self.pos = (self.pos + 1) % self.max_size

    def sample(self, batch_size: int, beta: float = 0.4):
        """우선순위 가중 비례 샘플링 및 중요도 샘플링 가중치(IS weights) 계산."""
        total_len = len(self.buffer)
        if total_len == 0:
            return [], [], []

        # P_i = p_i^alpha / sum_k p_k^alpha
        prios = self.priorities[:total_len]
        prios = np.clip(prios, 1e-5, 100.0)
        probs = prios ** self.alpha
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs /= prob_sum
        else:
            probs = np.ones_like(probs) / total_len

        indices = np.random.choice(total_len, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # w_i = (N * P_i)^(-beta) / max_k w_k
        weights = (total_len * probs[indices]) ** (-beta)
        weights /= weights.max() if weights.max() > 0 else 1e-5
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """TD-error를 바탕으로 우선순위 업데이트."""
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = np.clip(prio, 1e-5, 100.0)

    def __len__(self):
        return len(self.buffer)


@register_agent("dqn")
class DQNAgent(BaseAgent):
    """표준 DQN 에이전트."""

    def __init__(self, config: dict, device: torch.device | None = None,
                 model: nn.Module | None = None):
        super().__init__(config, device)

        # 모델 생성
        if model is not None:
            self.q_net = model.to(self.device)
        else:
            self.q_net = create_model(config).to(self.device)
        self.target_net = create_model(config).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # 옵티마이저
        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=self.agent_cfg["lr"],
        )

        # PER 및 Replay buffer 설정
        self.use_per = self.agent_cfg.get("use_per", False)
        self.per_alpha = self.agent_cfg.get("per_alpha", 0.6)
        self.per_beta = self.agent_cfg.get("per_beta", 0.4)
        self.per_beta_increment = self.agent_cfg.get("per_beta_increment", 0.001)

        # Action Chunking 설정
        self.action_chunking = self.agent_cfg.get("action_chunking", False)
        self.chunk_size = self.agent_cfg.get("chunk_size", 1)

        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(
                max_size=self.agent_cfg["buffer_size"],
                alpha=self.per_alpha
            )
        else:
            self.buffer = deque(maxlen=self.agent_cfg["buffer_size"])

    def select_action(self, state: np.ndarray, mask: np.ndarray,
                      explore: bool = True, row_hints: list | None = None, col_hints: list | None = None) -> int:
        """ε-greedy action 선택."""
        if explore and random.random() < self.get_epsilon():
            valid = np.where(mask > 0)[0]
            if len(valid) == 0:
                return 0
            return int(random.choice(valid))

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            model_type = self.config["model"]["type"]
            is_2d_board_model = model_type in ("board_cnn", "board_transformer")
            
            if is_2d_board_model and row_hints is not None and col_hints is not None:
                N = self.config["env"]["N"]
                k = hint_length(N)
                padded_rh = torch.tensor(pad_hints_batch([row_hints], k, N), dtype=torch.long, device=self.device)
                padded_ch = torch.tensor(pad_hints_batch([col_hints], k, N), dtype=torch.long, device=self.device)
                q_values = self.q_net(state_t, padded_rh, padded_ch).squeeze(0).cpu().numpy()
            else:
                q_values = self.q_net(state_t).squeeze(0).cpu().numpy()

        # 무효 action 마스킹
        q_values = np.where(mask > 0, q_values, -np.inf)
        return int(np.argmax(q_values))

    def select_action_chunk(self, state: np.ndarray, mask: np.ndarray,
                            explore: bool = True, row_hints: list | None = None, col_hints: list | None = None) -> list[int]:
        """ε-greedy 기반으로 한번에 여러 개의 (non-conflicting) action chunk 선택."""
        N = self.config["env"]["N"]
        N2 = N * N

        if explore and random.random() < self.get_epsilon():
            valid = np.where(mask > 0)[0]
            if len(valid) == 0:
                return []
            # 무작위로 선택하되, 동일한 셀에 대해 Paint와 X가 중복되지 않도록 방지
            np.random.shuffle(valid)
            chosen_cells = set()
            chunk = []
            for act in valid:
                cell = act % N2
                if cell not in chosen_cells:
                    chosen_cells.add(cell)
                    chunk.append(int(act))
                    if len(chunk) >= self.chunk_size:
                        break
            return chunk

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            model_type = self.config["model"]["type"]
            is_2d_board_model = model_type in ("board_cnn", "board_transformer")
            
            if is_2d_board_model and row_hints is not None and col_hints is not None:
                k = hint_length(N)
                padded_rh = torch.tensor(pad_hints_batch([row_hints], k, N), dtype=torch.long, device=self.device)
                padded_ch = torch.tensor(pad_hints_batch([col_hints], k, N), dtype=torch.long, device=self.device)
                q_values = self.q_net(state_t, padded_rh, padded_ch).squeeze(0).cpu().numpy()
            else:
                q_values = self.q_net(state_t).squeeze(0).cpu().numpy()

        # 무효 action 마스킹
        q_values = np.where(mask > 0, q_values, -np.inf)

        # 각 셀에 대해 Paint(act < N2)와 X(act >= N2) 중 큰 Q-value를 가진 행동만 선별
        cell_best_action = []
        for cell in range(N2):
            paint_act = cell
            x_act = N2 + cell
            paint_q = q_values[paint_act]
            x_q = q_values[x_act]
            
            if paint_q == -np.inf and x_q == -np.inf:
                continue
            
            if paint_q >= x_q:
                cell_best_action.append((paint_q, paint_act))
            else:
                cell_best_action.append((x_q, x_act))

        # Q-value 기준으로 내림차순 정렬 후 상위 chunk_size 개를 리턴
        cell_best_action.sort(key=lambda x: x[0], reverse=True)
        chunk = [int(act) for q, act in cell_best_action[:self.chunk_size] if q != -np.inf]
        return chunk

    def store_transition(self, state, action, reward, next_state, done, future_state=None, row_hints=None, col_hints=None):
        """Replay buffer에 transition 저장."""
        # future_state가 없으면 (에피소드 끝자락) next_state를 사용
        f_state = future_state.copy() if future_state is not None else next_state.copy()
        transition = (
            state.copy(), action, reward, next_state.copy(), float(done), f_state, row_hints, col_hints
        )
        if self.use_per:
            # 초기 우선순위는 현재 버퍼의 최고 우선순위로 주어 한 번도 학습하지 않은 데이터 우선순위 부여
            max_prio = self.buffer.priorities[:len(self.buffer)].max() if len(self.buffer) > 0 else 1.0
            # +1 reward(성공)를 받은 전이인 경우, 우선순위를 즉시 크게 높여서 조기 복습 유도
            if reward == 1.0:
                max_prio = max(max_prio * 2.0, 5.0)
            max_prio = min(max_prio, 100.0)
            self.buffer.push(transition, max_prio)
        else:
            self.buffer.append(transition)

    def can_train(self) -> bool:
        """학습 가능 여부."""
        return len(self.buffer) >= self.agent_cfg["min_buffer"]

    def should_train(self) -> bool:
        """이번 step에 학습해야 하는지."""
        return (self.can_train() and
                self.step_count % self.agent_cfg["train_freq"] == 0)

    def should_sync_target(self) -> bool:
        """Target network 동기화 필요 여부."""
        return self.step_count % self.agent_cfg["target_update_freq"] == 0

    def update(self, batch: dict[str, Any] | None = None) -> dict[str, float]:
        """Replay buffer에서 샘플링하여 한 step 학습.

        batch가 None이면 buffer에서 자동 샘플링.
        """
        indices = None
        is_weights = None

        if batch is None:
            if not self.can_train():
                return {}
            
            if self.use_per:
                self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)
                samples, indices, is_weights = self.buffer.sample(
                    batch_size=self.agent_cfg["batch_size"],
                    beta=self.per_beta
                )
                states, actions, rewards, next_states, dones, future_states, row_hints, col_hints = zip(*samples)
                batch = {
                    "states": list(states),
                    "actions": list(actions),
                    "rewards": list(rewards),
                    "next_states": list(next_states),
                    "dones": list(dones),
                    "future_states": list(future_states),
                    "row_hints": list(row_hints),
                    "col_hints": list(col_hints),
                }
            else:
                batch = self._sample_batch()

        N = self.config["env"]["N"]
        gamma = self.agent_cfg["gamma"]
        grad_clip = self.agent_cfg["grad_clip"]

        b_states = torch.tensor(np.array(batch["states"]),
                                dtype=torch.float32, device=self.device)
        b_actions = torch.tensor(batch["actions"],
                                 dtype=torch.long, device=self.device)
        b_rewards = torch.tensor(batch["rewards"],
                                 dtype=torch.float32, device=self.device)
        b_next_states = torch.tensor(np.array(batch["next_states"]),
                                     dtype=torch.float32, device=self.device)
        b_dones = torch.tensor(batch["dones"],
                               dtype=torch.float32, device=self.device)
        b_futures = torch.tensor(np.array(batch["future_states"]),
                                 dtype=torch.float32, device=self.device)

        model_type = self.config["model"]["type"]
        is_2d_board_model = model_type in ("board_cnn", "board_transformer")

        # 2D 보드 모델의 경우 힌트 정보 패딩 텐서 생성
        if is_2d_board_model:
            k = hint_length(N)
            padded_rh = torch.tensor(pad_hints_batch(batch["row_hints"], k, N), dtype=torch.long, device=self.device)
            padded_ch = torch.tensor(pad_hints_batch(batch["col_hints"], k, N), dtype=torch.long, device=self.device)
            
            # Current Q
            q_pred = self.q_net(b_states, padded_rh, padded_ch).gather(
                1, b_actions.unsqueeze(1)
            ).squeeze(1)
            
            # Target Q
            with torch.no_grad():
                next_q = self._compute_target_q(b_next_states, batch["next_states"], N, padded_rh, padded_ch)
                target = b_rewards + (1.0 - b_dones) * gamma * next_q
        else:
            # Current Q
            q_pred = self.q_net(b_states).gather(
                1, b_actions.unsqueeze(1)
            ).squeeze(1)
            
            # Target Q
            with torch.no_grad():
                next_q = self._compute_target_q(b_next_states, batch["next_states"], N)
                target = b_rewards + (1.0 - b_dones) * gamma * next_q

        # TD-Error 업데이트
        td_errors = torch.abs(q_pred - target).detach().cpu().numpy()
        if self.use_per and indices is not None:
            self.buffer.update_priorities(indices, td_errors)

        # LQL Loss (Q(s_t, a_t) >= max_a' Q(s_{t+n}, a'))
        lql_weight = self.agent_cfg.get("lql_weight", 0.0)
        lql_loss = torch.tensor(0.0, device=self.device)
        
        if lql_weight > 0:
            with torch.no_grad():
                if is_2d_board_model:
                    next_v_lql = self._compute_target_q(b_futures, batch["future_states"], N, padded_rh, padded_ch)
                else:
                    next_v_lql = self._compute_target_q(b_futures, batch["future_states"], N)
            
            # future_state가 terminal인지 판정하여 마스킹 (terminal 상태의 가치는 감수하므로 상한 없음)
            future_dones = torch.tensor(
                [float(is_terminal(s, N)) for s in batch["future_states"]],
                dtype=torch.float32, device=self.device
            )
            lql_loss = torch.mean((1.0 - future_dones) * torch.nn.functional.relu(next_v_lql - q_pred)**2)

        # Loss 계산 및 IS 가중치 적용
        if self.use_per and is_weights is not None:
            is_weights_t = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
            td_loss = torch.mean(is_weights_t * (q_pred - target) ** 2)
        else:
            td_loss = nn.functional.mse_loss(q_pred, target)

        loss = td_loss + lql_weight * lql_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), grad_clip)
        self.optimizer.step()

        return {"loss": loss.item(), "lql_loss": lql_loss.item()}

    def sync_target(self):
        """Target network를 online network로 hard copy."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str | Path) -> None:
        """체크포인트 저장."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.q_net.state_dict(),
            "N": self.config["env"]["N"],
            "hidden_dim": self.config["model"]["hidden_dim"],
            "model_type": self.config["model"]["type"],
            "agent_type": self.config["agent"]["type"],
            "step_count": self.step_count,
        }, path)

    def load(self, path: str | Path) -> None:
        """체크포인트 로드."""
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["model_state_dict"])
        self.target_net.load_state_dict(ckpt["model_state_dict"])
        self.target_net.eval()
        if "step_count" in ckpt:
            self.step_count = ckpt["step_count"]

    def load_legacy(self, path: str | Path) -> None:
        """기존 형식의 체크포인트 로드 (호환성 유지).

        기존 utils.QNetwork의 state_dict를 로드.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["model_state_dict"])
        self.target_net.load_state_dict(ckpt["model_state_dict"])
        self.target_net.eval()

    def _sample_batch(self) -> dict[str, Any]:
        """Replay buffer에서 미니배치 샘플링."""
        batch_size = self.agent_cfg["batch_size"]
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, future_states, row_hints, col_hints = zip(*samples)
        return {
            "states": list(states),
            "actions": list(actions),
            "rewards": list(rewards),
            "next_states": list(next_states),
            "dones": list(dones),
            "future_states": list(future_states),
            "row_hints": list(row_hints),
            "col_hints": list(col_hints),
        }

    def _compute_target_q(self, next_states_t: torch.Tensor,
                          next_states_np: list, N: int,
                          row_hints_t: torch.Tensor | None = None,
                          col_hints_t: torch.Tensor | None = None) -> torch.Tensor:
        """Target Q 계산 (표준 DQN: max_a' Q_target(s', a'))."""
        if row_hints_t is not None and col_hints_t is not None:
            next_q = self.target_net(next_states_t, row_hints_t, col_hints_t)
        else:
            next_q = self.target_net(next_states_t)
        # 무효 action 마스킹
        next_masks = np.stack([valid_actions_mask(s, N) for s in next_states_np])
        next_masks_t = torch.tensor(next_masks, dtype=torch.float32, device=self.device)
        next_q = next_q.masked_fill(next_masks_t == 0, -1e9)
        return next_q.max(dim=1).values
