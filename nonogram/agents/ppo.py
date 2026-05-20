"""
PPO (Proximal Policy Optimization) Agent.

On-policy 알고리즘. Actor-Critic 구조로 policy와 value를 동시에 학습.
Clipped surrogate objective로 안정적인 policy 업데이트.
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from nonogram.agents.base import BaseAgent
from nonogram.agents.registry import register_agent
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


class ActorCriticNetwork(nn.Module):
    """Actor-Critic 네트워크 (공유 backbone + 분리된 head)."""

    def __init__(self, N: int, hidden_dim: int = 128, num_layers: int = 3, env_type: str = "line", model_type: str = "mlp"):
        super().__init__()
        self.env_type = env_type
        self.model_type = model_type
        self.N = N

        if env_type == "board":
            if model_type == "board_gnn":
                # 2D Board 환경용 GNN backbone
                from nonogram.models.board_gnn import BoardGNNNetwork
                self.gnn = BoardGNNNetwork(N=N, hidden_dim=hidden_dim, num_layers=num_layers)
                # GNN 모델이 평탄화된 (B, 2*N*N) logits를 출력하므로, 이를 input으로 받아 
                # Actor와 Critic용 공유 feature 공간을 만듭니다.
                self.shared = nn.Sequential(
                    nn.Linear(2 * N * N, hidden_dim),
                    nn.ReLU()
                )
                action_dim = 2 * N * N
            else:
                # 2D Board 환경용 CNN backbone
                num_channels = 5
                self.cnn = nn.Sequential(
                    nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                cnn_out_dim = 64 * N * N
                action_dim = 2 * N * N

                # 공유 backbone FC layers
                self.shared = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(cnn_out_dim, hidden_dim),
                    nn.ReLU(),
                )
        else:
            # 1D Line 환경용 MLP backbone
            k = hint_length(N)
            input_dim = k + N
            action_dim = 2 * N

            shared_layers = []
            prev_dim = input_dim
            for _ in range(num_layers - 1):
                shared_layers.append(nn.Linear(prev_dim, hidden_dim))
                shared_layers.append(nn.ReLU())
                prev_dim = hidden_dim
            self.shared = nn.Sequential(*shared_layers)

        # Actor head: action logits
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Critic head: state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, row_hints_t=None, col_hints_t=None):
        if self.env_type == "board":
            if self.model_type == "board_gnn":
                x = self.gnn(x)
            else:
                x = self.cnn(x)
        features = self.shared(x)
        return self.actor(features), self.critic(features)

    def get_action_and_value(self, x, mask, action=None, row_hints_t=None, col_hints_t=None):
        """Action 샘플링 + log_prob + entropy + value 계산."""
        logits, value = self.forward(x, row_hints_t, col_hints_t)
        # 무효 action 마스킹
        logits = logits.masked_fill(mask == 0, -1e9)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)


@register_agent("ppo")
class PPOAgent(BaseAgent):
    """PPO 에이전트."""

    def __init__(self, config: dict, device: torch.device | None = None, **kwargs):
        super().__init__(config, device)

        N = config["env"]["N"]
        hidden_dim = config["model"]["hidden_dim"]
        num_layers = config["model"]["num_layers"]
        env_type = config["env"].get("type", "line")
        model_type = config["model"].get("type", "mlp")

        self.network = ActorCriticNetwork(
            N=N, hidden_dim=hidden_dim, num_layers=num_layers, env_type=env_type, model_type=model_type
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.agent_cfg["lr"],
        )

        # PPO hyperparameters
        self.ppo_epochs = self.agent_cfg.get("ppo_epochs", 4)
        self.ppo_clip = self.agent_cfg.get("ppo_clip", 0.2)
        self.entropy_coef = self.agent_cfg.get("entropy_coef", 0.01)
        self.value_coef = self.agent_cfg.get("value_coef", 0.5)
        self.gamma = self.agent_cfg.get("gamma", 1.0)

        # Rollout buffer
        self.rollout: list[dict] = []

    def select_action(self, state: np.ndarray, mask: np.ndarray,
                      explore: bool = True, row_hints: list | None = None, col_hints: list | None = None) -> int:
        """Policy에서 action 샘플링."""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

            # 2D 보드 환경인 경우에만 힌트 텐서 가공하여 전달
            env_type = self.config["env"].get("type", "line")
            if env_type == "board" and row_hints is not None and col_hints is not None:
                N = self.config["env"]["N"]
                k = hint_length(N)
                padded_rh = torch.tensor(pad_hints_batch([row_hints], k, N), dtype=torch.long, device=self.device)
                padded_ch = torch.tensor(pad_hints_batch([col_hints], k, N), dtype=torch.long, device=self.device)
                action, log_prob, _, value = self.network.get_action_and_value(
                    state_t, mask_t, row_hints_t=padded_rh, col_hints_t=padded_ch
                )
            else:
                action, log_prob, _, value = self.network.get_action_and_value(
                    state_t, mask_t
                )

        action_int = action.item()

        # Rollout에 저장
        self._last_log_prob = log_prob.item()
        self._last_value = value.item()

        return action_int

    def store_transition(self, state, action, reward, next_state, done, mask, row_hints=None, col_hints=None):
        """Rollout buffer에 transition 저장."""
        self.rollout.append({
            "state": state.copy(),
            "action": action,
            "reward": reward,
            "done": done,
            "mask": mask.copy(),
            "log_prob": self._last_log_prob,
            "value": self._last_value,
            "row_hints": row_hints,
            "col_hints": col_hints,
        })

    def update(self, batch: dict[str, Any] | None = None) -> dict[str, float]:
        """Rollout buffer로 PPO 업데이트."""
        if len(self.rollout) == 0:
            return {}

        # GAE (Generalized Advantage Estimation)
        returns, advantages = self._compute_gae()

        states = torch.tensor(
            np.array([t["state"] for t in self.rollout]),
            dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            [t["action"] for t in self.rollout],
            dtype=torch.long, device=self.device
        )
        masks = torch.tensor(
            np.array([t["mask"] for t in self.rollout]),
            dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.tensor(
            [t["log_prob"] for t in self.rollout],
            dtype=torch.float32, device=self.device
        )
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # 2D 보드 환경인 경우 힌트 가공
        env_type = self.config["env"].get("type", "line")
        N = self.config["env"]["N"]
        is_board = (env_type == "board")
        if is_board:
            k = hint_length(N)
            padded_rh = torch.tensor(pad_hints_batch([t["row_hints"] for t in self.rollout], k, N), dtype=torch.long, device=self.device)
            padded_ch = torch.tensor(pad_hints_batch([t["col_hints"] for t in self.rollout], k, N), dtype=torch.long, device=self.device)
        else:
            padded_rh, padded_ch = None, None

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.ppo_epochs):
            if is_board:
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    states, masks, actions, row_hints_t=padded_rh, col_hints_t=padded_ch
                )
            else:
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    states, masks, actions
                )

            # Policy loss (clipped surrogate)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(new_values, returns_t)

            # LQL Loss (arXiv:2605.05812): V(s_t) >= V(s_{t+n})
            lql_weight = self.agent_cfg.get("lql_weight", 0.0)
            lql_n_step = self.agent_cfg.get("lql_n_step", 3)
            lql_loss = torch.tensor(0.0, device=self.device)
            
            if lql_weight > 0 and len(new_values) > lql_n_step:
                # 같은 rollout 내에서 n-step 뒤의 가치와 비교
                v_t = new_values[:-lql_n_step]
                v_future = new_values[lql_n_step:]
                # 부등식: v_t >= v_future (r=0, gamma=1 기준)
                lql_loss = torch.mean(torch.nn.functional.relu(v_future - v_t)**2)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss + lql_weight * lql_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.agent_cfg.get("grad_clip", 10.0)
            )
            self.optimizer.step()
            total_loss += loss.item()

        self.rollout.clear()
        return {
            "loss": total_loss / self.ppo_epochs,
            "lql_loss": lql_loss.item() if lql_weight > 0 else 0.0
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "N": self.config["env"]["N"],
            "hidden_dim": self.config["model"]["hidden_dim"],
            "model_type": "ppo_actor_critic",
            "agent_type": "ppo",
            "step_count": self.step_count,
        }, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["model_state_dict"])
        if "step_count" in ckpt:
            self.step_count = ckpt["step_count"]

    def _compute_gae(self, gae_lambda: float = 0.95) -> tuple[list, list]:
        """Generalized Advantage Estimation."""
        rewards = [t["reward"] for t in self.rollout]
        values = [t["value"] for t in self.rollout]
        dones = [t["done"] for t in self.rollout]

        returns = []
        advantages = []
        gae = 0.0
        next_value = 0.0

        for i in reversed(range(len(self.rollout))):
            if dones[i]:
                next_value = 0.0
                gae = 0.0
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            next_value = values[i]

        return returns, advantages
