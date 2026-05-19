"""
GFlowNet Agent — Trajectory Balance (TB) 목적 함수를 사용하는 GFlowNet 에이전트.
"""

import random
from pathlib import Path
from typing import Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nonogram.agents.base import BaseAgent
from nonogram.agents.registry import register_agent
from nonogram.models.registry import create_model


@register_agent("gflownet")
class GFlowNetAgent(BaseAgent):
    """Trajectory Balance (TB) 목적 함수 기반 GFlowNet 에이전트."""

    def __init__(self, config: dict, device: torch.device | None = None, **kwargs):
        super().__init__(config, device)

        self.N = config["env"]["N"]
        self.epsilon = self.agent_cfg.get("epsilon", 1e-4)
        self.reward_temp = self.agent_cfg.get("reward_temp", 1.0)
        self.pb_type = self.agent_cfg.get("pb_type", "uniform")
        self.pb_epsilon = self.agent_cfg.get("pb_epsilon", 0.1)
        self.batch_size = self.agent_cfg.get("batch_size", 128)
        self.grad_clip = self.agent_cfg.get("grad_clip", 10.0)

        # Symmetry GNN 모델 또는 지정된 모델 생성
        self.model = create_model(config).to(self.device)

        # 분배 함수 Z의 로그값 (log Z)
        self.log_Z = nn.Parameter(torch.tensor(0.0, device=self.device))

        # Optimizer에 모델 파라미터와 log Z를 함께 등록
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + [self.log_Z],
            lr=self.agent_cfg["lr"]
        )

        # Trajectory Replay Buffer
        buffer_size = self.agent_cfg.get("buffer_size", 5000)
        self.buffer = deque(maxlen=buffer_size)

        # 현재 수집 중인 trajectory
        self.current_trajectory = []

        self._last_state = None
        self._last_action = None
        self._last_mask = None

    def select_action(self, state: np.ndarray, mask: np.ndarray, explore: bool = True) -> int:
        """현재 정책에 따라 행 채우기 액션을 선택."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # 모델 출력 로짓: (1, N, 2^N)
            logits = self.model(state_t, symmetrize=True)
            logits_flat = logits.view(1, -1)
            # 유효하지 않은 액션 마스킹 (-infinity)
            logits_flat = logits_flat.masked_fill(mask_t == 0, -1e9)

            if explore:
                dist = torch.distributions.Categorical(logits=logits_flat)
                action = dist.sample().item()
            else:
                action = logits_flat.argmax(dim=-1).item()

        self._last_state = state.copy()
        self._last_action = action
        self._last_mask = mask.copy()

        return action

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool, mask: np.ndarray = None,
                         row_hints: list = None, col_hints: list = None):
        """에피소드 내 Transition을 임시 큐에 적재하고, 완료 시 Trajectory로 저장."""
        row_idx = action // (2 ** self.N)

        self.current_trajectory.append({
            "state": state.copy(),
            "action": action,
            "mask": mask.copy() if mask is not None else self._last_mask,
            "next_state": next_state.copy(),
            "row_idx": row_idx,
        })

        if done:
            # Trajectory 전체 저장
            trajectory = {
                "transitions": list(self.current_trajectory),
                "reward": reward
            }
            self.buffer.append(trajectory)
            self.current_trajectory.clear()

    def update(self, batch: dict[str, Any] | None = None) -> dict[str, float]:
        """Trajectory Balance (TB) Loss 계산 및 가중치 업데이트 (Vectorized)."""
        if len(self.buffer) < self.batch_size:
            return {}

        # Trajectories 무작위 샘플링
        sampled_trajs = random.sample(self.buffer, self.batch_size)

        all_states = []
        all_masks = []
        all_actions = []
        all_pb_logs = []
        traj_indices = []

        for traj_idx, traj in enumerate(sampled_trajs):
            transitions = traj["transitions"]
            for trans in transitions:
                all_states.append(trans["state"])
                all_masks.append(trans["mask"])
                all_actions.append(trans["action"])

                # s_t와 s_{t+1}의 보드 상태
                s = trans["state"]
                ns = trans["next_state"]
                prev_board = s[2] - s[3]
                curr_board = ns[2] - ns[3]

                # Backward probability P_B 계산
                pb = self.get_backward_prob(prev_board, curr_board, trans["row_idx"])
                all_pb_logs.append(np.log(pb + 1e-15))
                traj_indices.append(traj_idx)

        # PyTorch 텐서 변환
        all_states_t = torch.tensor(np.array(all_states), dtype=torch.float32, device=self.device)
        all_masks_t = torch.tensor(np.array(all_masks), dtype=torch.float32, device=self.device)
        all_actions_t = torch.tensor(all_actions, dtype=torch.long, device=self.device)

        # 모델 순전파 (배치 일괄 수행)
        # symmetrize=True 로 설정하여 정확한 Equivariant 확률 계산
        all_logits = self.model(all_states_t, symmetrize=True)
        all_logits_flat = all_logits.view(all_logits.size(0), -1)
        all_logits_flat = all_logits_flat.masked_fill(all_masks_t == 0, -1e9)

        all_log_probs = F.log_softmax(all_logits_flat, dim=-1)
        all_log_pf = all_log_probs.gather(1, all_actions_t.unsqueeze(1)).squeeze(1)

        # 각 Trajectory별 sum(log P_F) 및 sum(log P_B) 집계
        log_pf_sums = torch.zeros(self.batch_size, device=self.device)
        log_pb_sums = torch.zeros(self.batch_size, device=self.device)

        for idx, traj_idx in enumerate(traj_indices):
            log_pf_sums[traj_idx] += all_log_pf[idx]
            log_pb_sums[traj_idx] += all_pb_logs[idx]

        # Terminal Reward R(x) 텐서 생성
        rewards = [max(traj["reward"], self.epsilon) for traj in sampled_trajs]
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        log_R = torch.log(rewards_t)

        # Trajectory Balance Loss: L_TB = (log Z + sum(log P_F) - log R - sum(log P_B))^2
        losses = (self.log_Z + log_pf_sums - log_R - log_pb_sums) ** 2
        loss_batch = losses.mean()

        # 경사하강법 적용
        self.optimizer.zero_grad()
        loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + [self.log_Z],
            self.grad_clip
        )
        self.optimizer.step()

        self.step_count += 1

        return {
            "loss": loss_batch.item(),
            "log_Z": self.log_Z.item()
        }

    def get_backward_prob(self, prev_board: np.ndarray, curr_board: np.ndarray, cleared_row: int) -> float:
        """curr_board -> prev_board로의 Backward Transition 확률 계산."""
        filled_rows = np.any(curr_board != 0, axis=1)
        k = int(np.sum(filled_rows))
        if k == 0:
            return 1.0

        if self.pb_type == "uniform":
            return 1.0 / k

        # 칠해진 픽셀 수에 반비례하는 경험적 분포
        weights = []
        target_weight = 0.0
        for r in range(self.N):
            if filled_rows[r]:
                c_r = int(np.sum(curr_board[r] == 1))
                w = 1.0 / (c_r + self.pb_epsilon)
                weights.append(w)
                if r == cleared_row:
                    target_weight = w

        sum_w = sum(weights)
        if sum_w == 0:
            return 1.0 / k
        return target_weight / sum_w

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "log_Z": self.log_Z.item(),
            "N": self.N,
            "hidden_dim": self.config["model"]["hidden_dim"],
            "model_type": "symmetry_gnn",
            "agent_type": "gflownet",
            "step_count": self.step_count,
        }, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.log_Z.data.copy_(torch.tensor(ckpt["log_Z"], device=self.device))
        if "step_count" in ckpt:
            self.step_count = ckpt["step_count"]
