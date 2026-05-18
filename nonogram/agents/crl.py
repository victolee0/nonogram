"""
Contrastive RL (CRL) Agent — 1000층 아키텍처 학습을 안정화하기 위한 대비 학습 에이전트.

InfoNCE (Contrastive Loss)를 활용하여 정답/도달 상태의 임베딩을 가깝게 당기고,
부정적인 매칭 임베딩을 밀어냅니다. HER 모드와 Clue 모드를 둘 다 지원합니다.
"""

import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nonogram.agents.base import BaseAgent
from nonogram.agents.registry import register_agent
from nonogram.models.registry import create_model
from nonogram.env.state import valid_actions_mask


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


@register_agent("crl")
class CRLAgent(BaseAgent):
    """대비 학습(Contrastive RL) 에이전트."""

    def __init__(self, config: dict, device: torch.device | None = None,
                 model: nn.Module | None = None):
        super().__init__(config, device)

        # CRL 전용 하이퍼파라미터
        self.crl_mode = self.agent_cfg.get("crl_mode", "clue") # "her" | "clue"
        self.temp = self.agent_cfg.get("crl_temperature", 0.1)

        # 모델 생성
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = create_model(config).to(self.device)

        # 옵티마이저
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.agent_cfg["lr"],
        )

        # 경험 재생 버퍼
        self.buffer = deque(maxlen=self.agent_cfg["buffer_size"])

    def select_action(self, state: np.ndarray, mask: np.ndarray,
                      explore: bool = True) -> int:
        """현재 상태에서 대비 임베딩 내적(Q-value) 기반의 행동 선택."""
        if explore and random.random() < self.get_epsilon():
            valid = np.where(mask > 0)[0]
            if len(valid) == 0:
                return 0
            return int(random.choice(valid))

        # 추론 시에는 현재 주어진 Clues(단서)를 목표로 삼고 Q-value 산출
        with torch.no_grad():
            N = self.config["env"]["N"]
            k = (N + 1) // 2
            
            # 1D 또는 2D 감지
            is_1d = (len(state.shape) == 1)
            
            if is_1d:
                state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                goal_input = {
                    "state": state_t,
                    "future_state": state_t
                }
                q_values, _, _ = self.model(state_t, goal_input, self.crl_mode)
                q_values = q_values.squeeze(0).cpu().numpy()
            else:
                # 2D 보드 상태 및 단서 준비
                b_state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                board_t = b_state_t[:, 2, :, :] - b_state_t[:, 3, :, :]
                
                # 임시로 더미 hints를 사용하여 forward 호출
                dummy_rh = torch.zeros((1, N, k), dtype=torch.long, device=self.device)
                dummy_ch = torch.zeros((1, N, k), dtype=torch.long, device=self.device)
                
                goal_input = {
                    "row_hints": dummy_rh,
                    "col_hints": dummy_ch,
                    "future_board": board_t
                }
                
                q_values, _, _ = self.model(board_t, dummy_rh, dummy_ch, goal_input, self.crl_mode)
                q_values = q_values.squeeze(0).cpu().numpy()

        # 무효 action 마스킹
        q_values = np.where(mask > 0, q_values, -np.inf)
        return int(np.argmax(q_values))

    def store_transition(self, state, action, reward, next_state, done, future_state=None, row_hints=None, col_hints=None):
        """대비 학습을 위한 전환 데이터 저장."""
        transition = (
            state.copy(),
            action,
            reward,
            next_state.copy(),
            float(done),
            future_state.copy() if future_state is not None else next_state.copy(),
            row_hints,
            col_hints
        )
        self.buffer.append(transition)

    def can_train(self) -> bool:
        return len(self.buffer) >= self.agent_cfg["min_buffer"] and len(self.buffer) >= self.agent_cfg["batch_size"]


    def should_train(self) -> bool:
        return (self.can_train() and
                self.step_count % self.agent_cfg["train_freq"] == 0)

    def should_sync_target(self) -> bool:
        # 대비 학습에서는 Target Network 동기가 필요하지 않음
        return False

    def sync_target(self):
        pass

    def update(self, batch: dict[str, Any] | None = None) -> dict[str, float]:
        """InfoNCE 대비 손실 함수로 네트워크 가중치 업데이트."""
        if batch is None:
            if not self.can_train():
                return {}
            batch = self._sample_batch()

        N = self.config["env"]["N"]
        k = (N + 1) // 2
        grad_clip = self.agent_cfg.get("grad_clip", 1.0)

        # 1. 텐서 변환
        b_states = torch.tensor(np.array(batch["states"]), dtype=torch.float32, device=self.device)
        b_actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        b_futures = torch.tensor(np.array(batch["future_states"]), dtype=torch.float32, device=self.device)
        
        is_1d = (len(b_states.shape) == 2)
        B = b_states.size(0)

        if is_1d:
            # 4. 대비 상태 표상 및 목표 표상 도출
            phi_s_a = self.model.encode_state(b_states) # (B, 2*N, dim)
            
            # 5. 실제로 행해진 행동의 표상만 추출 (b_actions: [0, 2*N))
            phi_taken = phi_s_a[torch.arange(B), b_actions, :] # (B, dim)
            
            # 6. Goal 임베딩 구성
            goal_input = {
                "state": b_states,
                "future_state": b_futures
            }
            psi_g = self.model.encode_goal(goal_input, self.crl_mode) # (B, dim)
        else:
            N2 = N * N
            # 2. 보드 상태 재구성: (filled - empty) -> {-1, 0, 1}
            board = b_states[:, 2, :, :] - b_states[:, 3, :, :]
            future_board = b_futures[:, 2, :, :] - b_futures[:, 3, :, :]

            # 3. 힌트 제로 패딩
            padded_rh = torch.tensor(pad_hints_batch(batch["row_hints"], k, N), dtype=torch.long, device=self.device)
            padded_ch = torch.tensor(pad_hints_batch(batch["col_hints"], k, N), dtype=torch.long, device=self.device)

            # 4. 대비 상태 표상 및 목표 표상 도출
            # phi_s_a: (B, 2, N, N, dim)
            phi_s_a = self.model.encode_state(board, padded_rh, padded_ch)
            
            # 5. 실제로 행해진 행동(action)의 표상만 배치에서 슬라이싱하여 추출
            # action은 flat index in [0, 2*N*N)
            action_channel = (b_actions >= N2).long()
            cell_idx = b_actions % N2
            r = cell_idx // N
            c = cell_idx % N
            
            phi_taken = phi_s_a[torch.arange(B), action_channel, r, c, :] # (B, dim)

            # 6. Goal 임베딩 구성
            goal_input = {
                "future_board": future_board,
                "row_hints": padded_rh,
                "col_hints": padded_ch
            }
            psi_g = self.model.encode_goal(goal_input, self.crl_mode) # (B, dim)

        # 7. InfoNCE Loss 계산 (코사인 유사도 기반 배치 내 매칭 분류)
        phi_norm = F.normalize(phi_taken, dim=-1)
        psi_norm = F.normalize(psi_g, dim=-1)
        
        # 유사도 행렬: (B, B)
        similarity_matrix = torch.matmul(phi_norm, psi_norm.T) / self.temp
        
        # 정답 목표 인덱스는 배치 상의 자기 번호 (Diagonal)
        targets = torch.arange(B, device=self.device)
        
        loss = F.cross_entropy(similarity_matrix, targets)

        # 8. 역전파 및 최적화
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        return {"loss": loss.item()}

    def _sample_batch(self) -> dict[str, Any]:
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
            "col_hints": list(col_hints)
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "N": self.config["env"]["N"],
            "hidden_dim": self.config["model"]["hidden_dim"],
            "model_type": self.config["model"]["type"],
            "agent_type": self.config["agent"]["type"],
            "crl_mode": self.crl_mode,
            "step_count": self.step_count,
        }, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "step_count" in ckpt:
            self.step_count = ckpt["step_count"]
