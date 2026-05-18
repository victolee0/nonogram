"""
Double DQN Agent — Overestimation bias를 줄이기 위해
action 선택(online)과 가치 평가(target)를 분리.

target = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
"""

import numpy as np
import torch

from nonogram.agents.dqn import DQNAgent
from nonogram.agents.registry import register_agent
from nonogram.env.state import valid_actions_mask


@register_agent("double_dqn")
class DoubleDQNAgent(DQNAgent):
    """Double DQN 에이전트.

    DQNAgent를 상속하고, target Q 계산만 오버라이드.
    """

    def _compute_target_q(self, next_states_t: torch.Tensor,
                          next_states_np: list, N: int) -> torch.Tensor:
        """Double DQN target Q:
        1. Online network로 best action 선택
        2. Target network로 그 action의 Q값 평가
        """
        # 무효 action mask
        next_masks = np.stack([valid_actions_mask(s, N) for s in next_states_np])
        next_masks_t = torch.tensor(next_masks, dtype=torch.float32, device=self.device)

        # 1. Online net으로 action 선택
        online_q = self.q_net(next_states_t)
        online_q = online_q.masked_fill(next_masks_t == 0, -1e9)
        best_actions = online_q.argmax(dim=1, keepdim=True)   # (batch, 1)

        # 2. Target net으로 그 action의 가치 평가
        target_q = self.target_net(next_states_t)
        target_q_selected = target_q.gather(1, best_actions).squeeze(1)

        return target_q_selected
