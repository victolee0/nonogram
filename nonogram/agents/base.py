"""
에이전트 베이스 클래스 — 모든 RL 에이전트의 공통 인터페이스.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch


class BaseAgent(ABC):
    """RL 에이전트 베이스 클래스."""

    def __init__(self, config: dict, device: torch.device | None = None):
        self.config = config
        self.agent_cfg = config["agent"]
        self.device = device or torch.device("cpu")
        self.step_count = 0

    @abstractmethod
    def select_action(self, state: np.ndarray, mask: np.ndarray,
                      explore: bool = True) -> int:
        """State와 action mask가 주어졌을 때 action 선택.

        Args:
            state: 현재 state
            mask: 유효 action mask (1=유효, 0=무효)
            explore: True면 탐색 정책 사용, False면 greedy

        Returns:
            선택된 action index
        """
        ...

    @abstractmethod
    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """학습 step 수행.

        Args:
            batch: 경험 데이터 dict (states, actions, rewards, next_states, dones, ...)

        Returns:
            학습 메트릭 dict (loss 등)
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """모델 저장."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """모델 로드."""
        ...

    def get_epsilon(self) -> float:
        """현재 ε 값 (ε-greedy 탐색용)."""
        eps_start = self.agent_cfg.get("epsilon_start", 1.0)
        eps_end = self.agent_cfg.get("epsilon_end", 0.05)
        eps_decay = self.agent_cfg.get("epsilon_decay", 10_000)
        return eps_end + (eps_start - eps_end) * max(0, 1 - self.step_count / eps_decay)
