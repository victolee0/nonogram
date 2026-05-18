import random
import numpy as np
from nonogram.env.state import (
    initial_state, apply_action, is_terminal, check_hint_match,
    valid_actions_mask, extract_hint_from_blocks, get_blocks
)
from utils import is_feasible

class LineEnv:
    """1D 노노그램 줄(Line) 환경."""
    def __init__(self, N: int, early_stop: bool = True, 
                 reward_type: str = "step_correct",
                 reward_shaping: bool = False,
                 reward_shaping_weight: float = 0.1):
        self.N = N
        self.early_stop = early_stop
        self.reward_type = reward_type
        self.reward_shaping = reward_shaping
        self.reward_shaping_weight = reward_shaping_weight
        
        self.state = None
        self.gt_line = None
        self.hint = None

    def reset(self, gt_line: np.ndarray = None, hint: list = None) -> np.ndarray:
        if gt_line is not None:
            self.gt_line = gt_line.copy()
            self.hint = self._extract_hint(self.gt_line)
        elif hint is not None:
            self.gt_line = None
            self.hint = hint
        else:
            self.gt_line = np.random.choice([-1, 1], size=self.N).astype(np.int64)
            self.hint = self._extract_hint(self.gt_line)
            
        self.state = initial_state(self.hint, self.N)
        return self.state

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        # step_correct reward 계산을 위해 이전 blocks 저장
        prev_blocks = get_blocks(self.state, self.N).copy()
        
        self.state = apply_action(self.state, action, self.N)
        blocks = get_blocks(self.state, self.N)
        
        # 1. Terminal Reward
        if is_terminal(self.state, self.N):
            reward = 1.0 if check_hint_match(blocks, self.state[:-self.N], self.N) else -1.0
            return self.state, reward, True
            
        # 2. Early Stop (Feasibility)
        if self.early_stop and not is_feasible(blocks, self.state[:-self.N], self.N):
            return self.state, -1.0, True
            
        # 3. Intermediate Reward
        reward = 0.0
        if self.reward_type == "step_correct" and self.gt_line is not None:
            # 방금 둔 수가 GT와 일치하는지 확인
            cell_idx = action % self.N
            fill_val = 1 if action < self.N else -1
            if self.gt_line[cell_idx] == fill_val:
                reward = 0.1
            else:
                reward = -0.1
                
        return self.state, reward, False

    def get_valid_mask(self) -> np.ndarray:
        return valid_actions_mask(self.state, self.N)

    def select_towards_action(self) -> int | None:
        if self.gt_line is None: return None
        blocks = get_blocks(self.state, self.N)
        empty = [i for i in range(self.N) if blocks[i] == 0]
        if not empty: return None
        i = random.choice(empty)
        f = int(self.gt_line[i])
        return i if f == 1 else self.N + i

    def _extract_hint(self, line: np.ndarray) -> list[int]:
        runs = []
        cur = 0
        for v in line:
            if v == 1: cur += 1
            else:
                if cur > 0: runs.append(cur)
                cur = 0
        if cur > 0: runs.append(cur)
        return runs if runs else [0]
