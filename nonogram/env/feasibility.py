"""
Feasibility check — 부분 채움 줄이 hint를 만족시킬 수 있는지 판정.

학습 중 early stopping에 사용: 더 이상 정답 도달이 불가한 dead-end를
조기에 reward -1로 끊어 학습 효율을 높인다.
"""

from functools import lru_cache

import numpy as np


def is_feasible(blocks, hint_part, N: int) -> bool:
    """부분적으로 채워진 blocks가 어떤 완성으로든 hint_part를 만족시킬 수 있는지.

    Empty cells (== 0)은 +1 또는 -1로 채울 수 있고, 이미 결정된 cell은 고정.
    backtracking + memoization으로 판정.
    """
    H = tuple(int(h) for h in hint_part if int(h) > 0)
    B = tuple(int(b) for b in blocks)

    @lru_cache(maxsize=None)
    def rec(i, h_idx, cur_len):
        if i == N:
            if cur_len > 0:
                return h_idx == len(H) - 1 and cur_len == H[h_idx]
            return h_idx == len(H)

        cell = B[i]
        options = (1, -1) if cell == 0 else (cell,)
        for val in options:
            if val == 1:
                if h_idx >= len(H):
                    continue
                if cur_len + 1 > H[h_idx]:
                    continue
                if rec(i + 1, h_idx, cur_len + 1):
                    return True
            else:  # val == -1
                if cur_len > 0:
                    if cur_len != H[h_idx]:
                        continue
                    if rec(i + 1, h_idx + 1, 0):
                        return True
                else:
                    if rec(i + 1, h_idx, 0):
                        return True
        return False

    return rec(0, 0, 0)
