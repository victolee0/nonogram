"""
State/Action 인코딩 유틸리티.

기존 utils.py에서 분리한 핵심 state/action 표현 로직.

State 표현:
    s = [k hint values] + [N block states], where k = floor((N+1)/2)
    - hint part: right-aligned, zero-padded
    - block state: -1 (X), 0 (empty), 1 (painted)

Action 표현:
    a = (i, f) where i in {0,...,N-1}, f in {-1, +1}
    - idx = i           → paint (f=+1) cell i
    - idx = N + i       → X (f=-1) cell i
"""

import numpy as np


# ---------------------------------------------------------------------------
# Hint utilities
# ---------------------------------------------------------------------------

def hint_length(N: int) -> int:
    """한 줄(길이 N)의 hint 최대 개수: k = floor((N+1)/2)."""
    return (N + 1) // 2


def encode_hint(hint: list[int], N: int) -> list[int]:
    """Hint 리스트를 길이 k로 left-zero-padding (right-align).

    hint=[1,1], N=5 → [0, 1, 1]
    hint=[0] (빈 줄) → [0, 0, 0]
    """
    k = hint_length(N)
    actual = [int(h) for h in hint if int(h) > 0]
    if len(actual) > k:
        raise ValueError(f"Hint {hint} has more blocks than allowed (k={k}) for N={N}.")
    return [0] * (k - len(actual)) + actual


def extract_hint_from_blocks(blocks, N: int) -> list[int]:
    """완성된 줄 ({-1, +1}^N)에서 padded hint를 추출."""
    runs = []
    cur = 0
    for v in blocks:
        if int(v) == 1:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    if not runs:
        runs = [0]
    return encode_hint(runs, N)


# ---------------------------------------------------------------------------
# State utilities
# ---------------------------------------------------------------------------

def initial_state(hint: list[int], N: int) -> np.ndarray:
    """Hint만 주어진 초기 state 벡터 생성."""
    padded = encode_hint(hint, N)
    return np.array(padded + [0] * N, dtype=np.float32)


def get_blocks(state, N: int) -> np.ndarray:
    """State에서 block 부분 (마지막 N개) 반환."""
    return np.asarray(state)[-N:]


def get_hint_part(state, N: int) -> np.ndarray:
    """State에서 hint 부분 (앞 k개) 반환."""
    return np.asarray(state)[:-N]


def is_terminal(state, N: int) -> bool:
    """모든 cell이 결정되었는지 (0이 없는지)."""
    return bool(np.all(get_blocks(state, N) != 0))


def check_hint_match(blocks, hint_part, N: int) -> bool:
    """완성된 blocks가 hint_part와 일치하는지."""
    extracted = extract_hint_from_blocks(blocks, N)
    return list(int(x) for x in extracted) == list(int(x) for x in hint_part)


# ---------------------------------------------------------------------------
# Action utilities
# ---------------------------------------------------------------------------

def action_to_idx(i: int, f: int, N: int) -> int:
    """(cell index i, fill flag f) → flat action index in [0, 2N)."""
    if f == 1:
        return i
    elif f == -1:
        return N + i
    raise ValueError(f"f must be +1 or -1, got {f}")


def idx_to_action(idx: int, N: int) -> tuple[int, int]:
    """Flat action index → (cell index i, fill flag f)."""
    if idx < N:
        return idx, 1
    return idx - N, -1


def valid_actions_mask(state, N: int) -> np.ndarray:
    """유효 action mask (Line 환경은 2N, Board 환경은 2N^2). 이미 결정된 cell의 action은 0."""
    # 2D Board 상태인 경우 (shape이 (channels, N, N) 3차원인 경우)
    if len(state.shape) == 3:
        flat_unknown = state[4].flatten()
        mask = np.zeros(2 * N * N, dtype=np.float32)
        mask[:N*N] = flat_unknown
        mask[N*N:] = flat_unknown
        return mask

    # 1D Line 환경 상태인 경우
    blocks = get_blocks(state, N)
    mask = np.zeros(2 * N, dtype=np.float32)
    for i in range(N):
        if blocks[i] == 0:
            mask[i] = 1.0       # paint
            mask[N + i] = 1.0   # X
    return mask


def apply_action(state, action_idx: int, N: int) -> np.ndarray:
    """Action 적용 후 새 state 반환 (원본 불변)."""
    new_state = np.array(state, dtype=np.float32, copy=True)
    i, f = idx_to_action(action_idx, N)
    new_state[-N + i] = f
    return new_state
