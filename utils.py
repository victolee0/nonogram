"""
Common utilities for RL-based 1D nonogram solver.

State representation:
    s = [k hint values] + [N block states], where k = floor((N+1)/2)
    - hint part: right-aligned, zero-padded (e.g., for N=5, hint=[1,1] -> [0, 1, 1])
    - block state: -1 (X), 0 (empty), 1 (painted)

Action representation:
    a = (i, f) where i in {0,...,N-1}, f in {-1, +1}
    - Encoded as a single index in [0, 2N):
        - idx = i           -> paint (f=+1) cell i
        - idx = N + i       -> X (f=-1) cell i

Reward:
    +1 at terminal if blocks satisfy hint, -1 otherwise. gamma = 1.
"""


from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# State / hint utilities
# ---------------------------------------------------------------------------

def hint_length(N):
    """Maximum number of hint values for a row of length N."""
    return (N + 1) // 2


def encode_hint(hint, N):
    """Pad a hint list to length k by inserting zeros on the LEFT (right-align).

    A "hint" of [0] (or empty) means the row is entirely empty (no painted runs).
    """
    k = hint_length(N)
    actual = [int(h) for h in hint if int(h) > 0]
    if len(actual) > k:
        raise ValueError(f"Hint {hint} has more blocks than allowed (k={k}) for N={N}.")
    return [0] * (k - len(actual)) + actual


def extract_hint_from_blocks(blocks, N):
    """Given a fully filled row (values in {-1, +1}), return its padded hint."""
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


def initial_state(hint, N):
    """Build the initial state vector (hint part + N zeros)."""
    padded = encode_hint(hint, N)
    return np.array(padded + [0] * N, dtype=np.float32)


def get_blocks(state, N):
    """Return the block portion (last N entries) of the state."""
    return np.asarray(state)[-N:]


def get_hint_part(state, N):
    """Return the hint portion (first k entries) of the state."""
    return np.asarray(state)[:-N]


def is_terminal(state, N):
    """A state is terminal once every cell has been decided (no zeros left)."""
    return bool(np.all(get_blocks(state, N) != 0))


def check_hint_match(blocks, hint_part, N):
    """Return True iff the fully-filled blocks produce exactly `hint_part`."""
    extracted = extract_hint_from_blocks(blocks, N)
    return list(int(x) for x in extracted) == list(int(x) for x in hint_part)


# ---------------------------------------------------------------------------
# Action utilities
# ---------------------------------------------------------------------------

def action_to_idx(i, f, N):
    """Encode (cell index i, fill flag f) into a single integer in [0, 2N)."""
    if f == 1:
        return i
    elif f == -1:
        return N + i
    raise ValueError(f"f must be +1 or -1, got {f}")


def idx_to_action(idx, N):
    """Decode a flat action index back into (i, f)."""
    if idx < N:
        return idx, 1
    return idx - N, -1


def valid_actions_mask(state, N):
    """Boolean mask (length 2N) of legal actions: a cell already filled cannot be touched."""
    blocks = get_blocks(state, N)
    mask = np.zeros(2 * N, dtype=np.float32)
    for i in range(N):
        if blocks[i] == 0:
            mask[i] = 1.0          # paint
            mask[N + i] = 1.0      # X
    return mask


def apply_action(state, action_idx, N):
    """Return a new state obtained by applying `action_idx` to `state`."""
    new_state = np.array(state, dtype=np.float32, copy=True)
    i, f = idx_to_action(action_idx, N)
    new_state[-N + i] = f
    return new_state


# ---------------------------------------------------------------------------
# Feasibility check (used for early stopping during training)
# ---------------------------------------------------------------------------

def is_feasible(blocks, hint_part, N):
    """Can the partially filled `blocks` still be completed to satisfy `hint_part`?

    Empty cells (== 0) may be filled with either +1 or -1; already-decided cells
    are fixed. Returns True iff at least one completion satisfies the hint.
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


# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """MLP Q-network. Input: state (k + N), Output: Q-values over 2N actions."""

    def __init__(self, N, hidden_dim=128):
        super().__init__()
        k = hint_length(N)
        input_dim = k + N
        output_dim = 2 * N
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)