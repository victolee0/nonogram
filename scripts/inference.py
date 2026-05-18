"""
1D 추론 스크립트 — 학습된 모델로 한 줄 풀이 과정을 단계별로 시각화.

Usage:
    uv run python scripts/inference.py --config configs/dqn_5x5.yaml --hint 1 2
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nonogram.config import load_config, make_base_parser, resolve_device, get_experiment_dir
from nonogram.models.registry import create_model
from nonogram.env.state import (
    action_to_idx,
    apply_action,
    get_blocks,
    hint_length,
    idx_to_action,
    initial_state,
    is_terminal,
    valid_actions_mask,
)


def visualise(blocks):
    chars = []
    for b in blocks:
        if int(b) == 1:
            chars.append('#')
        elif int(b) == -1:
            chars.append('.')
        else:
            chars.append('?')
    return ''.join(chars)


def main():
    parser = make_base_parser()
    parser.add_argument("--hint", type=int, nargs="+", required=True,
                        help="Hint 값들 (공백 구분). 빈 줄: --hint 0")
    parser.add_argument("--load_path", type=str, default=None)
    known_args, remaining = parser.parse_known_args()

    config = load_config(known_args.config, remaining)
    device = resolve_device(config["device"])
    N = config["env"]["N"]
    neg_threshold = config["solver"]["neg_threshold"]

    # 모델 로드
    load_path = known_args.load_path
    if load_path is None:
        exp_dir = get_experiment_dir(config)
        load_path = str(exp_dir / "checkpoints" / "latest.pt")

    ckpt = torch.load(load_path, map_location=device)
    q_net = create_model(config).to(device)
    q_net.load_state_dict(ckpt["model_state_dict"])
    q_net.eval()

    k = hint_length(N)
    state = initial_state(known_args.hint, N)

    print(f"N = {N},   k = {k}")
    print(f"hint        : {known_args.hint}")
    print(f"hint padded : {state[:k].astype(int).tolist()}")
    print(f"initial s   : {state.astype(int).tolist()}")
    print()

    step = 0
    while not is_terminal(state, N):
        step += 1
        s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(s_t).squeeze(0).cpu().numpy()

        mask = valid_actions_mask(state, N)

        print(f"=== step {step} ===")
        print(f"state: {state.astype(int).tolist()}    "
              f"row: {visualise(get_blocks(state, N))}")
        print("Q(s, a):")
        for idx in range(2 * N):
            i, f = idx_to_action(idx, N)
            valid = mask[idx] > 0
            tag = "" if valid else "  [masked]"
            print(f"  a = ({i}, {f:+d})  ->  Q = {q_values[idx]:+.4f}{tag}")

        bad_candidates = [
            idx for idx in range(2 * N)
            if mask[idx] > 0 and q_values[idx] <= neg_threshold
        ]

        chosen = None
        if bad_candidates:
            worst = min(bad_candidates, key=lambda j: q_values[j])
            wi, wf = idx_to_action(worst, N)
            opp = action_to_idx(wi, -wf, N)
            if mask[opp] > 0:
                chosen = opp
                reason = (f"action ({wi}, {wf:+d}) has Q={q_values[worst]:+.4f} "
                          f"(<= {neg_threshold}); taking opposite ({wi}, {-wf:+d}).")

        if chosen is None:
            masked_q = np.where(mask > 0, q_values, -np.inf)
            chosen = int(np.argmax(masked_q))
            ci, cf = idx_to_action(chosen, N)
            reason = f"no valid action with Q <= {neg_threshold}; argmax picks ({ci}, {cf:+d})."

        ci, cf = idx_to_action(chosen, N)
        print(f"-> chosen: a = ({ci}, {cf:+d})    ({reason})")
        print()

        state = apply_action(state, chosen, N)

    blocks = get_blocks(state, N).astype(int)
    print("=== terminal ===")
    print(f"final state : {state.astype(int).tolist()}")
    print(f"blocks      : {blocks.tolist()}")
    print(f"row         : {visualise(blocks)}")


if __name__ == "__main__":
    main()
