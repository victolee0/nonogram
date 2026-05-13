"""
Inference script for the trained 1D nonogram solver.

Given a hint and N, this script:
  1. Loads the trained Q-network.
  2. Starts from the initial state s = [hint padded to k] + [0] * N.
  3. At every step, prints Q(s, a) for all 2N actions (with masking info).
  4. Picks the next action according to the policy:
        - if any valid action has Q ~= -1, take the opposite of the *most
          confidently-bad* such action: (i, -f).
        - otherwise, take argmax_a Q(s, a) over valid actions.
  5. Repeats until terminal, then prints the final row.

Usage example:
    python inference.py --N 5 --hint 1 2 --load_path qnet_N5.pt
    python inference.py --N 5 --hint 0 --load_path qnet_N5.pt
"""

import argparse

import numpy as np
import torch

from utils import (
    QNetwork,
    action_to_idx,
    apply_action,
    get_blocks,
    hint_length,
    idx_to_action,
    initial_state,
    is_terminal,
    valid_actions_mask,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=5, help='Row length.')
    parser.add_argument('--hint', type=int, nargs='+', required=True,
                        help='Hint values, space-separated. Use "--hint 0" for an empty row.')
    parser.add_argument('--load_path', type=str, default='qnet.pt')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Only used if the checkpoint did not save it.')
    parser.add_argument('--neg_threshold', type=float, default=-0.5,
                        help='An action whose Q-value is <= this is treated as "Q == -1".')
    return parser.parse_args()


def visualise(blocks):
    """Pretty-print the row: filled cells as '#', X cells as '.'."""
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
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.load_path, map_location=device)
    N = int(ckpt.get('N', args.N))
    hidden_dim = int(ckpt.get('hidden_dim', args.hidden_dim))
    if N != args.N:
        print(f"[warn] checkpoint N={N} != --N {args.N}. Using N={N} from checkpoint.")

    q_net = QNetwork(N, hidden_dim).to(device)
    q_net.load_state_dict(ckpt['model_state_dict'])
    q_net.eval()

    state = initial_state(args.hint, N)
    k = hint_length(N)

    print(f"N = {N},   k = {k}")
    print(f"hint        : {args.hint}")
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

        # --- Policy ---
        # 1) Look for any valid action whose Q is "essentially -1".
        bad_candidates = [
            idx for idx in range(2 * N)
            if mask[idx] > 0 and q_values[idx] <= args.neg_threshold
        ]

        chosen = None
        if bad_candidates:
            # take the MOST confident bad action and flip its sign
            worst = min(bad_candidates, key=lambda j: q_values[j])
            wi, wf = idx_to_action(worst, N)
            opp = action_to_idx(wi, -wf, N)
            if mask[opp] > 0:
                chosen = opp
                reason = (f"action ({wi}, {wf:+d}) has Q={q_values[worst]:+.4f} (<= {args.neg_threshold}); "
                          f"taking opposite ({wi}, {-wf:+d}).")
            else:
                reason = None  # fall through to argmax

        if chosen is None:
            masked_q = np.where(mask > 0, q_values, -np.inf)
            chosen = int(np.argmax(masked_q))
            ci, cf = idx_to_action(chosen, N)
            reason = f"no valid action with Q <= {args.neg_threshold}; argmax picks ({ci}, {cf:+d})."

        ci, cf = idx_to_action(chosen, N)
        print(f"-> chosen: a = ({ci}, {cf:+d})    ({reason})")
        print()

        state = apply_action(state, chosen, N)

    blocks = get_blocks(state, N).astype(int)
    print("=== terminal ===")
    print(f"final state : {state.astype(int).tolist()}")
    print(f"blocks      : {blocks.tolist()}")
    print(f"row         : {visualise(blocks)}")


if __name__ == '__main__':
    main()