"""
1D 줄(Line) 수준 평가 스크립트 — 학습된 1D Q-network를 단일 1D 줄 환경에서 평가.

Usage:
    uv run python scripts/evaluate_1d.py \
        --config runs/double_dqn_10x10_double_dqn_dueling_N10/config.yaml \
        --load_path runs/double_dqn_10x10_double_dqn_dueling_N10/checkpoints/best.pt \
        --num_lines 1000
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nonogram.config import load_config, make_base_parser, resolve_device, get_experiment_dir
from nonogram.models.registry import create_model
from nonogram.env.state import (
    initial_state, apply_action, is_terminal, check_hint_match,
    valid_actions_mask, get_blocks, idx_to_action, action_to_idx
)
from utils import is_feasible


def extract_hint(line: np.ndarray) -> list[int]:
    """1D 줄에서 +1의 run lengths 추출."""
    runs = []
    cur = 0
    for v in line:
        if int(v) == 1:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs if runs else [0]


def format_hint(hint):
    return ' '.join(str(x) for x in hint)


def main():
    parser = make_base_parser()
    parser.add_argument("--num_lines", type=int, default=1000,
                        help="평가할 1D 줄 개수")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neg_threshold", type=float, default=None,
                        help="틀린 행동 판정 임계값 (None이면 config의 solver.neg_threshold 사용)")
    parser.add_argument("--load_path", type=str, default=None,
                        help="체크포인트 경로")
    known_args, remaining = parser.parse_known_args()

    config = load_config(known_args.config, remaining)
    device = resolve_device(config["device"])
    N = config["env"]["N"]

    rng = np.random.default_rng(known_args.seed)
    torch.manual_seed(known_args.seed)

    # 모델 로드
    load_path = known_args.load_path
    if load_path is None:
        exp_dir = get_experiment_dir(config)
        load_path = str(exp_dir / "checkpoints" / "latest.pt")

    ckpt = torch.load(load_path, map_location=device)

    # PPO 에이전트를 Q-net 형식으로 래핑하는 헬퍼
    class PPOToQNetWrapper(torch.nn.Module):
        def __init__(self, ppo_network):
            super().__init__()
            self.ppo_network = ppo_network
        def forward(self, x):
            logits, _ = self.ppo_network(x)
            return logits

    # Contrastive RL 에이전트를 Q-net 형식으로 래핑하는 헬퍼
    class CRLToQNetWrapper(torch.nn.Module):
        def __init__(self, crl_model, crl_mode):
            super().__init__()
            self.crl_model = crl_model
            self.crl_mode = crl_mode
        def forward(self, x):
            goal_input = {
                "state": x,
                "future_state": x
            }
            q_values, _, _ = self.crl_model(x, goal_input, self.crl_mode)
            return q_values

    agent_type = config["agent"]["type"]
    if agent_type == "ppo":
        from nonogram.agents.ppo import ActorCriticNetwork
        env_type = config["env"].get("type", "line")
        model_type = config["model"].get("type", "mlp")
        
        raw_net = ActorCriticNetwork(
            N=N,
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            env_type=env_type,
            model_type=model_type
        ).to(device)
        raw_net.load_state_dict(ckpt["model_state_dict"])
        q_net = PPOToQNetWrapper(raw_net)
    elif agent_type == "crl":
        raw_net = create_model(config).to(device)
        raw_net.load_state_dict(ckpt["model_state_dict"])
        crl_mode = config["agent"].get("crl_mode", "clue")
        q_net = CRLToQNetWrapper(raw_net, crl_mode)
    else:
        q_net = create_model(config).to(device)
        q_net.load_state_dict(ckpt["model_state_dict"])
        
    q_net.eval()

    neg_threshold = known_args.neg_threshold
    if neg_threshold is None:
        neg_threshold = config["solver"].get("neg_threshold", -0.5)

    print(f"Evaluating 1D Line performance on {known_args.num_lines} random lines")
    print(f"Model: {load_path} ({config['model']['type']}+{config['agent']['type']})")
    print(f"N: {N},  neg_threshold: {neg_threshold}")
    print()

    n_hint_ok = 0
    n_exact = 0
    n_fully_decided = 0
    total_unknown = 0
    total_agree = 0
    total_steps = 0
    total_early_stops = 0

    t_start = time.time()

    for idx in range(1, known_args.num_lines + 1):
        # 1. 1D Ground Truth 생성 및 힌트 추출
        gt = rng.choice([-1, 1], size=N).astype(np.int64)
        hint = extract_hint(gt)

        # 2. 초기 상태 초기화
        state = initial_state(hint, N)
        early_stopped = False

        # 3. 1D 풀이 루프
        steps = 0
        while not is_terminal(state, N):
            steps += 1
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = q_net(s_t).squeeze(0).cpu().numpy()

            mask = valid_actions_mask(state, N)

            # --- Policy ---
            # 1) Q <= neg_threshold 인 행동 찾기 (확실히 틀린 행동 배제)
            bad_candidates = [
                j for j in range(2 * N)
                if mask[j] > 0 and q_values[j] <= neg_threshold
            ]

            chosen = None
            if bad_candidates:
                worst = min(bad_candidates, key=lambda j: q_values[j])
                wi, wf = idx_to_action(worst, N)
                opp = action_to_idx(wi, -wf, N)
                if mask[opp] > 0:
                    chosen = opp

            if chosen is None:
                masked_q = np.where(mask > 0, q_values, -np.inf)
                chosen = int(np.argmax(masked_q))

            state = apply_action(state, chosen, N)
            blocks = get_blocks(state, N)

            # Feasibility early stop 검사
            if not is_feasible(blocks, state[:-N], N):
                early_stopped = True
                break

        # 4. 결과 평가
        blocks = get_blocks(state, N)
        num_unknown = int((blocks == 0).sum())
        
        hint_satisfied = False
        if num_unknown == 0 and not early_stopped:
            hint_satisfied = check_hint_match(blocks, state[:-N], N)

        exact_match = False
        if not early_stopped:
            exact_match = bool(np.array_equal(blocks, gt))

        agree = int(((blocks == gt) & (blocks != 0)).sum())

        if hint_satisfied:
            n_hint_ok += 1
        if exact_match:
            n_exact += 1
        if num_unknown == 0 and not early_stopped:
            n_fully_decided += 1
        if early_stopped:
            total_early_stops += 1

        total_unknown += num_unknown
        total_agree += agree
        total_steps += steps

        if idx % known_args.log_every == 0 or idx == known_args.num_lines:
            elapsed = time.time() - t_start
            print(f"[{idx:5d}/{known_args.num_lines}]  "
                  f"hint_ok={n_hint_ok/idx:.3f}  "
                  f"exact={n_exact/idx:.3f}  "
                  f"decided={n_fully_decided/idx:.3f}  "
                  f"early_stop={total_early_stops/idx:.3f}  "
                  f"agree={total_agree/(idx * N):.3f}  "
                  f"({elapsed:.1f}s)")

    print("\n" + "=" * 60)
    print("1D Line Evaluation Final Results")
    print("=" * 60)
    n = known_args.num_lines
    print(f"hint-satisfied rate      : {n_hint_ok}/{n}  ({n_hint_ok/n:.3%})")
    print(f"exact-match rate         : {n_exact}/{n}  ({n_exact/n:.3%})")
    print(f"fully decided rate       : {n_fully_decided}/{n}  ({n_fully_decided/n:.3%})")
    print(f"early stop rate          : {total_early_stops}/{n}  ({total_early_stops/n:.3%})")
    print(f"avg unknown cells / line : {total_unknown/n:.2f}  / {N}")
    print(f"cell-level agreement     : {total_agree/(n * N):.3%}  ({total_agree}/{n * N})")
    print(f"avg steps taken / line   : {total_steps/n:.2f}")
    print(f"Total time elapsed       : {time.time() - t_start:.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
