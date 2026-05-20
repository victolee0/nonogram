"""
Hybrid Solver — 1D DFS 백트래킹과 2D Board Q-network 가치 평가의 계층적 융합 솔버.

1D Line Solver의 강력한 줄 단위 백트래킹 DFS를 뼈대로 삼으며,
줄 단위 교착 상태(Deadlock)에 도달해 "임의 가정(Guessing)"이 필요한 시점에서
2D Board RL 모델의 Q-value 추론 결과(글로벌 공간 가치)를 참조하여 정밀한 기하학적 한 수를 유도합니다.
"""

import numpy as np
import torch
import torch.nn as nn

from nonogram.env.state import hint_length
from nonogram.solvers.line_solver import LineSolver
from nonogram.solvers.registry import register_solver


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


@register_solver("hybrid")
class HybridSolver(LineSolver):
    """1D DFS 논리 추론 + 2D Board RL 글로벌 공간 추론 하이브리드 솔버."""

    def __init__(self, config: dict, q_net=None, q_net_2d=None, device=None, **kwargs):
        # 부모 클래스 (LineSolver) 생성자 호출로 1D Q-network 연동
        super().__init__(config, q_net, device, **kwargs)
        
        self.q_net_2d = q_net_2d
        
        # 만약 q_net_2d가 별도로 주어지지 않고 config에 2D 모델 체크포인트 경로가 있으면 자동 로드 시도
        if self.q_net_2d is None and "board_checkpoint" in config.get("solver", {}):
            try:
                from nonogram.models.registry import create_model
                ckpt_path = config["solver"]["board_checkpoint"]
                print(f"[HybridSolver] Loading 2D Board model from {ckpt_path}...")
                
                # 임시 config를 만들어 2D 모델을 생성
                cfg_2d = config.copy()
                cfg_2d["model"] = config["solver"].get("board_model_cfg", {"type": "board_cnn", "hidden_dim": 256, "num_layers": 3})
                self.q_net_2d = create_model(cfg_2d).to(self.device)
                
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.q_net_2d.load_state_dict(ckpt["model_state_dict"])
                self.q_net_2d.eval()
                print("[HybridSolver] 2D Board model loaded successfully.")
            except Exception as e:
                print(f"[HybridSolver] Failed to auto-load 2D model: {e}")
                self.q_net_2d = None

    def _build_board_state_2d(self, board, row_hints, col_hints, N):
        """BoardEnv와 동일한 형태의 (5, N, N) 상태 텐서 구축."""
        state = np.zeros((5, N, N), dtype=np.float32)
        
        # row hint features
        for i, hints in enumerate(row_hints):
            total = sum(h for h in hints if h > 0)
            state[0, i, :] = total / N

        # col hint features
        for j, hints in enumerate(col_hints):
            total = sum(h for h in hints if h > 0)
            state[1, :, j] = total / N
            
        state[2] = (board == 1).astype(np.float32)
        state[3] = (board == -1).astype(np.float32)
        state[4] = (board == 0).astype(np.float32)
        return state

    def _get_best_guess_candidate(self, board, row_hints_padded, col_hints_padded, N):
        """교착 상태 시 2D Board Q-network를 사용하여 전체 보드에서 가장 가치가 높은 셀 선별."""
        if self.q_net_2d is None:
            # 2D 모델이 준비되지 않은 경우 1D DFS candidate 탐색 방식으로 폴백
            return super()._get_best_guess_candidate(board, row_hints_padded, col_hints_padded, N)

        # 1. 2D 보드 상태 빌드
        # row_hints_padded, col_hints_padded 등에서 clean 힌트 도출
        row_hints_clean = [[int(x) for x in h if int(x) > 0] for h in row_hints_padded]
        col_hints_clean = [[int(x) for x in h if int(x) > 0] for h in col_hints_padded]
        
        state_2d = self._build_board_state_2d(board, row_hints_clean, col_hints_clean, N)
        state_t = torch.tensor(state_2d, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 2. 2D 힌트 배치 패딩 생성
        k = hint_length(N)
        padded_rh = torch.tensor(pad_hints_batch([row_hints_clean], k, N), dtype=torch.long, device=self.device)
        padded_ch = torch.tensor(pad_hints_batch([col_hints_clean], k, N), dtype=torch.long, device=self.device)
        
        # 3. 2D Q-network 포워딩
        with torch.no_grad():
            # 모델 타입에 따라 힌트 텐서 전달 분기
            model_type = self.config.get("solver", {}).get("board_model_cfg", {}).get("type", "board_cnn")
            if model_type in ("board_cnn", "board_transformer"):
                q_values = self.q_net_2d(state_t, padded_rh, padded_ch).squeeze(0).cpu().numpy()
            else:
                q_values = self.q_net_2d(state_t).squeeze(0).cpu().numpy()

        # 4. 미확정 셀 전용 valid mask 생성
        flat_board = board.flatten()
        mask = np.zeros(2 * N * N, dtype=np.float32)
        for i in range(N * N):
            if flat_board[i] == 0:
                mask[i] = 1.0               # Paint
                mask[N * N + i] = 1.0       # X

        # 무효 액션 차단
        q_values = np.where(mask > 0, q_values, -np.inf)
        
        # 5. 최대 Q-value 액션 선출
        best_act = int(np.argmax(q_values))
        q_val = float(q_values[best_act])
        
        if q_val == -np.inf:
            return None
            
        cell_idx = best_act % (N * N)
        r = cell_idx // N
        c = cell_idx % N
        f = 1 if best_act < N * N else -1
        
        # DFS 백트래킹의 통일 규격에 맞게 변환하여 리턴
        # (axis, line_idx, cell_idx_in_line, first_val, second_val, q_val)
        # 행 기준으로 가정 처리
        return "row", r, c, f, -f, q_val
