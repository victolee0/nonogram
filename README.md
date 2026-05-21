# RL-based Nonogram Solver

강화학습(RL)을 활용한 노노그램(네모로직) 풀이 프로젝트.

다양한 RL 모델, 알고리즘, 환경 설정을 **YAML config** 기반으로 교체/실험할 수 있는 모듈화된 구조.

---

## 폴더 구조

```
.
├── pyproject.toml              # uv 기반 프로젝트 정의
├── configs/                    # YAML 설정 파일
│   ├── default.yaml            # 기본 설정 (베이스 설정 제공)
│   ├── dqn_5x5.yaml            # 5x5 표준 DQN
│   ├── dqn_10x10.yaml          # 10x10 표준 DQN
│   ├── double_dqn_10x10.yaml   # 10x10 Double DQN + Dueling
│   ├── dqn_10x10_super_1d.yaml # 10x10 Double DQN 1D 성능 극대화 설정
│   ├── ppo_10x10.yaml          # 10x10 PPO
│   ├── board_dqn_10x10.yaml    # 10x10 2D Board DQN
│   ├── dqn_chunking_mlp_10x10.yaml  # 10x10 DQN + Action Chunking (MLP)
│   ├── dqn_chunking_transformer_10x10.yaml # 10x10 DQN + Action Chunking (Transformer)
│   ├── crl_clue_10x10.yaml     # 10x10 2D Board Clue 매칭 Contrastive RL
│   ├── crl_her_10x10.yaml      # 10x10 2D Board HER 기반 Contrastive RL
│   ├── crl_line_clue_10x10.yaml # 10x10 1D Line Clue 매칭 Contrastive RL
│   ├── crl_line_her_10x10.yaml  # 10x10 1D Line HER 기반 Contrastive RL
│   ├── alphazero_2d.yaml       # AlphaZero 2D MCTS + Transformer
│   ├── dqn_board_2d_rl.yaml    # [NEW] 2D Board Q-network (Cross-Attention) 학습 설정
│   ├── dqn_board_2d_rl_parallel.yaml # [NEW] Ray 기반 병렬 2D Board RL 학습 설정
│   └── gflownet_10x10.yaml     # 10x10 GFlowNet + Symmetry GNN (Trajectory Balance)
├── nonogram/                   # 메인 패키지
│   ├── config.py               # Config 로딩 및 유효성 검증
│   ├── env/                    # 환경 모듈
│   │   ├── state.py            # State/Action 인코딩 및 유틸리티
│   │   ├── line_env.py         # 1D 줄 환경
│   │   ├── board_env.py        # 2D 보드 환경
│   │   ├── feasibility.py      # Feasibility check (early stopping)
│   │   └── gflownet_env.py     # GFlowNet용 생성적 MDP 환경
│   ├── models/                 # Q-Network / Policy Network
│   │   ├── registry.py         # 모델 레지스트리
│   │   ├── mlp.py              # MLP (기존 QNetwork 호환)
│   │   ├── dueling.py          # Dueling DQN (Shared, Advantage, Value)
│   │   ├── cnn.py              # 1D CNN
│   │   ├── board_cnn.py        # 2D Board CNN
│   │   ├── board_cross_attn.py # [NEW] 2D Board Row/Col Cross-Attention 모델
│   │   ├── deep_crl.py         # 초심층 대비 학습 모델 (LayerScale, GroupNorm 탑재)
│   │   └── symmetry_gnn.py     # Symmetry-Aware Bipartite GNN
│   ├── agents/                 # RL 알고리즘
│   │   ├── registry.py         # 에이전트 레지스트리
│   │   ├── base.py             # 베이스 클래스 (LQL 손실 추가)
│   │   ├── dqn.py              # 표준 DQN (LQL 적용)
│   │   ├── double_dqn.py       # Double DQN (LQL 적용)
│   │   ├── ppo.py              # PPO (LQL 적용)
│   │   ├── crl.py              # Contrastive RL 에이전트 (InfoNCE 대비 손실 적용)
│   │   └── gflownet.py         # GFlowNet 에이전트 (Trajectory Balance 목적 함수)
│   ├── solvers/                # 2D 풀이 전략
│   │   ├── registry.py         # 풀이기 레지스트리
│   │   ├── line_solver.py      # 줄 교대 풀이 (DQN 기반 2D 풀이)
│   │   ├── board_solver.py     # 보드 레벨 풀이
│   │   └── hybrid_solver.py    # 1D DFS와 2D RL의 계층적 하이브리드 풀이 [NEW]
│   └── utils/                  # 유틸리티
│       ├── visualization.py    # 보드 렌더링
│       └── puzzle_io.py        # 퍼즐 I/O (YAML, CLI)
├── models_2d.py                # AlphaZero용 Dual-Attention Transformer
├── mcts_2d.py                  # AlphaZero MCTS (Dirichlet 노이즈, temperature 스케줄링)
├── train_alphazero.py          # AlphaZero 학습 루프 (Ray 병렬, 자기 대국)
├── train.py                    # 1D DQN 학습
├── train_parallel.py           # 1D DQN 병렬 학습 (Ray)
├── evaluate.py                 # 1D 솔버 평가 (레거시)
├── solve_nonogram.py           # 1D Q-network 기반 NxN 풀이기 (레거시)
├── inference.py                # 1D 추론 시각화 (레거시)
├── utils.py                    # 공통 유틸리티 (hint 인코딩, feasibility 등)
├── scripts/                    # 모듈화된 최신 실행 스크립트
│   ├── train.py                # 최신 모듈형 1D/2D 학습
│   ├── evaluate.py             # 최신 모듈형 솔버 평가
│   ├── evaluate_1d.py          # 최신 모듈형 1D Q-net 평가
│   ├── inference.py            # 최신 모듈형 1D 추론 시각화
│   └── train_parallel.py       # Ray 기반 대규모 병렬/분산 학습 [NEW]
├── runs/                       # 실험 결과 (실험별 하위 폴더)
└── README.md
```

---

## 설치 및 환경 구성

이 프로젝트는 **uv**를 사용하여 가상환경 및 종속성을 매우 빠르고 안전하게 관리합니다.

```bash
# uv로 가상환경 구성 및 패키지 동기화
uv sync

# GPU(CUDA/MPS) 지원은 자동 감지됩니다.
# 디바이스 우선순위: CUDA > MPS (Apple Silicon) > CPU
```

---

## 주요 기능

### 1. AC-3 Constraint Propagation + Q-value DFS — 2D 보드 풀이 (hint-satisfied 100%)
- **AC-3 스타일 제약 전파(Constraint Propagation)**: 각 unknown 셀에 대해 +1과 -1 양방향의 행/열 feasibility를 검사하여, 유일하게 가능한 값이 있으면 논리적으로 즉시 확정. 이 과정이 연쇄적으로 전파되어 10x10 보드에서 평균 86~100셀을 순수 논리로 완성.
- **Q-value 가이드 DFS Guessing**: 제약 전파로 확정할 수 없는 셀은 1D Q-network의 Q-value가 가장 높은 셀부터 우선 시도하는 백트래킹 DFS로 해결. 잘못된 가정은 백트래킹으로 자동 복구.
- **Lazy Feasibility Check**: DFS 분기 후보 선정 시 배치 Q-value 연산 후 상위 후보부터 순차 검증하여 최초 통과 후보를 즉시 채택.
- **Best Feasible Board 복원 전략**: 탐색 실패/중단 시 DFS 탐색 과정 중 가장 많은 셀을 채웠던 유효한(feasible) 보드 상태를 복원하여 최종 반환.

### 2. AlphaZero 2D — MCTS + Dual-Attention Transformer
- **Dual-Attention Transformer**: Axial Attention (행/열 독립 추론) + Global Attention (전체 보드 패턴)을 결합하여 2D 힌트와 셀 상태 파악.
- **AlphaZero MCTS**: 자기 대국 + 신경망 가치/정책 평가 + perfect simulator(feasibility 가지치기).
- **탐색 강화**: Dirichlet 노이즈(루트 탐색), temperature 스케줄링(초반 탐색 → 후반 greedy) 적용.
- **Ray 병렬**: 다수 워커가 자기 대국 수행, learner가 비동기 학습 진행.

### 3. LQL (Long-Horizon Q-Learning) 기법 탑재
강화학습 에이전트가 노노그램 퍼즐의 **논리적 비가역성**을 스스로 깨닫도록 **LQL (arXiv:2605.05812)** 방법론을 통합했습니다.
- **단조성 제약**: 노노그램은 칸을 칠해나갈수록 성공 가능성(Value)이 감소하거나 유지되어야 합니다 ($V(s_t) \ge V(s_{t+n})$).
- **Hinge Loss 기반 손실**: 가치가 비정상적으로 역전될 경우 페널티를 부과하여 가치 함수 전반의 논리적 일관성을 확보합니다.
- 적용 대상: DQN, Double DQN, PPO, AlphaZero 2D 에이전트 전반 적용.

### 4. 역방향 커리큘럼 학습 (Reverse Curriculum Learning) — 1D 줄 및 2D 보드 전체 지원
초기 상태 탐색의 희소 보상 문제를 방지하고 점진적으로 학습을 심화하도록 설계되었습니다.
- **정답 기반 초기 채우기**: 에피소드 초기 단계에서 1D 줄(또는 2D 보드)의 일부 칸을 정답 값으로 채워진 상태로 시작합니다.
- **다양한 동적 난이도 조절 유형 지원**:
  * **Linear Decay (`type: "linear"`)**: 에피소드 진행도에 따라 난이도(비공개 칸 수)를 선형적으로 증가시킵니다.
  * **Cosine Annealing (`type: "cosine"`)**: 코사인 그래프 형상을 따라 초반에는 천천히 비워지다 중간에 빠르게 난이도를 올리고 후반에 완만하게 빈 보드로 수렴시킵니다.
  * **Performance-based Adaptive Decay (`type: "adaptive"`)**: 에이전트의 실시간 성공 성과를 분석하여 **자동으로 난이도를 가감**합니다. 최근 슬라이딩 성공률이 상한 임계치(예: `0.8`)를 넘으면 빈 칸을 늘려 난이도를 올리고, 하한 임계치(예: `0.2`) 미만으로 떨어지면 난이도를 낮춰 정답 지원을 늘리는(Backtracking) 유연한 적응형 학습을 실현합니다.
- **자동 한 칸 시작 ("auto")**: `start_ratio: "auto"` 설정을 통해 크기에 맞추어 **정확히 단 1칸만 비우도록** (공개 비율 $\frac{Size-1}{Size}$) 초기 비율을 계산하여 수렴을 극대화합니다.


### 5. Action Chunking (Q-chunking) 기법 탑재 (NeurIPS 2025)
노노그램의 **장기 목표(Long-Horizon)**와 **희소 보상(Sparse-Reward)** 문제를 완화하기 위해 한 번의 추론으로 여러 스텝의 행동을 결정하고 실행하는 Action Chunking 기법을 통합하였습니다.
- **일관된 다중 탐색 (Temporal Coherence)**: 매 스텝 Q-value를 평가하는 대신, 한 번의 순방향 전파(forward pass)로 확신도가 높은 여러 셀의 플레이를 하나의 척(Chunk)으로 묶어 실행(Open-loop)합니다.
- **가치 백업 건너뛰기 (Skipped TD-backup)**: 척을 실행하고 난 뒤의 최종 상태($s_{t+H}$)를 기준으로 척 내부 액션들의 가치 함수를 n-step TD 목표값으로 업데이트하여 보상 정보를 빠르게 역방향 전파시킵니다.
- 적용 대상: DQN, Double DQN (MLP & Transformer) 에이전트에 적용 가능.

### 6. 초심층 대비 강화학습 (Contrastive RL) 기법 탑재 (arXiv:2503.14858)
딥 네트워크 스케일링에서 겪는 학습 불안정성과 기울기 폭발/소실 문제를 혁신적으로 극복하기 위해 대비 학습(Contrastive RL) 방법론을 노노그램에 완벽하게 내장하였습니다. 본 구현은 **2D Board 환경**과 **1D Line 환경**을 모두 공식 지원합니다.
- **초안정성 Deep ResNet 아키텍처**: Pre-LayerNorm, GroupNorm(소규모 배치 친화적), GELU 및 **LayerScale** ($10^{-3}$ 스케일 팩터)을 통합 설계하여 수십~수백 층 레이어에서도 수렴합니다.
- **InfoNCE 손실 함수**: 행동이 취해진 로컬 상태 표현 $\Phi(s, a)$와 도달하려는 글로벌 목표 표현 $\Psi(g)$의 매칭 관계를 배치 내 분류 문제(Classification)로 변환해 불확정성을 완전히 제어합니다.
- **방식 1 (HER 모드)**: Hindsight Experience Replay 방식을 결합하여, 에피소드의 미래 상태 $s_{t+k}$를 목표 $g$로 동적 추출하고 학습합니다.
- **방식 2 (Clue 모드)**: 주 단서(row/col clues)를 목표 $g$로 설정하여, 목표 단서를 만족하는 행동 쌍을 집중 매칭하는 방식으로 학습 효율을 향상합니다.
- **적용 대상**:
  - **2D Board**: `configs/crl_clue_10x10.yaml`, `configs/crl_her_10x10.yaml`로 훈련 가능.
  - **1D Line**: `configs/crl_line_clue_10x10.yaml`, `configs/crl_line_her_10x10.yaml`로 훈련 가능.

### 7. Symmetry-Aware GFlowNets (Trajectory Balance) 기법 탑재 (ICML 2025)
노노그램의 엄격한 논리적 제약과 높은 탐색 복잡도를 완화하기 위해, MCTS의 무거운 롤아웃 없이 상태 공간을 효율적으로 탐색하고 다양한 해를 샘플링하는 GFlowNet 구조를 도입하였습니다.
- **GFlowNet MDP 재정의**: 빈 보드에서 시작하여 행(Row) 단위로 오토레그레시브하게 보드를 결정해 나가는 전이 MDP 설계.
- **Symmetry-Aware Bipartite GNN**: 수평/수직 반전 및 전치(Transpose)에 대해 논리적 제약이 완전히 보존되는 $D_4$ 대칭군 특성을 GNN 파라미터 공유 및 레이어 설계에 주입하고, 정책 수준에서 $V_4$ 대칭 변환 로짓의 평균 연산을 수행하여 완벽한 동변성(Equivariance) 확보.
- **Trajectory Balance (TB) 목적 함수**: 단일 해 수렴(Mode Collapse) 문제를 근본적으로 예방하고 다중 정답을 고르게 샘플링할 수 있도록 궤적 균형(TB) 손실 함수를 최적화 타겟으로 설정하고 분배 함수 $Z$를 학습.
- **동적 보상 온도 어닐링 (Reward Temperature Annealing)**: 초반 탐색을 촉진하고 국소 최적(Local Optima)에 갇히는 현상을 예방하기 위해, 보상 지수 스케일 온도(`reward_temp`)를 에피소드 진행에 따라 점진적으로 증가시키는 스케줄러(80% 시점에서 최대 온도 도달)를 적용합니다.
- **완벽 정답 보상 임계치 기반 성공 판정**: 조기 종료 페널티(`epsilon`)를 피했을 뿐인 오답 궤적을 성공으로 기록하던 오류(Fake Success)를 방지하기 위해, 완벽한 정답 보상 임계치($e^{\text{temp} \times 2.0} \times 0.99$)에 실제로 도달했는지를 엄격히 대조하여 성공을 판정합니다.
- **적용 대상**: `configs/gflownet_10x10.yaml`로 훈련 가능.

### 8. 1D DFS & 2D RL 계층적 하이브리드 솔버 (Hierarchical Hybrid Solver) 및 Dynamic Pruning 탑재 [NEW]
노노그램의 성능과 정답률을 기하학적으로 끌어올리기 위해, 1D의 강력한 논리적 모순 제거력(DFS 백트래킹)과 2D의 글로벌 공간 맥락 지각력(2D Board RL 에이전트)을 유기적으로 융합하였습니다.
- **2D Board 힌트 유실 해소 (GRU/Transformer 힌트 임베딩)**: 2D 보드 모델(`board_cnn`, `board_transformer`)이 단순 힌트 합산 스칼라 브로드캐스트 대신, 힌트 시퀀스 원형을 GRU 및 Attention 구조로 직접 흡수·융합하도록 개선하여 정보 병목을 해소하였습니다.
- **동적 Action Masking (Feasibility Pruning)**: 각 칸에 가상 착수를 시도하여 가치 조건이 위배되는 불가능한 액션(Infeasible action)들을 $O(N)$으로 정밀 검사해 강제로 배제($-\infty$ 마스킹)합니다.
- **계층적 하이브리드 솔버 (`HybridSolver`)**: 1D Line Solver of 강력한 줄 단위 백트래킹 DFS를 골격으로 유지하되, 추론이 정체되는 교착(Deadlock) 지점에서 2D Board Q-network를 호출하여 전체 보드 가치가 가장 높은 최선의 한 수를 추천받아 탐색 실패를 타개합니다.

### 9. 2D Board RL (Cross-Attention) 및 순수 RL 기반 2D DFS 백트래킹 솔버 [NEW]
AC-3 알고리즘의 고정된 논리적 추론 없이 순수하게 강화학습만으로 2D 노노그램의 교차 제약을 풀어내기 위해, Row/Column Axial-Attention 아키텍처와 DFS 탐색 메커니즘을 통합했습니다.
- **2D Board Row-Column Cross-Attention Q-network**: 행(Row)과 열(Column) 방향의 Axial Self-Attention 및 글로벌 피드포워드 레이어를 통해, 2D 노노그램의 행-열 교차 제약 조건을 모델 내부 파라미터로 직접 추론하고 학습하도록 설계되었습니다.
- **Feasibility-Guarded Dense Reward (`feasibility_dense_2d`)**: 매 스텝 에이전트의 착수마다 2D 행/열 Feasibility 유지를 확인하여 타당한 결정이면 +0.01 보상을 지급하고, 제약 조건을 위반하는 오답을 두면 즉시 에피소드를 조기 종료(early stop, -1.0)시켜 학습 수렴을 비약적으로 가속합니다.
- **Q-value 가이드 DFS 백트래킹 Board Solver**: 2D Board Q-network의 출력값(Q-value)이 높은 착수 후보 순서대로 DFS 백트래킹 탐색을 진행하여, 순수 강화학습 에이전트의 판단에 의존해 2D 보드를 완성시킵니다.

### 10. 1D RL 모델 기반 2D AlphaZero 사전 학습 (1D-Guided BC Pre-training) [NEW]
- **1D 지식 전이 (Knowledge Distillation)**: 이미 뛰어난 성능으로 훈련된 1D Q-network 모델(`LineSolver`)을 활용하여, 무작위 2D 보드 퍼즐 중 풀 수 있는 성공 궤적(Trajectory)을 추출합니다.
- **1D-Guided BC Pre-training**: 2D AlphaZero 에이전트가 본격적인 자가 학습(MCTS Self-play) 전에 이 1D 논리적 궤적 데이터를 모방 학습(Behavior Cloning)하여 가중치를 초기화합니다. 이로써 2D 탐색 공간의 극심한 콜드 스타트 문제를 타개하고 수렴 속도를 혁신적으로 끌어올립니다.

---

## 핵심 코드

### AC-3 Constraint Propagation + DFS 아키텍처 (`nonogram/solvers/line_solver.py`)

```python
# 1. AC-3 스타일 제약 전파: 유일한 feasible 값을 가진 셀을 논리적으로 즉시 확정
def _propagate_constraints(self, board, N) -> bool:
    to_check = {(r, c) for r in range(N) for c in range(N) if board[r, c] == 0}
    while to_check:
        next_check = set()
        for (r, c) in list(to_check):
            if board[r, c] != 0:
                continue
            # 양방향 feasibility 검사 (in-place 후 원복)
            board[r, c] = 1
            feas_pos = is_feasible(board[r, :], hints_row[r], N) and \
                       is_feasible(board[:, c], hints_col[c], N)
            board[r, c] = -1
            feas_neg = is_feasible(board[r, :], hints_row[r], N) and \
                       is_feasible(board[:, c], hints_col[c], N)
            board[r, c] = 0  # 원복
            if feas_pos and not feas_neg:     # +1만 가능 → 확정
                board[r, c] = 1
            elif feas_neg and not feas_pos:   # -1만 가능 → 확정
                board[r, c] = -1
            elif not feas_pos and not feas_neg:
                return False  # 모순 발견
        to_check = next_check
    return True

# 2. DFS Guessing: 배치 Q-value로 최적 분기 셀 선별
def _get_best_guess_candidate(self, board, row_hints_padded, col_hints_padded, N):
    # 모든 행의 Q-value를 한 번에 배치 연산 (N개 → 1개 forward pass)
    states_t = torch.tensor(np.stack([build_state(hint[r], board[r,:]) for r in range(N)]))
    q_all = self.q_net(states_t).cpu().numpy()  # (N, 2*N)
    # Q-value 내림차순 정렬 후 Lazy Feasibility Check
    candidates = sorted([(q_all[r,c], r, c, 1) for r,c if board[r,c]==0], reverse=True)
    for q_val, r, c, f in candidates:
        if is_feasible(row) and is_feasible(col): return (r, c, f, q_val)

# 3. search 함수 핵심 흐름
def search(board, force_count):
    local_board = board.copy()
    if not _propagate_constraints(local_board, N):  # AC-3 제약 전파
        return None  # 모순 → 백트래킹
    if complete: return check(local_board)           # 완성 → 검증
    cell = _get_best_guess_candidate(local_board)    # DFS: Q-value 최선 분기
    res = search(board_with_first_value)             # 1차 시도
    if res: return res
    return search(board_with_second_value)            # 백트래킹
```

### AlphaZero 2D 아키텍처 (`models_2d.py`)

```python
class AlphaZeroNonogramNet(nn.Module):
    """
    Dual-Attention Transformer + AlphaZero Policy/Value Head
    
    입력: board (B, N, N) + row_hints (B, N, k) + col_hints (B, N, k)
    
    처리 흐름:
    1. Cell Embedding (0=미결정, 1=칠함, 2=빈칸)
    2. Hint RNN (GRU로 가변 길이 hint를 고정 벡터로 압축)
    3. Feature Fusion (cell + row_hint + col_hint → 3*dim → dim)
    4. Dual-Attention Transformer Blocks:
       - AxialAttention: 같은 행/열 셀 간 attention (행-열 제약 모델링)
       - GlobalAttention: 전체 N*N 셀 간 attention (보드 패턴 파악)
       - FFN + Residual + LayerNorm
    5. Policy Head → (B, 2, N, N) logits (칠함/빈칸 per cell)
    6. Value Head → (B, 1) ∈ [-1, 1] (현재 상태 승률 예측)
    """
```

### MCTS 탐색 및 최적화 (`mcts_2d.py`)

```python
class AlphaZeroMCTS:
    """
    핵심 기능:
    1. Perfect Simulator: 각 행/열의 feasibility를 검사하여 infeasible 분기 즉시 가지치기 (10x10 보드는 해당 행/열만 검사하도록 O(N) 최적화 적용).
    2. Dirichlet 노이즈: 루트 노드 prior에 Dir(α) 혼합으로 탐색 다양성 보장.
    3. Temperature 스케줄링: 초반 탐색적(τ=1) → 후반 greedy(τ→0).
    4. Masked Softmax: 빈 셀에 대해서만 확률 계산 (이미 결정된 셀 제외).
    """
```

### Action Chunking 핵심 로직 (`nonogram/agents/dqn.py`, `scripts/train.py`)

```python
# 1. 척(Chunk) 탐색 및 비충돌 액션 선별 (nonogram/agents/dqn.py)
def select_action_chunk(self, state, mask, explore=True):
    # 탐색(ε-greedy) 시 중복되지 않는 고유한 셀 액션들을 랜덤하게 선별
    # Greedy 평가 시: Q-value 계산 후 각 셀마다 Paint/X 중 큰 Q를 선별하고,
    # Q-value 기준 내림차순 정렬하여 상위 chunk_size(H) 개의 액션을 반환
    ...

# 2. 오픈루프 실행 및 스킵된 TD 백업 (scripts/train.py)
# 척에 들어있는 액션들을 순차적으로 실행하되, 척 적용 전의 state_before_chunk와
# 적용 완료 후의 next_state, 그리고 척 누적 보상을 한데 묶어 Replay Buffer에 각 액션별로 저장합니다.
# 이를 통해 TD 학습 시 N-step 도약 백업 효과(Skipped TD-backup)를 누릴 수 있습니다.

### 1D DFS & 2D RL 계층적 하이브리드 솔버 (`nonogram/solvers/hybrid_solver.py`)

```python
class HybridSolver(LineSolver):
    """1D DFS 논리 추론 + 2D Board RL 글로벌 공간 추론 하이브리드 솔버."""
    def _get_best_guess_candidate(self, board, row_hints_padded, col_hints_padded, N):
        if self.q_net_2d is None:
            # 2D 모델이 준비되지 않은 경우 1D DFS candidate 탐색 방식으로 폴백
            return super()._get_best_guess_candidate(board, row_hints_padded, col_hints_padded, N)

        # 1. 2D 보드 상태 빌드
        state_2d = self._build_board_state_2d(board, row_hints_clean, col_hints_clean, N)
        state_t = torch.tensor(state_2d, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 2. 2D 힌트 배치 패딩 생성 및 2D Q-network 포워딩
        padded_rh = torch.tensor(pad_hints_batch([row_hints_clean], k, N), dtype=torch.long, device=self.device)
        padded_ch = torch.tensor(pad_hints_batch([col_hints_clean], k, N), dtype=torch.long, device=self.device)
        q_values = self.q_net_2d(state_t, padded_rh, padded_ch).squeeze(0).cpu().numpy()

        # 3. 미확정 셀 중 Q-value가 최대인 글로벌 액션 선출 및 DFS 이진 분기 유도
        best_act = int(np.argmax(q_values))
        ...
        return "row", r, c, f, -f, q_val
```

### Row-Column Cross-Attention & DFS 백트래킹 Board Solver (`nonogram/models/board_cross_attn.py`, `nonogram/solvers/board_solver.py`)

```python
# 1. Row/Col 축방향 Axial-Attention 블록
class RowColAttentionBlock(nn.Module):
    def forward(self, x):
        B, D, N, _ = x.shape
        # Row Attention
        r_in = x.permute(0, 2, 3, 1).reshape(B * N, N, D)
        r_out, _ = self.row_attn(r_in, r_in, r_in)
        r_out = self.norm1(r_in + r_out)
        x = r_out.view(B, N, N, D).permute(0, 3, 1, 2)
        # Col Attention
        c_in = x.permute(0, 3, 2, 1).reshape(B * N, N, D)
        c_out, _ = self.col_attn(c_in, c_in, c_in)
        c_out = self.norm2(c_in + c_out)
        x = c_out.view(B, N, N, D).permute(0, 3, 2, 1)
        # Feed-Forward Network
        flat = x.permute(0, 2, 3, 1).reshape(B, N * N, D)
        x = self.norm3(flat + self.ffn(flat)).view(B, N, N, D).permute(0, 3, 1, 2)
        return x

# 2. Q-value 유도 DFS 백트래킹 탐색 (board_solver.py)
def search(board, force_count):
    # 완료 검증
    if (board != 0).all():
        return board if check_all_feasible(board) else None
    
    # Q-values 예측
    state_t = torch.tensor(get_state(board)).unsqueeze(0)
    q_values = q_net(state_t, row_hints, col_hints).squeeze(0).numpy()
    
    # Q-value 순위화 및 Feasibility 사전 검증
    candidates = get_sorted_feasible_candidates(board, q_values)
    if not candidates: return None
    
    best_cand = candidates[0]
    # 백트래킹 이진 탐색 트리 분기
    res = search(board_with_best_cand)
    if res is not None: return res
    
    return search(board_with_opposite_cand)
```
```,StartLine:281,TargetContent:
```

---

## 주요 명령어

### 1D/2D 강화학습 학습 (scripts/train.py)

모듈화된 최신 학습 스크립트를 사용하여 지정된 완결성 있는 YAML 설정으로 안정적인 강화학습을 수행합니다.

```bash
# 5x5 표준 DQN 학습
uv run python scripts/train.py --config configs/dqn_5x5.yaml

# 10x10 Double DQN + Dueling 학습 (LQL 탑재, 1D 줄 단위)
uv run python scripts/train.py --config configs/double_dqn_10x10.yaml

# 10x10 1D 극대화 모델 학습 (dense reward + curriculum + towards 혼합)
uv run python scripts/train.py --config configs/dqn_10x10_super_1d.yaml

# 10x10 DQN + Board CNN 학습 (LQL 탑재, 2D 보드 단위)
uv run python scripts/train.py --config configs/board_dqn_10x10.yaml

# 10x10 DQN + Board Bipartite GNN 학습 (LQL 탑재, 2D 이분 그래프 구조)
uv run python scripts/train.py --config configs/board_gnn_dqn_10x10.yaml

# 10x10 DQN + Board Axial Attention Transformer 학습 (LQL + PER 탑재, 2D 트랜스포머 구조)
uv run python scripts/train.py --config configs/board_transformer_dqn_10x10.yaml

# 10x10 DQN + Action Chunking (MLP 베이스라인) 학습
uv run python scripts/train.py --config configs/dqn_chunking_mlp_10x10.yaml

# 10x10 DQN + Action Chunking (Axial Attention Transformer) 학습
uv run python scripts/train.py --config configs/dqn_chunking_transformer_10x10.yaml

# 10x10 PPO 학습 (LQL 탑재, 1D 줄 단위)
uv run python scripts/train.py --config configs/ppo_10x10.yaml

# 10x10 PPO + Board CNN 학습 (LQL 탑재, 2D 보드 단위)
uv run python scripts/train.py --config configs/board_ppo_10x10.yaml

# 10x10 PPO + Board Bipartite GNN 학습 (LQL 탑재, 2D 이분 그래프 구조)
uv run python scripts/train.py --config configs/board_gnn_ppo_10x10.yaml

# 10x10 2D Board Clue-Board 매칭 대비 학습 (Contrastive RL)
uv run python scripts/train.py --config configs/crl_clue_10x10.yaml

# 10x10 2D Board HER 기반 상태 도달 대비 학습 (Contrastive RL)
uv run python scripts/train.py --config configs/crl_her_10x10.yaml

# 10x10 1D Line Clue 매칭 대비 학습 (Contrastive RL)
uv run python scripts/train.py --config configs/crl_line_clue_10x10.yaml

# 10x10 1D Line HER 기반 대비 학습 (Contrastive RL)
uv run python scripts/train.py --config configs/crl_line_her_10x10.yaml

# 10x10 Symmetry-Aware GFlowNet (Trajectory Balance) 학습
uv run python scripts/train.py --config configs/gflownet_10x10.yaml
```

### Ray 기반 대규모 병렬 강화학습 (scripts/train_parallel.py) [NEW]

여러 CPU Rollout Worker를 비동기적으로 실행하여 에피소드를 병렬로 빠르게 수집하고, 메인 Learner 프로세스(GPU 지원)에서 가중치 업데이트 및 브로드캐스트를 수행하여 10x10 크기의 고차원 보드 학습 수렴 성능을 획기적으로 향상시킵니다.

```bash
# 10x10 2D Board DQN 모델을 4개 Ray CPU 워커로 대규모 가속 병렬 학습
uv run python scripts/train_parallel.py --config configs/board_dqn_10x10.yaml --num_workers 4

# 10x10 Double DQN Dueling 모델을 8개 Ray CPU 워커로 병렬 학습
uv run python scripts/train_parallel.py --config configs/double_dqn_10x10.yaml --num_workers 8
```

### AlphaZero 2D 학습 (train_alphazero.py)

```bash
# 10x10 AlphaZero MCTS + Transformer 학습
uv run train_alphazero.py --config configs/alphazero_2d.yaml

# 기존 체크포인트에서 재개
uv run train_alphazero.py --config configs/alphazero_2d.yaml --resume

# [NEW] 1D RL 모델을 기반으로 2D AlphaZero 사전 학습 후 구동
uv run python train_alphazero.py --config configs/alphazero_2d_pretrain_1d.yaml
```

### 평가 및 추론 시각화 (scripts/)

```bash
# 2D 보드 솔버 (LineSolver) 다수 퍼즐 평가 (기본: 최종 요약 결과만 간결하게 출력)
uv run python scripts/evaluate.py --config configs/double_dqn_10x10.yaml --num_puzzles 200 --load_path ./runs/double_dqn_10x10_double_dqn_dueling_N10/checkpoints/latest.pt

# 2D 보드 솔버 상세 평가 (상세 출력 모드: Solver의 풀이 과정 및 중간 진행 상황 출력)
uv run python scripts/evaluate.py --config configs/double_dqn_10x10.yaml --num_puzzles 200 --load_path ./runs/double_dqn_10x10_double_dqn_dueling_N10/checkpoints/latest.pt --verbose

# 1D 줄(Line) 환경에서 1D Q-네트워크 자체의 성공률 및 성능 평가
uv run python scripts/evaluate_1d.py --config configs/double_dqn_10x10.yaml --load_path ./runs/double_dqn_10x10_double_dqn_dueling_N10/checkpoints/best.pt --num_lines 1000

# 1D 한 줄 풀이 과정의 단계별 Q-값 및 마스킹 추론 시각화
uv run python scripts/inference.py --config configs/double_dqn_10x10.yaml --hint 1 2 3 --load_path ./runs/double_dqn_10x10_double_dqn_dueling_N10/checkpoints/latest.pt
```

---

## DB 스키마 및 영속성 구조 (Data Persistence)

본 프로젝트는 외부 서버 DBMS 연동 대신 **YAML 구성 파일**, **PyTorch 가중치/메타데이터 파일(`.pt`)**, **실험 로그 디렉터리** 구조를 데이터 영속성 계층으로 활용하는 경량 파일 저장 구조를 따릅니다. 사용 규칙에 의거하여 논리적 영속 구조를 테이블(스키마) 형태로 표현하여 명시합니다.

### 1. 설정 테이블 (`configs/*.yaml`)
| 필드명 | 데이터 타입 | 제약 조건 | 설명 |
| :--- | :--- | :--- | :--- |
| `env.N` | Integer | NOT NULL, $\ge 5$ | 노노그램 보드 크기 ($N \times N$) |
| `env.reward_type` | String | sparse \| feasibility \| step_correct | 에이전트의 보상 형태 |
| `env.use_feasible_mask`| Boolean | NOT NULL | dynamic feasibility masking 활성화 여부 |
| `model.type` | String | mlp \| dueling \| cnn \| board_cnn \| deep_crl \| deep_line_crl \| symmetry_gnn | 가치/정책/대비/GFlowNet 네트워크 구조 |
| `model.num_layers` | Integer | $\ge 1$ | 초심층 대비 학습 모델 레이어 수 (ResNet 1D/2D 블록 수) |
| `model.dropout` | Float | $0.0 \sim 1.0$ | 드롭아웃 확률 |
| `agent.type` | String | dqn \| double_dqn \| ppo \| alphazero_2d \| crl \| gflownet | 강화학습 알고리즘 종류 |
| `agent.crl_mode` | String | clue \| her | 대비 학습 매칭 모드 (Clue 매칭 또는 HER 미래 상태 매칭) |
| `agent.crl_temperature`| Float | $\ge 0.0$ | InfoNCE 분류 온도 하이퍼파라미터 (기본 0.1) |
| `agent.lql_weight` | Float | $\ge 0.0$ | LQL 단조성 제약 손실 비중 |
| `agent.lql_n_step` | Integer | $\ge 1$ | LQL 및 HER 상태 간격 단계 |
| `agent.action_chunking` | Boolean | NOT NULL | Action Chunking 활성화 여부 |
| `agent.chunk_size` | Integer | $\ge 1$ | 한 번에 실행할 액션 척의 크기 |
| `training.reward_shaping` | Boolean | NOT NULL | A: 부분 달성도 기반 연속 보상 사용 여부 |
| `training.pretrain_bc` | Boolean | NOT NULL | B: MCTS 학습 시작 전 Behavior Cloning 사전 학습 여부 |
| `training.pretrain_episodes`| Integer | NOT NULL | BC 학습용 궤적 에피소드 수 |
| `training.pretrain_lr` | Float | NOT NULL | BC 학습용 learning rate |
| `training.pretrain_1d_model`| String | Option | [NEW] 1D 가이드 BC 학습용 1D Q-net 체크포인트 경로 (예: `qnet_N10.pt`) |
| `training.pretrain_1d_episodes`| Integer | Option | [NEW] 1D 가이드 BC용 궤적 에피소드 수 (기본 1000) |
| `training.pretrain_1d_lr` | Float | Option | [NEW] 1D 가이드 BC용 learning rate (기본 0.001) |
| `training.pretrain_1d_hidden_dim`| Integer | Option | [NEW] 로드할 1D 모델의 hidden_dim (기본 128) |
| `training.curriculum.enabled`| Boolean | NOT NULL | 역방향 커리큘럼 학습의 활성화 여부 (2D 보드 전용) |
| `training.curriculum.type` | String | linear \| cosine \| adaptive | 커리큘럼 감쇠 유형 (선형 / 코사인 어닐링 / 성과 기반 적응형) |
| `training.curriculum.start_ratio`| Float \| String | NOT NULL | 초기 정답 공개 비율 (또는 "auto" 설정 시 2D 보드 내 단 1칸만 비우도록 설정) |
| `training.curriculum.end_ratio`| Float | NOT NULL | 최종 정답 공개 비율 |
| `training.curriculum.decay_episodes`| Integer | NOT NULL | 공개 비율 감쇠 에피소드 수 (linear/cosine 용) |
| `training.curriculum.adaptive_window`| Integer | NOT NULL | 성과(성공률) 판단용 슬라이딩 윈도우 크기 (adaptive 용) |
| `training.curriculum.adaptive_threshold`| Float | NOT NULL | 난이도를 높이기(비공개칸 증가) 위한 목표 성공률 임계치 (adaptive 용, 기본 0.8) |
| `training.curriculum.adaptive_backtrack_threshold`| Float | NOT NULL | 난이도를 낮추기(복습용 정답 지원) 위한 최저 성공률 임계치 (adaptive 용, 기본 0.2) |
| `training.curriculum.adaptive_step`| Float | NOT NULL | 성과 판정 시 조정할 reveal_ratio 증감 보폭 (adaptive 용, 기본 0.02) |
| `solver.min_guess_q` | Float | NOT NULL | 교착 상태 DFS Guessing 진입을 위한 최저 Q-value 신뢰도 임계값 (기본 0.05) |
| `solver.max_backtracks` | Integer | NOT NULL | 무한 백트래킹 탐색 헬(Hell) 방지를 위한 최대 백트랙 시도 허용 횟수 (기본 1000) |
| `solver.board_checkpoint`| String | NOT NULL | Hybrid Solver에서 글로벌 가치 평가로 활용할 2D 보드 모델의 체크포인트 파일 경로 |

### 2. 체크포인트 테이블 (`runs/{exp_name}/checkpoints/*.pt`)
| 필드명 | 데이터 타입 | 제약 조건 | 설명 |
| :--- | :--- | :--- | :--- |
| `model_state_dict` | Dict (PyTorch) | Neural Network Weights | 학습된 정책/가치 신경망의 모든 가중치 텐서 |
| `optimizer_state_dict` | Dict (PyTorch) | Optimizer States | Adam 등 옵티마이저의 모멘텀 및 학습 상태 정보 |
| `episode` / `step` | Integer | NOT NULL, $\ge 0$ | 저장 시점의 학습 진행도 (에피소드 수 또는 스텝 수) |
| `config` | Dict (JSON-like) | NOT NULL | 학습 시점에 사용된 완결된 YAML 파라미터 백업 |
| `best_success_rate` | Float | $0.0 \sim 1.0$ | 평가 단계에서 기록된 최고 퍼즐 해결 성공률 |

---

## 보상 구조 (`env.reward_type`)

| 타입 | 보상 시점 | ground truth 사용 | 설명 |
|---|---|---|---|
| `sparse` | terminal만 | ✕ (hint만) | hint 일치 +1, 불일치 -1. towards 필수. |
| `feasibility` | 매 step | ✕ (hint만) | feasible completion 수 변화 기반. 순수 RL 가능. |
| `step_correct` | 매 step | ✓ (정답 비교) | 매 step 정답 비교 ±1/N. 독립 분류화 위험. |
| `feasibility_dense_2d` | 매 step | ✕ (hint만) | 매 step feasibility 유지 시 +0.01, 위반 시 즉시 early stop 및 -1.0. 2D 보드 학습용. |

---

## 실험 결과

| 실험 | Config | 결과 |
|------|--------|------|
| 1D DQN 5x5 | `dqn_5x5.yaml` | hint_satisfied **98%**, exact_match **90%** |
| 1D DQN 10x10 | `dqn_10x10.yaml` | 학습 수렴 확인 |
| Double DQN 10x10 | `double_dqn_10x10.yaml` | LQL 적용 및 체크포인트 최적 보존 완료 |
| Double DQN 1D 극대화 10x10 | `dqn_10x10_super_1d.yaml` | dense step_correct 보상과 커리큘럼 및 expert demonstration 비율 혼합 적용 |
| **AC-3 CP + DFS 2D Solver** [BEST] | `dqn_10x10_super_1d.yaml` | 10x10 퍼즐 **hint-satisfied 100% (100/100)**, exact-match **68%**, cell agreement **97.8%**, 평균 풀이 시간 **27.7ms** |
| 1D Line Solver (백트래킹 DFS) | `configs/double_dqn_10x10.yaml` | `qnet_N10.pt` 모델 사용, 10x10 퍼즐 완벽 해결 성공률 **30% (3/10)**, cell-level agreement **62.5%** 달성 |
| 1D Line Solver (점진적 최적화) | `configs/dqn_10x10_super_1d.yaml` | 다중 동시 결정에 의한 Feasibility 깨짐 문제 해결 후, 10x10 퍼즐 성공률 **8.0% (4/50)**, cell-level agreement **54.28%** |
| Contrastive RL 2D 10x10 | `crl_clue_10x10.yaml` & `crl_her_10x10.yaml` | 초심층 ResNet2D + InfoNCE 학습 안정성 검증 완료 |
| Contrastive RL 1D 10x10 | `crl_line_clue_10x10.yaml` & `crl_line_her_10x10.yaml` | 초심층 ResNet1D + InfoNCE 학습 안정성 검증 완료 |
| AlphaZero 2D 10x10 | `alphazero_2d.yaml` | Axial-Attention + LQL 튜닝 및 학습 재개 |
| GFlowNet 2D 10x10 | `gflownet_10x10.yaml` | Symmetry-Aware Bipartite GNN + Trajectory Balance 학습 파이프라인 검증 완료 |
| 1D DFS & 2D RL 하이브리드 솔버 | `configs/double_dqn_10x10.yaml` + `--solver.type hybrid` | 2D 보드 모델(`board_cnn`) 연동 성공 및 교착 상태 돌파 추론 정상 작동 확인 |
| 1D RL (CP 미사용) 2D 풀이 | `configs/dqn_10x10_super_1d.yaml` + `--solver.method rl` | 100개 10x10 퍼즐 성공률 **1% (1/100)** -> CP의 논리 추론 배제 시 1D 모델의 2D 전역 교착 상태 해결 불가 입증 |

---

## 주의사항

- **1D 모델의 N == 보드의 줄 길이**가 반드시 같아야 합니다.
- **N이 커질수록 학습이 기하급수적으로 어려워집니다.** action space: `2N`, state space: ~`3^N`.
- **2D Board 환경**은 action space가 `2*N*N`으로 매우 크므로, 충분한 학습 에피소가 필요합니다.
- **AlphaZero 워커**는 GPU 경합 방지를 위해 **항상 CPU에서 추론**합니다. Learner만 GPU 사용.
- **Device 자동 감지**: CUDA > MPS(Apple Silicon) > CPU 순으로 자동 선택됩니다.