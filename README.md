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
│   ├── ppo_10x10.yaml          # 10x10 PPO
│   ├── board_dqn_10x10.yaml    # 10x10 2D Board DQN
│   ├── dqn_chunking_mlp_10x10.yaml  # 10x10 DQN + Action Chunking (MLP)
│   ├── dqn_chunking_transformer_10x10.yaml # 10x10 DQN + Action Chunking (Transformer)
│   └── alphazero_2d.yaml       # AlphaZero 2D MCTS + Transformer
├── nonogram/                   # 메인 패키지
│   ├── config.py               # Config 로딩 및 유효성 검증
│   ├── env/                    # 환경 모듈
│   │   ├── state.py            # State/Action 인코딩 및 유틸리티
│   │   ├── line_env.py         # 1D 줄 환경
│   │   ├── board_env.py        # 2D 보드 환경
│   │   └── feasibility.py      # Feasibility check (early stopping)
│   ├── models/                 # Q-Network / Policy Network
│   │   ├── registry.py         # 모델 레지스트리
│   │   ├── mlp.py              # MLP (기존 QNetwork 호환)
│   │   ├── dueling.py          # Dueling DQN (Shared, Advantage, Value)
│   │   ├── cnn.py              # 1D CNN
│   │   └── board_cnn.py        # 2D Board CNN
│   ├── agents/                 # RL 알고리즘
│   │   ├── registry.py         # 에이전트 레지스트리
│   │   ├── base.py             # 베이스 클래스 (LQL 손실 추가)
│   │   ├── dqn.py              # 표준 DQN (LQL 적용)
│   │   ├── double_dqn.py       # Double DQN (LQL 적용)
│   │   └── ppo.py              # PPO (LQL 적용)
│   ├── solvers/                # 2D 풀이 전략
│   │   ├── registry.py         # 풀이기 레지스트리
│   │   ├── line_solver.py      # 줄 교대 풀이 (DQN 기반 2D 풀이)
│   │   └── board_solver.py     # 보드 레벨 풀이
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
│   └── inference.py            # 최신 모듈형 1D 추론 시각화
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

### 1. 1D Line DQN — 줄 단위 Q-learning 풀이
- 1D Q-network가 각 줄에 대해 "이 action이 실패로 이어지는가?"를 판단.
- `Q ≤ threshold` 이면 반대 값 적용 전략으로 5x5에서 **98% hint-satisfied** 달성.
- 행↔열 교대 적용 및 deadlock 시 최고 Q값 셀 강제 결정.

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

### 4. 역방향 커리큘럼 학습 (Reverse Curriculum Learning) — 2D 보드 전용 지원
초기 상태 탐색의 희소 보상 문제를 방지하고 점진적으로 학습을 심화하도록 설계되었습니다.
- **정답 기반 초기 채우기**: 에피소드 초기 단계에서 보드의 일부 칸을 정답 값으로 채워진 상태로 시작합니다.
- **다양한 동적 난이도 조절 유형 지원**:
  * **Linear Decay (`type: "linear"`)**: 에피소드 진행도에 따라 난이도(비공개 칸 수)를 선형적으로 증가시킵니다.
  * **Cosine Annealing (`type: "cosine"`)**: 코사인 그래프 형상을 따라 초반에는 천천히 비워지다 중간에 빠르게 난이도를 올리고 후반에 완만하게 빈 보드로 수렴시킵니다.
  * **Performance-based Adaptive Decay (`type: "adaptive"`)**: 에이전트의 실시간 성공 성과를 분석하여 **자동으로 난이도를 가감**합니다. 최근 슬라이딩 성공률이 상한 임계치(예: `0.8`)를 넘으면 빈 칸을 늘려 난이도를 올리고, 하한 임계치(예: `0.2`) 미만으로 떨어지면 난이도를 낮춰 정답 지원을 늘리는(Backtracking) 유연한 적응형 학습을 실현합니다.
- **자동 한 칸 시작 ("auto")**: `start_ratio: "auto"` 설정을 통해 보드 크기에 맞추어 **정확히 단 1칸만 비우도록** (공개 비율 $\frac{N^2-1}{N^2}$) 초기 비율을 계산하여 수렴을 극대화합니다.

### 5. Action Chunking (Q-chunking) 기법 탑재 (NeurIPS 2025)
노노그램의 **장기 목표(Long-Horizon)**와 **희소 보상(Sparse-Reward)** 문제를 완화하기 위해 한 번의 추론으로 여러 스텝의 행동을 결정하고 실행하는 Action Chunking 기법을 통합하였습니다.
- **일관된 다중 탐색 (Temporal Coherence)**: 매 스텝 Q-value를 평가하는 대신, 한 번의 순방향 전파(forward pass)로 확신도가 높은 여러 셀의 플레이를 하나의 척(Chunk)으로 묶어 실행(Open-loop)합니다.
- **가치 백업 건너뛰기 (Skipped TD-backup)**: 척을 실행하고 난 뒤의 최종 상태($s_{t+H}$)를 기준으로 척 내부 액션들의 가치 함수를 n-step TD 목표값으로 업데이트하여 보상 정보를 빠르게 역방향 전파시킵니다.
- 적용 대상: DQN, Double DQN (MLP & Transformer) 에이전트에 적용 가능.

---

## 핵심 코드

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
```

### AlphaZero 2D 학습 (train_alphazero.py)

```bash
# 10x10 AlphaZero MCTS + Transformer 학습
uv run train_alphazero.py --config configs/alphazero_2d.yaml

# 기존 체크포인트에서 재개
uv run train_alphazero.py --config configs/alphazero_2d.yaml --resume
```

### 평가 및 추론 시각화 (scripts/)

```bash
# 2D 보드 솔버 (LineSolver) 다수 퍼즐 평가
uv run python scripts/evaluate.py --config configs/double_dqn_10x10.yaml --num_puzzles 200 --load_path ./runs/double_dqn_10x10_double_dqn_dueling_N10/checkpoints/latest.pt

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
| `model.type` | String | mlp \| dueling \| cnn \| board_cnn | 가치/정책 네트워크 구조 |
| `agent.type` | String | dqn \| double_dqn \| ppo \| alphazero_2d | 강화학습 알고리즘 종류 |
| `agent.lql_weight` | Float | $\ge 0.0$ | LQL 단조성 제약 손실 비중 |
| `agent.lql_n_step` | Integer | $\ge 1$ | LQL 가치 비교 시점 간격 |
| `agent.action_chunking` | Boolean | NOT NULL | Action Chunking 활성화 여부 |
| `agent.chunk_size` | Integer | $\ge 1$ | 한 번에 실행할 액션 척의 크기 |
| `training.reward_shaping` | Boolean | NOT NULL | A: 부분 달성도 기반 연속 보상 사용 여부 |
| `training.pretrain_bc` | Boolean | NOT NULL | B: MCTS 학습 시작 전 Behavior Cloning 사전 학습 여부 |
| `training.pretrain_episodes`| Integer | NOT NULL | BC 학습용 궤적 에피소드 수 |
| `training.pretrain_lr` | Float | NOT NULL | BC 학습용 learning rate |
| `training.curriculum.enabled`| Boolean | NOT NULL | 역방향 커리큘럼 학습의 활성화 여부 (2D 보드 전용) |
| `training.curriculum.type` | String | linear \| cosine \| adaptive | 커리큘럼 감쇠 유형 (선형 / 코사인 어닐링 / 성과 기반 적응형) |
| `training.curriculum.start_ratio`| Float \| String | NOT NULL | 초기 정답 공개 비율 (또는 "auto" 설정 시 2D 보드 내 단 1칸만 비우도록 설정) |
| `training.curriculum.end_ratio`| Float | NOT NULL | 최종 정답 공개 비율 |
| `training.curriculum.decay_episodes`| Integer | NOT NULL | 공개 비율 감쇠 에피소드 수 (linear/cosine 용) |
| `training.curriculum.adaptive_window`| Integer | NOT NULL | 성과(성공률) 판단용 슬라이딩 윈도우 크기 (adaptive 용) |
| `training.curriculum.adaptive_threshold`| Float | NOT NULL | 난이도를 높이기(비공개칸 증가) 위한 목표 성공률 임계치 (adaptive 용, 기본 0.8) |
| `training.curriculum.adaptive_backtrack_threshold`| Float | NOT NULL | 난이도를 낮추기(복습용 정답 지원) 위한 최저 성공률 임계치 (adaptive 용, 기본 0.2) |
| `training.curriculum.adaptive_step`| Float | NOT NULL | 성과 판정 시 조정할 reveal_ratio 증감 보폭 (adaptive 용, 기본 0.02) |

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

---

## 실험 결과

| 실험 | Config | 결과 |
|------|--------|------|
| 1D DQN 5x5 | `dqn_5x5.yaml` | hint_satisfied **98%**, exact_match **90%** |
| 1D DQN 10x10 | `dqn_10x10.yaml` | 학습 수렴 확인 |
| Double DQN 10x10 | `double_dqn_10x10.yaml` | LQL 적용 및 체크포인트 최적 보존 완료 |
| AlphaZero 2D 10x10 | `alphazero_2d.yaml` | Axial-Attention + LQL 튜닝 및 학습 재개 |

---

## 주의사항

- **1D 모델의 N == 보드의 줄 길이**가 반드시 같아야 합니다.
- **N이 커질수록 학습이 기하급수적으로 어려워집니다.** action space: `2N`, state space: ~`3^N`.
- **2D Board 환경**은 action space가 `2*N*N`으로 매우 크므로, 충분한 학습 에피소가 필요합니다.
- **AlphaZero 워커**는 GPU 경합 방지를 위해 **항상 CPU에서 추론**합니다. Learner만 GPU 사용.
- **Device 자동 감지**: CUDA > MPS(Apple Silicon) > CPU 순으로 자동 선택됩니다.