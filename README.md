# RL-based Nonogram Solver

1D Deep Q-Network을 학습하고, 그 모델을 이용해 2D 노노그램(네모로직)을 푸는 프로젝트.

핵심 아이디어는 다음과 같다.

1. **1D 모델 학습** — 길이 N짜리 한 줄(row 또는 column)에 대해, hint와 현재 채움 상태가 주어졌을 때 어떤 action이 정답으로 이어지는지를 학습한 Q-함수를 만든다.
2. **2D 풀이** — NxN 보드의 각 row와 column을 독립적인 1D 문제로 보고, 학습된 Q-함수가 "이 칸은 ■여야 한다 / □여야 한다"고 확신하는 정보만 모아 보드에 기록한다. row와 column을 교대로 처리하면서 새 정보가 더는 나오지 않을 때까지 반복한다.

---

## 파일 구조

```
.
├── utils.py            # state/action 인코딩, 환경 로직, Q-network 정의
├── train.py            # 1D DQN 학습 스크립트
├── inference.py        # 1D 모델 단일 줄 풀이 (디버깅/시각화용)
├── solve_nonogram.py   # 학습된 1D 모델로 NxN 노노그램 풀이
├── requirements.txt    # 의존성 (numpy, torch)
└── README.md
```

---

## 설치

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

GPU 빌드의 PyTorch가 필요하면 PyPI 기본 빌드 대신 [PyTorch 공식 안내](https://pytorch.org/get-started/locally/)의 인덱스 URL을 사용한다.

---

## 1. `utils.py` — 공통 모듈

`train.py`, `inference.py`, `solve_nonogram.py`가 모두 import하는 핵심 라이브러리.

### State / Hint 표현

한 줄의 state는 길이 `k + N`의 벡터로 표현된다. 여기서 `k = floor((N+1)/2)`는 가능한 hint의 최대 개수다.

```
s = [k개의 hint padding] + [N개의 cell 상태]
```

- **hint padding**: right-align, zero-padded. 예) N=5에서 `hint=[1,1]` → `[0, 1, 1]`, `hint=[0]` (빈 줄) → `[0, 0, 0]`
- **cell 상태**: `+1` = 칠해짐(■), `-1` = X(□로 확정), `0` = 미결정

| 함수 | 역할 |
|---|---|
| `hint_length(N)` | `k = floor((N+1)/2)` 계산 |
| `encode_hint(hint, N)` | hint를 길이 k로 패딩 |
| `extract_hint_from_blocks(blocks, N)` | 완성된 줄에서 hint를 추출 (terminal에서의 reward 판정에 사용) |
| `initial_state(hint, N)` | hint만 주어진 초기 state 만들기 |
| `is_terminal(state, N)` | 모든 cell이 결정되었는지 |
| `check_hint_match(blocks, hint_part, N)` | 완성된 줄이 hint와 일치하는지 |

### Action 표현

`a = (i, f)`이며 `i ∈ {0,…,N-1}`, `f ∈ {-1, +1}`. flat index로 다음과 같이 인코딩한다.

- `idx = i`     → cell i를 ■(+1)
- `idx = N + i` → cell i를 X(-1)

총 `2N`개의 action.

| 함수 | 역할 |
|---|---|
| `action_to_idx(i, f, N)` / `idx_to_action(idx, N)` | 양방향 변환 |
| `valid_actions_mask(state, N)` | 이미 결정된 cell의 action은 마스킹 |
| `apply_action(state, idx, N)` | state에 action 적용 (불변 — 사본 반환) |

### Feasibility check (early stopping용)

```python
is_feasible(blocks, hint_part, N) -> bool
```

부분적으로 채워진 줄이 **어떤 완성으로든** hint를 만족시킬 수 있는지를 backtracking으로 판정한다. 학습 중 더 이상 정답에 도달할 수 없는 dead-end가 명백할 때 즉시 reward −1로 끊어 학습 효율을 높이는 데 쓴다.

### `QNetwork`

```
입력: (k + N)
중간: hidden_dim → hidden_dim → hidden_dim (모두 ReLU)
출력: 2N (action별 Q-value)
```

---

## 2. `train.py` — 1D DQN 학습

### MDP 설정

- **state**: 위에서 정의한 `(k + N)` 벡터
- **action space**: `2N` (cell index × {■, X})
- **transition**: `apply_action` 결정적
- **reward**: terminal에서만 발생. hint와 일치하면 `+1`, 아니면 `-1`. `γ = 1`

### 데이터 생성 (replay buffer 수집)

매 episode마다:

1. 길이 N짜리 정답을 `{-1, +1}^N`에서 균등 샘플링하여 `sample`로 둔다. 그로부터 hint를 추출한다 — 이렇게 하면 항상 풀 수 있는 hint가 보장된다.
2. 두 가지 행동 정책 중 하나를 사용해 한 episode를 굴린다.
   - **towards mode**: 아직 비어 있는 cell 중 하나를 무작위로 골라 `sample`의 정답값을 채운다. → terminal에서 reward `+1`이 보장된다.
   - **random mode**: 유효한 action 중 하나를 무작위로 고른다. → terminal에서 `+1` 또는 `-1`.

`--mode alternate`이면 **episode 단위로** 한 모드를 선택하고 (`--towards_ratio` 확률로 towards), `--mode mix`이면 **step 단위로** 두 모드를 섞는다 (`--mix_prob` 확률로 towards).

### Early stopping

매 transition 후 `is_feasible`로 검사한다. 더 이상 hint를 만족시킬 수 없는 부분 상태라면 즉시 reward `-1`로 종료한다. 예: `hint=[0]`(빈 줄)인데 한 칸이라도 ■가 되면 즉시 끝. `--no_early_stop`으로 끌 수 있다. (towards mode에서는 정답을 따라가므로 발동되지 않는다.)

### DQN 업데이트

표준 DQN. target은 다음과 같이 계산한다.

```
target = r + (1 - done) · max_{a'} Q_target(s', a')      (γ = 1)
```

`Q_target(s', a')`는 마스킹된 유효 action만 고려한 `max`이다. Target network는 `--target_update_freq` step마다 hard-copy로 동기화하고, replay buffer는 `--buffer_size`까지 채워가며 `--min_buffer` 이상 모이면 학습을 시작한다.

저장 시점에 `N`, `hidden_dim`이 체크포인트에 같이 들어가므로, 추후 inference 시 자동 복원된다.

### CLI 사용 예

```bash
# 가장 단순한 호출
python train.py --N 5 --save_path qnet_N5.pt

# 10x10용 모델: episode와 모델 크기를 키우는 게 좋다
python train.py --N 10 --num_episodes 80000 --hidden_dim 256 \
                --buffer_size 200000 --save_path qnet_N10.pt

# step 단위 mix mode (정답 30%, 랜덤 70%)
python train.py --N 5 --mode mix --mix_prob 0.3 --save_path qnet_N5_mix.pt
```

### 주요 hyperparameter

| Flag | 기본값 | 설명 |
|---|---|---|
| `--N` | 5 | 한 줄의 길이 |
| `--num_episodes` | 30000 | 학습 episode 수 |
| `--batch_size` | 128 | 미니배치 크기 |
| `--buffer_size` | 100000 | replay buffer 용량 |
| `--min_buffer` | 2000 | 학습 시작 전 채워야 하는 buffer 크기 |
| `--train_freq` | 1 | 몇 step마다 gradient step을 밟을지 |
| `--target_update_freq` | 500 | target network 동기화 주기 |
| `--lr` | 1e-3 | Adam learning rate |
| `--hidden_dim` | 128 | 은닉 차원 |
| `--mode` | `alternate` | `alternate` 또는 `mix` |
| `--towards_ratio` | 0.5 | (alternate) 정답향 episode 비율 |
| `--mix_prob` | 0.5 | (mix) per-step 정답향 확률 |
| `--no_early_stop` | (off) | feasibility 기반 조기 종료 비활성화 |
| `--save_path` | `qnet.pt` | 체크포인트 저장 경로 |
| `--log_freq` | 500 | 로그 출력 간격 |
| `--seed` | 42 | 랜덤 시드 |

학습 로그에는 최근 `log_freq` episode의 평균 reward와 성공률(`reward > 0` 비율)이 찍힌다. 성공률이 0.95 근처로 안정화되면 학습이 충분히 된 것으로 본다.

---

## 3. `inference.py` — 단일 줄 풀이

학습된 1D 모델을 한 줄에 적용해 풀이 과정을 단계별로 보여준다. 모델 디버깅/시각화용.

### 정책

매 step마다:

1. 현재 state에서 Q(s, a) 계산
2. **유효 action 중 `Q ≤ neg_threshold`(기본 -0.5)인 것**을 찾는다 — 이는 "이 action은 -1 reward로 이어진다", 즉 "cell j는 반대 값(-f)이어야 한다"는 정보
3. 그런 action이 있으면 → 가장 confident하게 나쁜 (Q가 가장 작은) action `(j, f)`의 **반대** `(j, -f)`를 수행
4. 없으면 → 유효 action 중 `argmax Q`

terminal에 도달할 때까지 반복하며, 매 step의 state, Q-value 표, 선택 사유를 출력한다.

### CLI 사용 예

```bash
python inference.py --N 5 --hint 1 2 --load_path qnet_N5.pt
python inference.py --N 5 --hint 0   --load_path qnet_N5.pt   # 빈 줄
```

### 주요 옵션

| Flag | 설명 |
|---|---|
| `--N` | 줄 길이 (체크포인트의 N이 우선 적용됨) |
| `--hint` | 공백으로 구분한 hint 정수. 빈 줄은 `--hint 0` |
| `--load_path` | 학습된 체크포인트 |
| `--neg_threshold` | "Q == -1"로 판정할 임계값 (기본 -0.5) |

### 출력 예시

```
=== step 1 ===
state: [0, 1, 2, 0, 0, 0, 0, 0]    row: ?????
Q(s, a):
  a = (0, +1)  ->  Q = +1.0245
  a = (1, +1)  ->  Q = +0.9960
  ...
  a = (3, -1)  ->  Q = -0.9966
  ...
-> chosen: a = (3, +1)    (action (3, -1) has Q=-0.9966 (<= -0.5); taking opposite (3, +1).)
```

> **Q 값이 ±1을 살짝 넘는 이유**: γ=1이고 reward가 ±1이므로 이론적 Q는 [-1, 1] 안이지만, MSE + bootstrapping을 쓰는 DQN은 `max` 연산자 때문에 overestimation bias가 생긴다. 출력층에 `tanh`를 씌우거나 Double DQN을 쓰면 [-1, 1]에 더 가깝게 학습된다.

---

## 4. `solve_nonogram.py` — 2D 풀이

학습된 1D 모델을 사용해 NxN 노노그램을 푼다.

### Hint 입력 방식

파일 상단의 두 상수를 직접 편집한다.

```python
ROW_HINTS = [
    [1, 3],
    [1, 1, 1, 3],
    ...
]

COL_HINTS = [
    [4],
    [1],
    ...
]
```

`N`은 `len(ROW_HINTS)`로 자동 추론한다. `ROW_HINTS`와 `COL_HINTS`의 길이가 다르면 즉시 에러. 1D 모델의 `N`도 같아야 한다.

### 알고리즘

```
board <- 모두 0(unknown)
dirty_rows <- {0,…,N-1},  dirty_cols <- {0,…,N-1}
repeat:
    row pass: dirty한 모든 row를 batch로 forward.
              각 row에서 valid & Q <= neg_threshold인 action (j, f)을
              찾아 cell (row, j)을 -f로 채운다. 채워진 cell의 column은
              dirty로 마킹. 변화가 있고 미완성인 row는 다음 inner pass에서
              다시 forward.
    col pass: 위와 대칭.
until 한 outer pass에서 row/col 모두 변화 없음
```

원래 사용자의 의사코드에 비해 세 가지가 효율화되어 있다.

1. **−1 action 일괄 적용**: 같은 시점에 Q ≈ −1인 action들은 서로 독립이므로 forward 한 번에 모두 board에 쓰고 다음 forward로 넘어간다.
2. **줄 batching**: 한 axis의 dirty한 모든 줄을 한 번의 forward로 평가.
3. **Dirty marking**: 어떤 cell이 변할 때만 직교 방향의 해당 줄을 dirty로 표시. 변화 없는 줄은 다음 pass에서 건드리지 않는다.

### CLI 사용 예

```bash
# ROW_HINTS / COL_HINTS는 코드 상단에서 직접 수정
python solve_nonogram.py --load_path qnet_N10.pt
python solve_nonogram.py --load_path qnet_N10.pt --verbose   # 매 pass마다 보드 출력
```

### 주요 옵션

| Flag | 설명 |
|---|---|
| `--load_path` | 1D 모델 체크포인트 (N이 보드 크기와 같아야 함) |
| `--hidden_dim` | 체크포인트에 누락된 경우만 사용 |
| `--neg_threshold` | "Q == -1" 판정 임계값 (기본 -0.5) |
| `--max_outer_iters` | row↔col 교대 횟수 상한 (안전장치) |
| `--verbose` | 매 outer pass마다 board 상태 출력 |

### 출력

위쪽에는 column hint(세로로 stack), 왼쪽에는 row hint, board는 ■(채움) / □(미결정 또는 X) 로 표시.

```
                         2
             2   3     2 3
         4 1 2 3 4 8 9 4 1 2
    1 3  ■ □ ■ □ ■ □ ■ □ ■ □
1 1 1 3  □ ■ □ ■ □ ■ □ ■ □ ■
    1 3  ■ □ ■ □ ■ □ ■ □ ■ □
    ...
```

종료 후 `filled / X / unknown` 통계를 같이 출력한다. unknown 칸이 남으면 1D 모델만으로는 풀리지 않는 케이스(학습 부족이거나 줄 단위 Q-함수의 정보로는 부족한 퍼즐)이다.

---

## 권장 워크플로

1. 풀고 싶은 보드 크기 `N` 결정
2. 그 `N`으로 1D 모델 학습
   ```bash
   python train.py --N 10 --num_episodes 80000 --hidden_dim 256 \
                   --save_path qnet_N10.pt
   ```
3. (선택) `inference.py`로 모델이 한 줄을 잘 푸는지 확인
   ```bash
   python inference.py --N 10 --hint 2 3 1 --load_path qnet_N10.pt
   ```
4. `solve_nonogram.py` 상단의 `ROW_HINTS` / `COL_HINTS`를 풀고 싶은 퍼즐로 채우고 실행
   ```bash
   python solve_nonogram.py --load_path qnet_N10.pt --verbose
   ```

---

## 주의사항

- **1D 모델의 N == 보드의 줄 길이**가 반드시 같아야 한다. 5x5 보드에는 N=5 모델, 10x10 보드에는 N=10 모델이 필요하다. `solve_nonogram.py`는 mismatch면 명확한 에러를 띄운다.
- **N이 커질수록 학습이 빠르게 어려워진다.** action space는 `2N`이지만 state space는 `3^N`(가능한 cell 조합) 정도라 N=10만 되어도 state 공간이 5만 배 가까이 커진다. `--num_episodes`, `--buffer_size`, `--hidden_dim`을 키우자.
- **2D 풀이가 모든 노노그램을 풀어주지는 않는다.** 줄 단위 Q-함수가 잡지 못하는 "두 줄 이상의 cross-line 추론이 필요한" 퍼즐은 unknown 칸이 남는다. 이런 케이스는 추후 2D 모델로 확장 가능한 영역이다.