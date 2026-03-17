# 동적 뉴로제네시스 네트워크 V2 — 설계 문서

> 기존 DAG 기반 GraphNet에서 출발해, 생물학적 뇌에 가까운 3-레이어 아키텍처로의 발전 방향을 기록한 설계 문서.

---

## 1. 현재 구조 (V1) 요약

### 파일 구조

```
dynamic_net/
├── models/
│   ├── dynamic_net.py              # 레이어 기반 동적 MLP (Baseline)
│   ├── graph_net.py                # 그래프 기반 동적 네트워크 (인접 행렬 + DAG)
│   ├── neurogenesis_controller.py  # 정체 감지 & 뉴런 추가/프루닝
│   └── rl_controller.py           # Policy Gradient 연결 결정 (보존)
├── train.py
└── README.md
```

### 핵심 알고리즘 (V1)

- **GraphNet**: N×N 인접행렬(W) + 이진 마스크(mask). Kahn's 알고리즘으로 위상 정렬 후 레벨별 행렬 연산.
- **Neurogenesis**: validation loss 정체 감지 → 활성화 분산 기반 샘플링으로 K개 뉴런 삽입 → 프루닝 (dead / low-variance / low-contribution).
- **제약**: DAG 강제 (상삼각 마스크) — 순환 연결 불허.

### V1의 한계

| 한계 | 설명 |
|---|---|
| DAG 제약 | 실제 뇌는 순환 연결을 가짐. 피드백 루프 불가. |
| 연속값 뉴런 | 실제 뉴런은 스파이크(0/1)로 발화. |
| 단일 시간 스케일 | 뇌는 ms / 100ms / epoch 단위 세 레이어가 동시 작동. |
| 글로벌 역전파만 | 실제 뇌는 로컬 학습 규칙(STDP) 사용. |

---

## 2. 관련 연구 지형

### 직접 연관 연구

| 연구 | 유사도 | 설명 |
|---|---|---|
| Growing Networks via Gradient Descent (Royal Society A, 2025) | ★★★★★ | 크기 자체를 미분 가능한 파라미터로. 동일 아이디어 병행 연구. |
| Growth Strategies for Arbitrary DAG Neural Architectures (ESANN 2025) | ★★★★★ | DAG에서 자유 성장. MNIST 벤치마크 동일. |
| NEAT (Stanley & Miikkulainen, 2002) | ★★★★★ | initial_hidden=0 시작 + 점진적 성장 철학 동일. |
| SET / RigL | ★★★★☆ | 간선 추가/제거 알고리즘의 선행 연구. |
| CHTss (NeurIPS 2025) | ★★★★☆ | 뇌 네트워크 과학 기반 DST. LLM까지 확장. |
| NDSNN | ★★★★☆ | 뉴로제네시스 영감 SNN. Drop-and-grow 전략. |

### 점진적 증가 방법 비교

| 방법 | 추가 단위 | 트리거 | 위치 결정 |
|---|---|---|---|
| Cascade Correlation | 뉴런 1개씩 | 오차 정체 | 잔차 상관 |
| DeDNN | 뉴런 or 레이어 | 분포 변화 | 성능 기반 |
| SET | 간선 | 매 epoch | 무작위 |
| RigL | 간선 | 매 epoch | 그래디언트 크기 |
| **DNN (V1)** | **뉴런 K개 + 간선** | **loss 정체** | **활성화 분산 샘플링** |

---

## 3. 생물학적 뉴런 작동 원리

### 뉴런 구조

```
수상돌기(dendrite) → 세포체(soma) → 축삭돌기(axon) → 시냅스 말단
```

- 수상돌기: 다른 뉴런으로부터 신호 수신
- 세포체: 입력 신호 통합, 핵(nucleus) 포함
- 축삭돌기: 말이집(myelin)으로 감싸인 신호 전달 경로
- 시냅스 소포: 신경전달물질(NT) 저장 및 방출

### 활동전위 (Action Potential) 4단계

1. **휴지 상태**: −70 mV. K+ 채널 열림, Na+ 채널 닫힘.
2. **탈분극**: 역치(−55 mV) 초과 → Na+ 급속 유입 → −70 → +40 mV.
3. **재분극**: Na+ 닫힘, K+ 유출 → 전위 하강.
4. **과분극 / 불응기**: −90 mV까지 하강. 약 1~2ms 재발화 불가.

> 전부 아니면 전무(all-or-nothing): 역치를 넘으면 반드시 발화, 못 넘으면 침묵.

### 시냅스 전달

```
활동전위 도달 → Ca²+ 유입 → 소포 융합 → NT 방출 → 간극(20~40nm) 확산
→ 후시냅스 수용체 결합 → EPSP(흥분) or IPSP(억제)
```

### STDP (Spike-Timing-Dependent Plasticity)

```
ΔW = +A+ · exp(Δt / τ+)   if Δt < 0   → LTP (전시냅스 먼저 발화 → 강화)
ΔW = −A− · exp(−Δt / τ−)  if Δt > 0   → LTD (후시냅스 먼저 발화 → 약화)

Δt = t_post − t_pre
```

역전파 없이 **로컬 타이밍만으로** 가중치를 업데이트하는 생물학적 학습 규칙.

---

## 4. V2 핵심 설계: 3-레이어 아키텍처

### 전체 구조

```
레이어 1 — SNN T스텝 (판단)        ~1ms 단위
────────────────────────────────────────────────
  t=0 발화 → t=1 전파 → ... → t=T 행동 결정
  입력 고정, 내부 스파이크만 전파
  출력: 행동 a_t (발화율 → argmax)
                    │ a_t
                    ▼
레이어 2 — RL 스텝 (학습)          ~100ms 단위
────────────────────────────────────────────────
  (s_t, a_t) → r_t → s_{t+1}
  보상 신호로 SNN 가중치 W 업데이트 (PPO / A3C)
                    │ 성능 정체 감지
                    ▼
레이어 3 — Neurogenesis (성장)     epoch 단위
────────────────────────────────────────────────
  정체 감지 → 뉴런 추가(K개) → 프루닝 → 옵티마이저 재구성
  구조 변경 → 레이어 1/2에 피드백
```

### 시간 스케일 대응

| 레이어 | 시간 단위 | 무엇이 바뀌는가 | 뇌의 대응 |
|---|---|---|---|
| SNN T스텝 | ~1ms | 뉴런 막전위, 스파이크 | 신경 발화 |
| RL 스텝 | ~100ms | 환경 상태, 시냅스 가중치 | 행동 결정 |
| Neurogenesis | epoch | 네트워크 구조 | 시냅스 생성/소멸 |

### RL과 SNN의 시간 개념 차이

RL의 `s_t`는 외부 환경이 바뀐 결과. 에이전트 행동 → 환경 반응 → 새 상태.

SNN의 `t`는 같은 입력에 대한 내부 신호 전파 과정. 입력은 T스텝 내내 고정, 스파이크가 네트워크 안에서 퍼져나가는 시간만 흐름.

```
RL 스텝 (환경 상호작용)
────────────────────────────────────►
  s₀              s₁              s₂
  │               │               │
  ▼               ▼               ▼
[SNN T스텝]   [SNN T스텝]   [SNN T스텝]
 t=0..T-1      t=0..T-1      t=0..T-1
  │               │               │
  a₀              a₁              a₂
```

---

## 5. SNN 구현 설계

### LIF (Leaky Integrate-and-Fire) 뉴런 모델

```
τ · dV/dt = −(V − V_rest) + R·I(t)

V ≥ V_thresh → 스파이크 발화, V = V_reset
```

- `τ` (시상수): 전위 누설 속도. 높을수록 천천히 쌓임. **학습 가능한 파라미터**.
- `V_thresh` (역치): 기본 −55 mV.
- `V_reset` (리셋 전위): 기본 −75 mV.
- **불응기**: 발화 후 ~2ms 재발화 불가.

### 서로게이트 그래디언트 (Surrogate Gradient)

스파이크는 0/1 불연속 → 미분 불가. Forward는 헤비사이드, Backward는 시그모이드 미분으로 근사.

```python
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, threshold=1.0):
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        return (v >= threshold).float()          # 헤비사이드

    @staticmethod
    def backward(ctx, grad_output):
        v, = ctx.saved_tensors
        sg = torch.sigmoid(4 * (v - ctx.threshold))
        surrogate_grad = sg * (1 - sg) * 4      # 시그모이드 도함수
        return grad_output * surrogate_grad, None
```

### 자유 그래프 Forward Pass (순환 허용)

DAG 제약 제거의 핵심 원칙:

```python
# 순환이 있어도 안전한 이유:
# S는 항상 '이전' 타임스텝의 스파이크를 참조
I = S @ W_eff.T   # S = 직전 타임스텝 스파이크

# → 현재 타임스텝 계산이 막히지 않음
# → 임의의 순환 구조 허용
```

Kahn 알고리즘(위상 정렬) 제거. 시간 축 도입으로 대체.

### 입력 인코딩 방식 (선택 필요)

| 방식 | 원리 | 장단점 |
|---|---|---|
| Rate coding | 픽셀값 → 포아송 스파이크 확률 | 구현 쉬움, T스텝 필요 |
| Temporal coding | 밝을수록 먼저 발화 | 빠름, T=1 가능 |
| Direct coding | 픽셀값을 전류로 직접 주입 | 가장 단순, 비생물학적 |

### GraphNet → GraphSNN 교체 인터페이스

기존 `graph_net.py`의 W, mask 인터페이스를 유지하면서 내부만 교체:

```python
class GraphSNN(nn.Module):
    # 동일한 외부 인터페이스
    # W     : (N×N) 시냅스 가중치
    # mask  : (N×N) 연결 존재 여부 — 상삼각 제약 제거
    # node_type: INPUT(0) / HIDDEN(1) / OUTPUT(2)

    # 추가되는 SNN 전용 상태
    # log_tau : (N,) 각 뉴런의 시상수 (학습 가능)
    # threshold: 발화 역치

    # Kahn 알고리즘 → T스텝 시뮬레이션으로 대체
    # add_neurons(), prune_neuron() 인터페이스 동일 유지
```

---

## 6. 3-레이어 통합 시 예상 창발적 특성

### 난이도 자동 적응

```
쉬운 환경 → RL만으로 해결 → Neurogenesis 미발동 → 작은 네트워크
어려운 환경 → RL 정체 → Neurogenesis 발동 → 성장 → 재학습
```

네트워크 크기가 문제 난이도에 맞게 자동 조절됨.

### 점진적 커리큘럼 학습

```
작은 SNN (단순 전략)
→ RL 정체
→ Neurogenesis (표현력 확장)
→ 더 복잡한 패턴 학습
→ RL 정체
→ Neurogenesis
→ ...
```

인간의 학습 방식과 구조적으로 동일.

### Catastrophic Forgetting 완화

기존 뉴런은 유지하고 새 뉴런을 추가하는 방식 → 기존 지식이 비교적 보존됨.

---

## 7. 미해결 문제 (Open Problems)

### 문제 1: RL 신호 → SNN 전달

SNN은 스파이크(0/1) 출력. RL policy gradient는 연속 확률 분포 필요.

```
선택지:
A) Rate coding → 발화율을 확률로 변환 (단순, 정보 손실)
B) Population coding → 뉴런 집단 패턴을 확률로 (복잡, 풍부)
C) 별도 readout layer → 스파이크를 연속값으로 변환 (절충)
```

### 문제 2: Neurogenesis 타이밍 결정

RL 보상은 noisy → 정체 판단이 어려움.

```
선택지:
A) 보상 이동 평균 정체 (단순, 노이즈 취약)
B) 정책 엔트로피 모니터링 (탐색 감소 = 정체)
C) SNN 내부 활성화 분산 (직접적 표현력 측정)
```

### 문제 3: 구조 변화 후 학습 불안정

뉴런 추가 시 가치 함수(critic) 입력 차원 변화 → 기존 가치 추정 순간 파괴.

```
선택지:
A) 새 뉴런 학습률 일시적으로 높이기
B) Critic을 구조 변화에 독립적으로 설계 (고정 크기 상태 표현)
C) 성장 직후 cooldown (학습률 잠시 낮추기)
```

---

## 8. 구현 로드맵

```
1단계 ── SNN forward 교체
│   GraphNet의 Kahn 알고리즘 → T스텝 시뮬레이션
│   연속값 뉴런 → LIF 스파이크 뉴런
│   DAG mask → 순환 허용 mask
│   입력: Poisson rate coding
│   출력: 발화율 → 분류
│
2단계 ── RL 연결
│   SNN 출력 → 행동 확률 변환
│   환경 선택 (CartPole / MiniGrid)
│   PPO or A3C 학습 루프
│
3단계 ── Neurogenesis 통합
│   RL 보상 정체 감지
│   뉴런 추가/제거 (V1 인터페이스 유지)
│   옵티마이저 재구성
│
4단계 ── STDP 추가 (도전적)
    역전파(글로벌) + STDP(로컬) 하이브리드
    완전한 생물학적 학습
```

---

## 9. 파일 구조 (V2 목표)

```
dynamic_net_v2/
├── models/
│   ├── snn/
│   │   ├── lif_neuron.py          # LIF 뉴런 + 서로게이트 그래디언트
│   │   ├── graph_snn.py           # 자유 그래프 SNN (순환 허용)
│   │   └── stdp.py                # STDP 학습 규칙
│   ├── rl/
│   │   ├── ppo_agent.py           # PPO 에이전트
│   │   └── reward_monitor.py      # 정체 감지
│   ├── neurogenesis/
│   │   ├── controller.py          # 성장/프루닝 (V1 인터페이스 유지)
│   │   └── growth_strategy.py     # Small-world 초기화 등
│   └── __init__.py
├── envs/
│   └── wrapper.py                 # 환경 래퍼 (Gym 호환)
├── train_v2.py                    # 3-레이어 통합 학습 루프
└── README_V2.md                   # 이 문서
```

---

## 10. 핵심 하이퍼파라미터 (V2 예상)

### SNN

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `T` | 20 | 타임스텝 수 |
| `tau_init` | 20.0 ms | 초기 시상수 |
| `threshold` | 1.0 | 발화 역치 |
| `reset_mode` | `zero` | 리셋 방식 (zero / subtract) |
| `refrac_period` | 2 ms | 불응기 |
| `encoding` | `rate` | 입력 인코딩 방식 |

### RL

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `gamma` | 0.99 | 할인율 |
| `lr_actor` | 3e-4 | 정책 학습률 |
| `lr_critic` | 1e-3 | 가치 함수 학습률 |
| `clip_eps` | 0.2 | PPO 클리핑 |
| `n_steps` | 2048 | 롤아웃 길이 |

### Neurogenesis

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `patience` | 10 에피소드 | 정체 판단 기준 |
| `growth_neurons` | 16 | 한 번에 추가할 뉴런 수 |
| `max_hidden` | 512 | 최대 히든 노드 수 |
| `prune_var_threshold` | 1e-6 | 프루닝 분산 임계값 |
| `growth_cooldown` | 5 에피소드 | 성장 후 쿨다운 |

---

*작성일: 2026-03-15 | 기반 대화: DAG 구조 한계 분석 → 뉴런 생물학 → SNN 설계 → 3-레이어 아키텍처*
