# 동적 뉴로제네시스 네트워크 (DNN)

학습 중 성능 정체를 감지하면 자동으로 뉴런을 추가하고, 비효율적 뉴런은 제거하는 동적 네트워크.

## 실행 방법

```bash
conda activate dynamic_net
python train.py
```

## 파일 구조

```
dynamic_net/
├── models/
│   ├── __init__.py                 # 모듈 import
│   ├── dynamic_net.py              # 레이어 기반 동적 MLP (Baseline용)
│   ├── graph_net.py                # 그래프 기반 동적 네트워크 (인접 행렬 + DAG)
│   ├── neurogenesis_controller.py  # 정체 감지 & 뉴런 추가/프루닝
│   └── rl_controller.py           # Policy Gradient 연결 결정 (보존)
├── train.py                        # 실험 실행 (2가지 조건 비교)
└── README.md
```

## 실험 조건

| 모델 | 설명 |
|------|------|
| Baseline | 고정 MLP [64→64→10] |
| GraphNet-Growth | 그래프 기반 동적 노드 추가 + 프루닝 (인접 행렬 + 위상 정렬) |

- Dataset: MNIST (기본) / CIFAR-10 (선택 가능)
- Epochs: 50

## 핵심 아키텍처

### GraphNet (graph_net.py)

레이어 개념 없이 개별 노드를 자유롭게 연결하는 그래프 기반 네트워크.

- **W** (N×N 인접 행렬): 연결 가중치
- **mask** (N×N 이진 행렬): 연결 존재 여부
- **DAG 제약**: 위상 순서 유지로 순환 방지
- **Forward pass**: Kahn's 알고리즘 위상 정렬 → 레벨별 행렬 연산 (GPU 병렬화)
- **노드 종류**: INPUT(0), HIDDEN(1), OUTPUT(2)
- **initial_hidden=0**: 초기에는 input→output 직접 연결, 이후 성장으로 hidden 노드 추가

### DynamicNet (dynamic_net.py)

Baseline으로 사용하는 고정 구조 MLP. 레이어별 노드 삽입 기능을 갖추고 있으나, 실험에서는 고정 크기로 사용.

## 알고리즘

### 뉴런 삽입 (GraphNeurogenesisController)
1. 매 epoch validation loss 추적
2. `patience` epoch 동안 개선 < `min_delta`이면 정체 선언
3. 활성화 분산이 높은 top-K 노드를 incoming 연결 소스로 선택
4. 모든 출력 노드에 outgoing 연결 (즉시 기여)
5. 새 가중치는 작은 노이즈(std=0.01)로 초기화

### 프루닝
매 epoch 비효율적 HIDDEN 노드를 자동 제거:
- **Dead neuron**: 모든 샘플에서 활성화가 0
- **Low variance**: 활성화 분산이 임계값 미만
- **Low contribution**: 나가는 간선 가중치 절대값 합이 임계값 미만

### 옵티마이저 재구성
성장 또는 프루닝 발생 시 Adam 옵티마이저와 CosineAnnealing 스케줄러를 재구성하여 새 파라미터 반영.

## 핵심 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `patience` | 4 | 정체 판단 epoch 수 |
| `min_delta` | 5e-4 | 최소 개선 임계값 |
| `growth_neurons` | 16 | 한 번에 추가할 뉴런 수 |
| `growth_cooldown` | 3 | 성장 후 쿨다운 epoch |
| `top_k_incoming` | 16 | incoming 연결 소스 수 |
| `max_hidden` | 256 | 최대 히든 노드 수 |
| `prune_var_threshold` | 1e-6 | 프루닝 분산 임계값 |
| `prune_contrib_threshold` | 0.01 | 프루닝 기여도 임계값 |

## 출력

- `results.json` — 실험별 학습 이력 (loss, accuracy, 파라미터 수, 성장/프루닝 이벤트)
- `results.png` — Val Accuracy / Val Loss / Parameter Count 비교 그래프 (성장/프루닝 이벤트 표시)
