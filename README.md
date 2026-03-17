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
- Epochs: 150

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
3. 활성화 분산으로 가중된 확률 분포에서 K개 노드를 샘플링하여 incoming 연결 소스로 선택
4. 출력 노드의 부분집합(ceil(n_out/2)~n_out개)에 outgoing 연결 (노드마다 서로 다른 부분집합)
5. 새 가중치는 작은 노이즈(std=0.01)로 초기화

### 노드 프루닝
매 epoch 비효율적 HIDDEN 노드를 자동 제거 (grace period 이후):
- **Dead neuron**: 모든 샘플에서 활성화가 0
- **Low variance**: 활성화 분산이 임계값 미만
- **Low contribution**: 나가는 간선 가중치 절대값 합이 임계값 미만

### 간선 프루닝
매 epoch 가중치 절대값이 임계값 미만인 간선을 제거. 노드 고립 방지를 위해 최소 incoming/outgoing 간선 수를 보장.

### 간선 추가
노드 프루닝이 없는 epoch에 실행. 비출력×비입력 노드 쌍에서 랜덤 후보를 샘플링하고, 양쪽 활성화 분산의 곱으로 점수를 매겨 상위 N개 간선을 추가.

### 옵티마이저 재구성
성장 또는 프루닝 발생 시 Adam 옵티마이저와 CosineAnnealing 스케줄러를 재구성하여 새 파라미터 반영.

## 핵심 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `patience` | 2 | 정체 판단 epoch 수 |
| `min_delta` | 5e-4 | 최소 개선 임계값 |
| `growth_neurons` | 16 | 한 번에 추가할 뉴런 수 |
| `growth_cooldown` | 1 | 성장 후 쿨다운 epoch |
| `incoming_k` | 8 | incoming 연결 소스 수 |
| `max_hidden` | 256 | 최대 히든 노드 수 |
| `edges_per_epoch` | 32 | epoch당 추가할 간선 수 |
| `edge_candidates` | 200 | 간선 추가 후보 샘플 수 |
| `edge_prune_threshold` | 0.001 | 간선 프루닝 가중치 임계값 |
| `prune_grace_epochs` | 3 | 새 노드 프루닝 유예 epoch |
| `prune_var_threshold` | 1e-6 | 노드 프루닝 분산 임계값 |
| `prune_contrib_threshold` | 0.01 | 노드 프루닝 기여도 임계값 |

## 출력

- `results.json` — 실험별 학습 이력 (loss, accuracy, 파라미터 수, 노드/간선 수, 성장/프루닝/간선 이벤트)
- `results.png` — 2×3 비교 그래프 (Val Accuracy, Val Loss, Parameter Count, Node Count, Edge Count, Train vs Val Accuracy / 성장·프루닝·간선 이벤트 표시)
- `graphnet_model.pt` — 학습된 GraphNet 모델 저장 (시각화용)

## 실험 결과 (MNIST, 150 epochs)

| 모델 | 최고 val_acc | 최종 params | 성장 횟수 | 프루닝 횟수 | 간선 추가 | 간선 제거 |
|------|-------------|------------|----------|-----------|----------|----------|
| Baseline (Fixed MLP) | **98.19%** | 55,306 | - | - | - | - |
| GraphNet-Growth | 96.83% | 8,924 | 32회 | 62회 | 88회 | 150회 |

### 분석

- **정확도**: Baseline이 1.36%p 높음 (98.19% vs 96.83%)
- **파라미터 효율성**: GraphNet이 Baseline 대비 **약 16%의 파라미터**(8,924 vs 55,306)로 96.83% 달성
- **동적 구조 변화**:
  - 초기 hidden 노드 0개에서 시작 → 최종 201개 hidden 노드로 성장
  - 최종 구조: 995 노드, 8,880 간선, 6 레벨 DAG
  - 첫 번째 성장: epoch 42 (loss 정체 감지)
  - 성장과 프루닝이 반복되며 구조가 자율적으로 최적화됨
- **학습 곡선**:
  - Baseline: 빠르게 수렴 (epoch ~50에서 대부분 수렴)
  - GraphNet: 점진적 성장과 함께 꾸준히 개선 (epoch 142에서 best val_loss=0.1084)
- **구조 진화 패턴**:
  - epoch 1~41: hidden 0→16, 직접 연결(input→output)로 학습
  - epoch 42~100: 활발한 성장기, hidden 16→153
  - epoch 100~150: 성장+프루닝 균형, hidden 150~200 범위에서 진동

### 환경

- GPU: CUDA
- PyTorch: 2.10.0
- Dataset: MNIST (60K train / 10K test)
- Optimizer: Adam (lr=1e-3) + CosineAnnealingLR
