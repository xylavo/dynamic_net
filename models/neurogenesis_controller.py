"""
NeurogenesisController: 학습 중 성능(loss)을 모니터링해서
정체 구간에 뉴런을 삽입하는 컨트롤러.

알고리즘:
1. 매 epoch마다 validation loss를 추적
2. 최근 K epoch 동안 개선이 δ 미만이면 '정체' 선언
3. 각 레이어의 활성화 분산을 계산해서 가장 '포화'된 레이어에 뉴런 삽입

GraphNeurogenesisController: GraphNet용 컨트롤러
- 노드별 활성화 분산을 기반으로 연결할 소스 노드 선택
- 출력 노드에 즉시 연결하여 기여 시작
"""
import torch
import torch.nn as nn
from collections import deque
from typing import Optional, List, Tuple
import numpy as np

from models.dynamic_net import DynamicNet


class ActivationRecorder:
    """
    Forward hook으로 각 레이어의 활성화값을 기록.
    어느 레이어가 포화됐는지 판단하는 데 사용.
    """

    def __init__(self):
        self.activations: dict = {}
        self._hooks = []

    def register(self, net: DynamicNet) -> None:
        """모든 히든 레이어에 hook 등록."""
        self.clear()
        for i, layer in enumerate(net.layers[:-1]):
            hook = layer.register_forward_hook(
                lambda m, inp, out, idx=i: self._record(idx, out)
            )
            self._hooks.append(hook)

    def _record(self, idx: int, output: torch.Tensor) -> None:
        acts = output.detach()
        if idx not in self.activations:
            self.activations[idx] = []
        self.activations[idx].append(acts.cpu())

    def get_saturation_scores(self) -> List[float]:
        """
        레이어별 포화도 점수 계산.
        높을수록 더 많은 뉴런이 필요한 레이어.

        포화도 = 활성화 분산의 역수 (분산이 낮으면 뉴런들이 비슷하게 반응 → 다양성 부족)
        """
        scores = []
        for idx in sorted(self.activations.keys()):
            all_acts = torch.cat(self.activations[idx], dim=0)
            # 뉴런별 분산 계산 후 평균
            var_per_neuron = all_acts.var(dim=0).mean().item()
            scores.append(1.0 / (var_per_neuron + 1e-8))
        return scores

    def get_dead_neuron_ratio(self) -> List[float]:
        """
        ReLU dead neuron 비율 계산 (항상 0인 뉴런).
        높을수록 해당 레이어가 용량을 낭비하는 중.
        """
        ratios = []
        for idx in sorted(self.activations.keys()):
            all_acts = torch.cat(self.activations[idx], dim=0)
            dead = (all_acts == 0).all(dim=0).float().mean().item()
            ratios.append(dead)
        return ratios

    def clear(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.activations.clear()


class NeurogenesisController:
    """
    학습 중 성능 정체를 감지하고 뉴런을 동적으로 삽입.
    """

    def __init__(
        self,
        net: DynamicNet,
        patience: int = 5,            # 정체 판단 epoch 수
        min_delta: float = 1e-4,      # 최소 개선 임계값
        max_neurons_per_layer: int = 256,  # 레이어 최대 크기
        growth_neurons: int = 8,      # 한 번에 추가할 뉴런 수
        growth_cooldown: int = 3,     # 성장 후 쿨다운 epoch
        verbose: bool = True,
    ):
        self.net = net
        self.patience = patience
        self.min_delta = min_delta
        self.max_neurons = max_neurons_per_layer
        self.growth_neurons = growth_neurons
        self.cooldown = growth_cooldown
        self.verbose = verbose

        self.loss_history: deque = deque(maxlen=patience + 1)
        self.best_loss: float = float('inf')
        self.epochs_since_improvement: int = 0
        self.cooldown_counter: int = 0
        self.growth_events: List[dict] = []

        self.recorder = ActivationRecorder()

    def register_hooks(self) -> None:
        """학습 시작 전 hook 등록."""
        self.recorder.register(self.net)

    def step(self, val_loss: float, epoch: int) -> Optional[int]:
        """
        매 epoch 끝에 호출. 뉴런을 추가했으면 추가된 레이어 인덱스 반환.

        Returns:
            int if growth occurred, None otherwise
        """
        self.loss_history.append(val_loss)

        # 개선 여부 확인
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        # 쿨다운 중
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None

        # 정체 감지
        is_stagnated = (
            self.epochs_since_improvement >= self.patience
            and len(self.loss_history) >= self.patience
        )

        if not is_stagnated:
            return None

        # 어느 레이어에 뉴런을 추가할지 결정
        layer_idx = self._choose_layer_to_grow()
        if layer_idx is None:
            return None

        # 뉴런 추가
        old_size = self.net.hidden_sizes[layer_idx]
        self.net.add_neuron_to_layer(layer_idx, self.growth_neurons)
        new_size = self.net.hidden_sizes[layer_idx]

        event = {
            'epoch': epoch,
            'layer_idx': layer_idx,
            'old_size': old_size,
            'new_size': new_size,
            'val_loss': val_loss,
        }
        self.growth_events.append(event)

        if self.verbose:
            print(
                f"  [Neurogenesis] epoch={epoch} | layer={layer_idx} | "
                f"{old_size} → {new_size} neurons | loss={val_loss:.4f}"
            )

        # 리셋
        self.epochs_since_improvement = 0
        self.cooldown_counter = self.cooldown
        self.recorder.activations.clear()  # 다음 기록 시작

        return layer_idx

    def _choose_layer_to_grow(self) -> Optional[int]:
        """
        포화도 기반으로 성장시킬 레이어 선택.
        포화도가 가장 높은 레이어 (분산이 낮은 레이어) 선택.
        최대 크기에 도달한 레이어는 제외.
        """
        if not self.recorder.activations:
            # hook 데이터 없으면 첫 번째 레이어
            return 0 if self.net.hidden_sizes[0] < self.max_neurons else None

        scores = self.recorder.get_saturation_scores()
        hidden_sizes = self.net.hidden_sizes

        # 최대 크기 미달인 레이어만 후보
        candidates = [
            (score, idx)
            for idx, (score, size) in enumerate(zip(scores, hidden_sizes))
            if size + self.growth_neurons <= self.max_neurons
        ]

        if not candidates:
            if self.verbose:
                print("  [Neurogenesis] 모든 레이어가 최대 크기에 도달. 성장 중단.")
            return None

        # 가장 포화도 높은 레이어 선택
        candidates.sort(reverse=True)
        return candidates[0][1]

    def get_summary(self) -> dict:
        return {
            'total_growth_events': len(self.growth_events),
            'final_architecture': self.net.hidden_sizes,
            'total_params': self.net.total_params,
            'growth_history': self.growth_events,
        }


class GraphNeurogenesisController:
    """
    GraphNet용 뉴로제네시스 컨트롤러.

    정체 감지 후 노드를 추가하고, 휴리스틱으로 연결을 결정:
    - incoming: 활성화 분산이 높은 top-K 노드에서 수신 (정보가 풍부한 노드)
    - outgoing: 모든 출력 노드에 연결 (즉시 기여)
    """

    def __init__(
        self,
        net,  # GraphNet
        patience: int = 5,
        min_delta: float = 1e-4,
        max_hidden: int = 256,
        growth_neurons: int = 8,
        growth_cooldown: int = 3,
        top_k_incoming: int = 16,
        verbose: bool = True,
        prune_var_threshold: float = 1e-6,
        prune_contrib_threshold: float = 0.01,
    ):
        self.net = net
        self.patience = patience
        self.min_delta = min_delta
        self.max_hidden = max_hidden
        self.growth_neurons = growth_neurons
        self.cooldown = growth_cooldown
        self.top_k_incoming = top_k_incoming
        self.verbose = verbose
        self.prune_var_threshold = prune_var_threshold
        self.prune_contrib_threshold = prune_contrib_threshold

        self.loss_history: deque = deque(maxlen=patience + 1)
        self.best_loss: float = float('inf')
        self.epochs_since_improvement: int = 0
        self.cooldown_counter: int = 0
        self.growth_events: List[dict] = []
        self.prune_events: List[dict] = []

        # 노드별 활성화 누적 (분산 계산용)
        self._activation_buffer: List[torch.Tensor] = []

    def record_activations(self):
        """
        매 배치 후 호출하여 노드별 활성화 기록.
        GraphNet.last_activations에서 가져옴.
        """
        if self.net.last_activations is not None:
            self._activation_buffer.append(self.net.last_activations.cpu())

    def _find_prunable_nodes(self) -> List[int]:
        """
        비효율적 HIDDEN 노드를 식별.

        기준:
        1. Dead neuron: 모든 샘플에서 활성화가 0
        2. Low variance: 활성화 분산이 임계값 미만
        3. Low contribution: 나가는 간선 가중치 절대값 합이 임계값 미만
        """
        net = self.net
        hidden_indices = (net.node_types == net.HIDDEN).nonzero(as_tuple=True)[0].tolist()
        if not hidden_indices:
            return []

        prunable = set()

        # 활성화 기반 분석 (dead neuron + low variance)
        if self._activation_buffer:
            all_acts = torch.cat(self._activation_buffer, dim=0)  # (total_samples, N)

            for idx in hidden_indices:
                acts = all_acts[:, idx]

                # Dead neuron: 전체 활성화 == 0
                if (acts == 0).all():
                    prunable.add(idx)
                    continue

                # Low variance
                if acts.var().item() < self.prune_var_threshold:
                    prunable.add(idx)

        # Low contribution: 나가는 간선 가중치 절대값 합
        W_eff = (net.W.data * net.mask).abs()
        for idx in hidden_indices:
            outgoing_sum = W_eff[idx, :].sum().item()
            if outgoing_sum < self.prune_contrib_threshold:
                prunable.add(idx)

        return sorted(prunable)

    def step(self, val_loss: float, epoch: int) -> Optional[int]:
        """
        매 epoch 끝에 호출. 노드를 추가했으면 추가된 노드 수 반환.

        Returns:
            int if growth occurred (number of added nodes), None otherwise
        """
        self.loss_history.append(val_loss)

        # 개선 여부 확인
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        # ── 프루닝: 쿨다운/정체와 무관하게 매 epoch 실행 ──
        prunable = self._find_prunable_nodes()
        if prunable:
            old_hidden = self.net.n_hidden
            old_params = self.net.total_params
            n_pruned = self.net.remove_nodes(prunable)
            if n_pruned > 0:
                event = {
                    'epoch': epoch,
                    'n_pruned': n_pruned,
                    'old_hidden': old_hidden,
                    'new_hidden': self.net.n_hidden,
                    'old_params': old_params,
                    'new_params': self.net.total_params,
                    'val_loss': val_loss,
                }
                self.prune_events.append(event)
                if self.verbose:
                    print(
                        f"  [Pruning] epoch={epoch} | "
                        f"-{n_pruned} nodes | "
                        f"hidden: {old_hidden} → {self.net.n_hidden} | "
                        f"edges: {self.net.n_edges} | "
                        f"loss={val_loss:.4f}"
                    )

        # 쿨다운 중
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self._activation_buffer.clear()
            return None

        # 정체 감지
        is_stagnated = (
            self.epochs_since_improvement >= self.patience
            and len(self.loss_history) >= self.patience
        )

        if not is_stagnated:
            self._activation_buffer.clear()
            return None

        # 최대 히든 노드 수 체크
        if self.net.n_hidden + self.growth_neurons > self.max_hidden:
            if self.verbose:
                print(f"  [GraphNeurogenesis] 최대 히든 노드 수({self.max_hidden})에 도달. 성장 중단.")
            self._activation_buffer.clear()
            return None

        # 연결 결정
        connect_from_list, connect_to_list = self._choose_connections()

        old_hidden = self.net.n_hidden
        old_params = self.net.total_params

        # 노드 추가
        new_indices = self.net.add_nodes(
            n=self.growth_neurons,
            connect_from_list=connect_from_list,
            connect_to_list=connect_to_list,
            init_std=0.01,
        )

        event = {
            'epoch': epoch,
            'n_added': self.growth_neurons,
            'old_hidden': old_hidden,
            'new_hidden': self.net.n_hidden,
            'old_params': old_params,
            'new_params': self.net.total_params,
            'val_loss': val_loss,
            'new_node_indices': new_indices,
        }
        self.growth_events.append(event)

        if self.verbose:
            print(
                f"  [GraphNeurogenesis] epoch={epoch} | "
                f"+{self.growth_neurons} nodes | "
                f"hidden: {old_hidden} → {self.net.n_hidden} | "
                f"edges: {self.net.n_edges} | "
                f"loss={val_loss:.4f}"
            )

        # 리셋
        self.epochs_since_improvement = 0
        self.cooldown_counter = self.cooldown
        self._activation_buffer.clear()

        return self.growth_neurons

    def _choose_connections(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        휴리스틱 연결 규칙:
        - incoming: 활성화 분산이 높은 top-K 비출력 노드에서 수신
        - outgoing: 모든 출력 노드에 연결
        """
        net = self.net
        output_indices = (net.node_types == net.OUTPUT).nonzero(as_tuple=True)[0].tolist()
        non_output_indices = (net.node_types != net.OUTPUT).nonzero(as_tuple=True)[0].tolist()

        # 활성화 분산 계산
        if self._activation_buffer:
            all_acts = torch.cat(self._activation_buffer, dim=0)  # (total_samples, N)
            # 노드별 분산 (샘플 차원에서)
            var_per_node = all_acts.var(dim=0)  # (N,)

            # 비출력 노드 중 분산이 높은 top-K 선택
            non_output_vars = [(idx, var_per_node[idx].item()) for idx in non_output_indices]
            non_output_vars.sort(key=lambda x: x[1], reverse=True)
            k = min(self.top_k_incoming, len(non_output_vars))
            top_k_nodes = [idx for idx, _ in non_output_vars[:k]]
        else:
            # 활성화 데이터 없으면 입력 노드에서만 연결
            top_k_nodes = (net.node_types == net.INPUT).nonzero(as_tuple=True)[0].tolist()

        # 각 새 노드마다 동일한 연결 구조
        connect_from_list = [top_k_nodes for _ in range(self.growth_neurons)]
        connect_to_list = [output_indices for _ in range(self.growth_neurons)]

        return connect_from_list, connect_to_list

    def get_summary(self) -> dict:
        return {
            'total_growth_events': len(self.growth_events),
            'total_prune_events': len(self.prune_events),
            'final_hidden': self.net.n_hidden,
            'final_nodes': self.net.n_nodes,
            'final_edges': self.net.n_edges,
            'total_params': self.net.total_params,
            'growth_history': self.growth_events,
            'prune_history': self.prune_events,
        }
