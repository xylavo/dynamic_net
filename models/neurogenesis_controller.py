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
        patience: int = 2,
        min_delta: float = 1e-4,
        max_hidden: int = 256,
        growth_neurons: int = 8,
        growth_cooldown: int = 1,
        incoming_k: int = 8,
        edges_per_epoch: int = 32,
        edge_candidates: int = 200,
        prune_grace_epochs: int = 3,
        edge_prune_threshold: float = 0.001,
        verbose: bool = True,
        prune_var_threshold: float = 1e-6,
        prune_contrib_threshold: float = 0.01,
        rl_controller=None,
    ):
        self.net = net
        self.patience = patience
        self.min_delta = min_delta
        self.max_hidden = max_hidden
        self.growth_neurons = growth_neurons
        self.cooldown = growth_cooldown
        self.incoming_k = incoming_k
        self.edges_per_epoch = edges_per_epoch
        self.edge_candidates = edge_candidates
        self.prune_grace_epochs = prune_grace_epochs
        self.edge_prune_threshold = edge_prune_threshold
        self.verbose = verbose
        self.prune_var_threshold = prune_var_threshold
        self.prune_contrib_threshold = prune_contrib_threshold
        self.rl_controller = rl_controller

        self.loss_history: deque = deque(maxlen=patience + 1)
        self.best_loss: float = float('inf')
        self.epochs_since_improvement: int = 0
        self.cooldown_counter: int = 0
        self.growth_events: List[dict] = []
        self.prune_events: List[dict] = []
        self.edge_events: List[dict] = []
        self.edge_prune_events: List[dict] = []

        # 노드별 탄생 epoch 추적 (grace period용)
        self._node_birth_epoch: dict = {}

        # 노드별 활성화 누적 (분산 계산용)
        self._activation_buffer: List[torch.Tensor] = []

    def record_activations(self):
        """
        매 배치 후 호출하여 노드별 활성화 기록.
        GraphNet.last_activations에서 가져옴.
        """
        if self.net.last_activations is not None:
            self._activation_buffer.append(self.net.last_activations.cpu())

    def _find_prunable_nodes(self, current_epoch: int) -> List[int]:
        """
        비효율적 HIDDEN 노드를 식별.

        기준:
        1. Dead neuron: 모든 샘플에서 활성화가 0
        2. Low variance: 활성화 분산이 임계값 미만
        3. Low contribution: 나가는 간선 가중치 절대값 합이 임계값 미만

        Grace period 내 노드는 제외.
        """
        net = self.net
        N = net.n_nodes
        hidden_indices = (net.node_types[:N] == net.HIDDEN).nonzero(as_tuple=True)[0].tolist()
        if not hidden_indices:
            return []

        # Grace period 내 노드 제외
        eligible = [
            idx for idx in hidden_indices
            if current_epoch - self._node_birth_epoch.get(idx, -999) >= self.prune_grace_epochs
        ]
        if not eligible:
            return []

        eligible_t = torch.tensor(eligible, dtype=torch.long)
        prunable_mask = torch.zeros(eligible_t.shape[0], dtype=torch.bool)

        # 활성화 기반 분석 (dead neuron + low variance) — 벡터화
        if self._activation_buffer:
            all_acts = torch.cat(self._activation_buffer, dim=0)  # (total_samples, N)
            eligible_acts = all_acts[:, eligible_t]  # (samples, K)

            # Dead neuron: 전체 활성화 == 0
            dead = (eligible_acts == 0).all(dim=0)  # (K,)
            prunable_mask |= dead

            # Low variance
            var_per = eligible_acts.var(dim=0)  # (K,)
            prunable_mask |= (var_per < self.prune_var_threshold)

        # Low contribution: 나가는 간선 가중치 절대값 합 — 벡터화
        W_eff = (net.W.data * net.mask).abs()
        eligible_gpu = eligible_t.to(net.W.device)
        outgoing_sums = W_eff[eligible_gpu, :].sum(dim=1).cpu()  # (K,)
        prunable_mask |= (outgoing_sums < self.prune_contrib_threshold)

        return eligible_t[prunable_mask].tolist()

    def step(self, val_loss: float, epoch: int, val_acc: float = None) -> Optional[int]:
        """
        매 epoch 끝에 호출. 노드를 추가했으면 추가된 노드 수 반환.

        Args:
            val_loss: 검증 손실
            epoch: 현재 epoch
            val_acc: 검증 정확도 (RL 컨트롤러 사용 시 필요)

        Returns:
            int if growth occurred (number of added nodes), None otherwise
        """
        # RL 컨트롤러: epoch 시작 시 활성화 통계 사전계산
        if self.rl_controller is not None:
            self.rl_controller.precompute_activation_stats(self._activation_buffer)

        self.loss_history.append(val_loss)

        # 개선 여부 확인
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        # ── 프루닝: 쿨다운/정체와 무관하게 매 epoch 실행 ──
        pruned_this_epoch = False
        prunable = self._find_prunable_nodes(epoch)
        if prunable:
            old_hidden = self.net.n_hidden
            old_params = self.net.total_params
            # 프루닝 전 인덱스 매핑 준비
            old_N = self.net.n_nodes
            remove_set = set(prunable)
            n_pruned = self.net.remove_nodes(prunable)
            if n_pruned > 0:
                # birth epoch 재매핑: 제거된 노드 삭제 + 인덱스 시프트
                keep_indices = [i for i in range(old_N) if i not in remove_set]
                new_birth = {}
                for new_idx, old_idx in enumerate(keep_indices):
                    if old_idx in self._node_birth_epoch:
                        new_birth[new_idx] = self._node_birth_epoch[old_idx]
                self._node_birth_epoch = new_birth

                pruned_this_epoch = True
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

        # ── 간선 프루닝: 매 epoch 실행 ──
        self._prune_edges_step(epoch)

        # ── 간선 추가: 노드 프루닝 미발생 시 매 epoch 실행 ──
        if not pruned_this_epoch:
            self._add_edges_step(epoch)

        # 쿨다운 중
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self._activation_buffer.clear()
            self._rl_record_and_update(val_acc, epoch)
            return None

        # 정체 감지
        is_stagnated = (
            self.epochs_since_improvement >= self.patience
            and len(self.loss_history) >= self.patience
        )

        if not is_stagnated:
            self._activation_buffer.clear()
            self._rl_record_and_update(val_acc, epoch)
            return None

        # 최대 히든 노드 수 체크
        if self.net.n_hidden + self.growth_neurons > self.max_hidden:
            if self.verbose:
                print(f"  [GraphNeurogenesis] 최대 히든 노드 수({self.max_hidden})에 도달. 성장 중단.")
            self._activation_buffer.clear()
            self._rl_record_and_update(val_acc, epoch)
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

        # Birth epoch 기록
        for idx in new_indices:
            self._node_birth_epoch[idx] = epoch

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

        self._rl_record_and_update(val_acc, epoch)

        return self.growth_neurons

    def _rl_record_and_update(self, val_acc, epoch):
        """RL 컨트롤러에 epoch 결과 기록 및 정책 업데이트."""
        if self.rl_controller is not None and val_acc is not None:
            self.rl_controller.record_epoch_result(val_acc, self.net.total_params)
            self.rl_controller.maybe_update_policy(epoch)

    def _choose_connections(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        K:M 연결 규칙:
        - rl_controller 있으면 → RL 결정으로 위임
        - 없으면 → 기존 휴리스틱 (활성화 분산 기반)
        """
        # RL 위임
        if self.rl_controller is not None:
            new_indices = list(range(self.net.n_nodes, self.net.n_nodes + self.growth_neurons))
            return self.rl_controller.decide_growth_connections(
                new_node_indices=new_indices,
                n_growth=self.growth_neurons,
            )

        net = self.net
        N = net.n_nodes
        output_indices = (net.node_types[:N] == net.OUTPUT).nonzero(as_tuple=True)[0].tolist()
        non_output_indices = (net.node_types[:N] != net.OUTPUT).nonzero(as_tuple=True)[0].tolist()

        n = self.growth_neurons
        k = min(self.incoming_k, len(non_output_indices))
        n_out = len(output_indices)

        # ── Incoming: 분산 기반 확률적 K개 샘플링 ──
        if self._activation_buffer and non_output_indices:
            all_acts = torch.cat(self._activation_buffer, dim=0)
            var_per_node = all_acts.var(dim=0)

            non_output_t = torch.tensor(non_output_indices, dtype=torch.long)
            variances = var_per_node[non_output_t]
            probs = variances + 1e-8
            probs = probs / probs.sum()

            connect_from_list = []
            for _ in range(n):
                sampled = torch.multinomial(probs, num_samples=k, replacement=False)
                connect_from_list.append([non_output_indices[s.item()] for s in sampled])
        else:
            input_indices = (net.node_types[:N] == net.INPUT).nonzero(as_tuple=True)[0].tolist()
            k_in = min(k, len(input_indices))
            connect_from_list = []
            for _ in range(n):
                perm = torch.randperm(len(input_indices))[:k_in]
                connect_from_list.append([input_indices[p.item()] for p in perm])

        # ── Outgoing: 노드마다 서로 다른 출력 부분집합 ──
        # 각 노드에 ceil(n_out/2)~n_out 개 출력 연결 (다양성 + 충분한 기여)
        min_out = max(1, (n_out + 1) // 2)
        connect_to_list = []
        for i in range(n):
            n_connect = min_out + (i % (n_out - min_out + 1))
            perm = torch.randperm(n_out)[:n_connect]
            connect_to_list.append([output_indices[p.item()] for p in perm])

        return connect_from_list, connect_to_list

    def _prune_edges_step(self, epoch: int) -> int:
        """
        매 epoch 약한 간선 제거. 노드 고립 방지 보장.

        Returns:
            제거된 간선 수
        """
        old_edges = self.net.n_edges

        # RL 위임
        if self.rl_controller is not None:
            edges_to_remove = self.rl_controller.decide_edge_pruning(current_epoch=epoch)
            if edges_to_remove:
                n_removed = self.net.remove_edges(
                    edges_to_remove, min_outgoing=1, min_incoming=1
                )
            else:
                n_removed = 0
        else:
            n_removed = self.net.prune_edges(
                threshold=self.edge_prune_threshold,
                min_outgoing=1,
                min_incoming=1,
            )

        if n_removed > 0:
            event = {
                'epoch': epoch,
                'n_removed': n_removed,
                'old_edges': old_edges,
                'new_edges': self.net.n_edges,
            }
            self.edge_prune_events.append(event)
            if self.verbose:
                print(
                    f"  [EdgePrune] epoch={epoch} | "
                    f"-{n_removed} edges | "
                    f"edges: {old_edges} → {self.net.n_edges}"
                )

        return n_removed

    def _add_edges_step(self, epoch: int) -> int:
        """
        매 epoch 간선 추가.
        - rl_controller 있으면 → RL 결정으로 위임
        - 없으면 → 기존 분산 점수 기반 로직

        Returns:
            추가된 간선 수
        """
        net = self.net

        # RL 위임
        if self.rl_controller is not None:
            edges_to_add = self.rl_controller.decide_edge_additions()
            if edges_to_add:
                n_added = net.add_edges(edges_to_add)
                if n_added > 0:
                    event = {
                        'epoch': epoch,
                        'n_added': n_added,
                        'n_edges': net.n_edges,
                        'candidates': len(edges_to_add),
                    }
                    self.edge_events.append(event)
                    if self.verbose:
                        print(
                            f"  [RL EdgeAdd] epoch={epoch} | "
                            f"+{n_added} edges | "
                            f"total_edges: {net.n_edges}"
                        )
                return n_added
            return 0

        if not self._activation_buffer:
            return 0

        N = net.n_nodes
        non_output_t = (net.node_types[:N] != net.OUTPUT).nonzero(as_tuple=True)[0]
        non_input_t = (net.node_types[:N] != net.INPUT).nonzero(as_tuple=True)[0]

        if non_output_t.shape[0] == 0 or non_input_t.shape[0] == 0:
            return 0

        all_acts = torch.cat(self._activation_buffer, dim=0)
        var_per_node = all_acts.var(dim=0).to(net.W.device)

        # 벡터화된 랜덤 후보 쌍 생성
        n_cand = min(self.edge_candidates, non_output_t.shape[0] * non_input_t.shape[0])
        src_idx = non_output_t[torch.randint(non_output_t.shape[0], (n_cand,))]
        tgt_idx = non_input_t[torch.randint(non_input_t.shape[0], (n_cand,))]

        # 벡터화된 필터링
        node_levels = net._node_levels.to(src_idx.device)
        valid = (src_idx != tgt_idx)
        valid &= (net.mask[src_idx, tgt_idx] == 0)
        valid &= (node_levels[src_idx] < node_levels[tgt_idx])

        src_valid = src_idx[valid]
        tgt_valid = tgt_idx[valid]

        if src_valid.shape[0] == 0:
            return 0

        # 벡터화된 점수 계산 및 top-K 선택
        scores = var_per_node[src_valid] * var_per_node[tgt_valid]
        k = min(self.edges_per_epoch, scores.shape[0])
        top_k_idx = scores.topk(k).indices
        top_edges = list(zip(src_valid[top_k_idx].tolist(), tgt_valid[top_k_idx].tolist()))

        n_added = net.add_edges(top_edges)

        if n_added > 0:
            n_candidates = src_valid.shape[0]
            event = {
                'epoch': epoch,
                'n_added': n_added,
                'n_edges': net.n_edges,
                'candidates': n_candidates,
            }
            self.edge_events.append(event)
            if self.verbose:
                print(
                    f"  [EdgeAdd] epoch={epoch} | "
                    f"+{n_added} edges | "
                    f"total_edges: {net.n_edges} | "
                    f"candidates: {n_candidates}"
                )

        return n_added

    def get_summary(self) -> dict:
        return {
            'total_growth_events': len(self.growth_events),
            'total_prune_events': len(self.prune_events),
            'total_edge_events': len(self.edge_events),
            'total_edge_prune_events': len(self.edge_prune_events),
            'final_hidden': self.net.n_hidden,
            'final_nodes': self.net.n_nodes,
            'final_edges': self.net.n_edges,
            'total_params': self.net.total_params,
            'growth_history': self.growth_events,
            'prune_history': self.prune_events,
            'edge_history': self.edge_events,
            'edge_prune_history': self.edge_prune_events,
        }
