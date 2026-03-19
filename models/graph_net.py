"""
GraphNet: 가중치 인접 행렬 기반 동적 뉴로제네시스 네트워크

핵심 아이디어:
- 레이어 개념 없이 개별 노드를 동적으로 생성하고 자유롭게 연결
- W (N×N 인접 행렬)로 연결 구조와 가중치를 동시에 표현
- mask (N×N 이진 행렬)로 연결 존재 여부를 분리 관리
- DAG 제약: 위상 순서를 유지하여 순환 방지
- Forward pass: 위상 정렬 레벨별 행렬 연산으로 GPU 병렬화
- Capacity-doubling: 텐서를 2배씩 확장하여 리사이징 오버헤드 amortize
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from collections import deque


class GraphNet(nn.Module):
    """
    가중치 인접 행렬 기반 동적 그래프 네트워크.

    노드 종류: INPUT(0) | HIDDEN(1) | OUTPUT(2)
    W[i][j] = 노드 i → 노드 j 간선 가중치
    mask[i][j] = 연결 존재 여부

    텐서는 capacity 크기로 할당되고, 실제 노드 수(_n_nodes)만 사용.
    capacity 부족 시 2배로 확장 (amortized O(1)).
    """

    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2

    def __init__(self, n_inputs: int, n_outputs: int, initial_hidden: int = 0):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        n_total = n_inputs + initial_hidden + n_outputs
        self._n_nodes = n_total
        self._capacity = n_total

        # 노드 종류 추적
        node_types = (
            [self.INPUT] * n_inputs
            + [self.HIDDEN] * initial_hidden
            + [self.OUTPUT] * n_outputs
        )
        self.register_buffer('node_types', torch.tensor(node_types, dtype=torch.long))

        # 가중치 인접 행렬과 마스크
        self.W = nn.Parameter(torch.zeros(n_total, n_total))
        self.register_buffer('mask', torch.zeros(n_total, n_total))

        # 노드별 바이어스 (입력 노드는 바이어스 불필요하지만 통일성을 위해 유지)
        self.bias = nn.Parameter(torch.zeros(n_total))

        # 초기 연결: initial_hidden이 있으면 input→hidden→output
        # initial_hidden이 없으면 input→output 직접 연결
        self._init_connections(n_inputs, initial_hidden, n_outputs)

        # 위상 레벨 계산
        self._levels: List[List[int]] = []
        self._compute_levels()

        # 성장 기록
        self.growth_history: List[dict] = []

        # 노드별 활성화 저장 (neurogenesis controller가 사용)
        self.last_activations: Optional[torch.Tensor] = None

    def _init_connections(self, n_inputs: int, n_hidden: int, n_outputs: int):
        """초기 연결 구조 설정."""
        with torch.no_grad():
            if n_hidden > 0:
                # input → hidden
                for i in range(n_inputs):
                    for h in range(n_inputs, n_inputs + n_hidden):
                        self.mask[i, h] = 1.0
                # hidden → output
                for h in range(n_inputs, n_inputs + n_hidden):
                    for o in range(n_inputs + n_hidden, n_inputs + n_hidden + n_outputs):
                        self.mask[h, o] = 1.0
                # Xavier-like 초기화
                fan_in_h = n_inputs
                fan_out_h = n_outputs
                std_h = (2.0 / (fan_in_h + fan_out_h)) ** 0.5
                self.W.data[:n_inputs, n_inputs:n_inputs + n_hidden].normal_(0, std_h)
                self.W.data[n_inputs:n_inputs + n_hidden, n_inputs + n_hidden:].normal_(0, std_h)
            else:
                # input → output 직접 연결
                for i in range(n_inputs):
                    for o in range(n_inputs, n_inputs + n_outputs):
                        self.mask[i, o] = 1.0
                std = (2.0 / (n_inputs + n_outputs)) ** 0.5
                self.W.data[:n_inputs, n_inputs:n_inputs + n_outputs].normal_(0, std)

    def _grow_capacity(self, min_capacity: int):
        """Capacity를 2배씩 늘려서 min_capacity 이상으로 확장."""
        new_cap = self._capacity
        while new_cap < min_capacity:
            new_cap *= 2

        if new_cap == self._capacity:
            return

        N = self._n_nodes
        device = self.W.device
        dtype = self.W.dtype

        # W 확장
        new_W = torch.zeros(new_cap, new_cap, device=device, dtype=dtype)
        new_W[:N, :N] = self.W.data[:N, :N]
        self.W = nn.Parameter(new_W)

        # mask 확장
        new_mask = torch.zeros(new_cap, new_cap, device=self.mask.device, dtype=self.mask.dtype)
        new_mask[:N, :N] = self.mask[:N, :N]
        self.register_buffer('mask', new_mask)

        # bias 확장
        new_bias = torch.zeros(new_cap, device=device, dtype=dtype)
        new_bias[:N] = self.bias.data[:N]
        self.bias = nn.Parameter(new_bias)

        # node_types 확장 (빈 슬롯은 -1로 초기화하여 INPUT/HIDDEN/OUTPUT과 구분)
        new_types = torch.full((new_cap,), -1, device=self.node_types.device, dtype=self.node_types.dtype)
        new_types[:N] = self.node_types[:N]
        self.register_buffer('node_types', new_types)

        self._capacity = new_cap

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    @property
    def n_hidden(self) -> int:
        return int((self.node_types[:self._n_nodes] == self.HIDDEN).sum().item())

    @property
    def total_params(self) -> int:
        """유효 파라미터 수: 활성 간선(mask=1) 가중치 + 비입력 노드 바이어스."""
        N = self._n_nodes
        n_active_weights = int(self.mask[:N, :N].sum().item())
        n_biases = int((self.node_types[:N] != self.INPUT).sum().item())
        return n_active_weights + n_biases

    @property
    def n_edges(self) -> int:
        N = self._n_nodes
        return int(self.mask[:N, :N].sum().item())

    def _compute_levels(self):
        """
        Kahn's 알고리즘 변형으로 위상 레벨 계산 (longest path 기반).
        각 노드를 가능한 한 늦은 레벨에 배치하여 정보 전파를 최대화.
        """
        N = self._n_nodes
        adj = self.mask[:N, :N].detach()
        node_types = self.node_types[:N]

        # longest path 기반 레벨 할당
        level = torch.zeros(N, dtype=torch.long)

        # 입력 노드는 레벨 0
        input_mask = (node_types == self.INPUT)
        level[input_mask] = 0

        # BFS로 longest path 계산
        # in_degree 계산
        in_degree = adj.sum(dim=0).long()  # 각 노드로 들어오는 간선 수

        queue = deque()
        for i in range(N):
            if node_types[i] == self.INPUT:
                queue.append(i)

        visited_count = torch.zeros(N, dtype=torch.long)

        while queue:
            node = queue.popleft()
            for j in range(N):
                if adj[node, j] > 0:
                    # longest path: 최대 레벨 + 1
                    new_level = level[node].item() + 1
                    if new_level > level[j].item():
                        level[j] = new_level
                    visited_count[j] += 1
                    if visited_count[j] >= in_degree[j]:
                        queue.append(j)

        # 레벨별 노드 그룹화
        max_level = level.max().item()
        self._levels = []
        for lv in range(max_level + 1):
            nodes_at_level = (level == lv).nonzero(as_tuple=True)[0].tolist()
            if nodes_at_level:
                self._levels.append(nodes_at_level)

        # 레벨 정보 저장 (노드 추가 시 DAG 검증에 사용)
        self._node_levels = level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        레벨별 행렬 연산으로 forward pass.

        Args:
            x: (batch, n_inputs) 입력 텐서

        Returns:
            (batch, n_outputs) 출력 텐서
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        batch_size = x.size(0)
        N = self._n_nodes

        # 유효 가중치 (마스크 적용, 활성 노드 영역만)
        W_eff = self.W[:N, :N] * self.mask[:N, :N]

        # 모든 노드의 활성화를 저장할 텐서
        h = torch.zeros(batch_size, N, device=x.device, dtype=x.dtype)

        # 레벨 0: 입력 노드에 데이터 할당
        input_indices = (self.node_types[:N] == self.INPUT).nonzero(as_tuple=True)[0]
        h[:, input_indices] = x

        # 레벨 1부터: 이전 레벨들의 활성화를 기반으로 계산
        for level_nodes in self._levels[1:]:
            idx = torch.tensor(level_nodes, device=x.device, dtype=torch.long)
            # h @ W_eff[:, idx] → (batch, len(idx))
            pre_act = h @ W_eff[:, idx] + self.bias[idx]

            # 출력 노드는 활성화 함수 없이 raw logits
            is_output = torch.tensor(
                [self.node_types[n] == self.OUTPUT for n in level_nodes],
                device=x.device
            )
            activated = F.relu(pre_act)
            # 출력 노드는 raw, 히든 노드는 ReLU
            result = torch.where(is_output.unsqueeze(0), pre_act, activated)
            # clone으로 inplace 연산 충돌 방지 (autograd 호환)
            h = h.clone()
            h[:, idx] = result

        # 노드별 활성화 저장 (neurogenesis controller용)
        self.last_activations = h.detach()

        # 출력 노드만 추출
        output_indices = (self.node_types[:N] == self.OUTPUT).nonzero(as_tuple=True)[0]
        return h[:, output_indices]

    def add_node(
        self,
        connect_from: List[int],
        connect_to: List[int],
        init_std: float = 0.01,
    ) -> int:
        """
        히든 노드 1개 추가.

        Args:
            connect_from: 이 노드로 들어오는 연결의 소스 노드 인덱스
            connect_to: 이 노드에서 나가는 연결의 타겟 노드 인덱스
            init_std: 새 가중치 초기화 표준편차

        Returns:
            새 노드의 인덱스
        """
        N = self._n_nodes
        new_N = N + 1
        new_idx = N

        # Capacity 부족 시 2배 확장
        if new_N > self._capacity:
            self._grow_capacity(new_N)

        # 노드 타입 설정
        self.node_types[new_idx] = self.HIDDEN
        self._n_nodes = new_N

        # 연결 설정
        with torch.no_grad():
            for src in connect_from:
                self.mask[src, new_idx] = 1.0
                self.W.data[src, new_idx] = torch.randn(1).item() * init_std
            for tgt in connect_to:
                self.mask[new_idx, tgt] = 1.0
                self.W.data[new_idx, tgt] = torch.randn(1).item() * init_std

        # 위상 레벨 재계산
        self._compute_levels()

        return new_idx

    def add_nodes(
        self,
        n: int,
        connect_from_list: Optional[List[List[int]]] = None,
        connect_to_list: Optional[List[List[int]]] = None,
        init_std: float = 0.01,
    ) -> List[int]:
        """
        여러 노드를 한번에 추가.
        Capacity 부족 시 2배씩 확장 (amortized O(1)).

        Args:
            n: 추가할 노드 수
            connect_from_list: 각 노드별 incoming 연결 소스 리스트
            connect_to_list: 각 노드별 outgoing 연결 타겟 리스트
            init_std: 새 가중치 초기화 표준편차

        Returns:
            새 노드들의 인덱스 리스트
        """
        N = self._n_nodes
        new_N = N + n
        new_indices = list(range(N, new_N))

        # Capacity 부족 시 2배 확장
        if new_N > self._capacity:
            self._grow_capacity(new_N)

        # 노드 타입 설정
        self.node_types[N:new_N] = self.HIDDEN
        self._n_nodes = new_N

        # 연결 설정
        with torch.no_grad():
            for i, new_idx in enumerate(new_indices):
                if connect_from_list is not None and i < len(connect_from_list):
                    for src in connect_from_list[i]:
                        self.mask[src, new_idx] = 1.0
                        self.W.data[src, new_idx] = torch.randn(1).item() * init_std
                if connect_to_list is not None and i < len(connect_to_list):
                    for tgt in connect_to_list[i]:
                        self.mask[new_idx, tgt] = 1.0
                        self.W.data[new_idx, tgt] = torch.randn(1).item() * init_std

        # 위상 레벨 재계산
        self._compute_levels()

        return new_indices

    def add_edges(self, edge_list: List[Tuple[int, int]], init_std: float = 0.01) -> int:
        """
        기존 노드 간 간선 추가.

        Args:
            edge_list: (src, tgt) 튜플 리스트
            init_std: 새 가중치 초기화 표준편차

        Returns:
            실제로 추가된 간선 수
        """
        N = self._n_nodes
        added = 0
        with torch.no_grad():
            for src, tgt in edge_list:
                # 범위 검사
                if src < 0 or src >= N or tgt < 0 or tgt >= N:
                    continue
                # 자기 연결 스킵
                if src == tgt:
                    continue
                # 기존 간선 스킵
                if self.mask[src, tgt] > 0:
                    continue
                # INPUT 노드로 들어오는 간선 스킵
                if self.node_types[tgt] == self.INPUT:
                    continue
                # OUTPUT 노드에서 나가는 간선 스킵
                if self.node_types[src] == self.OUTPUT:
                    continue
                # DAG 제약: src 레벨 < tgt 레벨
                if self._node_levels[src] >= self._node_levels[tgt]:
                    continue

                self.mask[src, tgt] = 1.0
                self.W.data[src, tgt] = torch.randn(1).item() * init_std
                added += 1

        return added

    def prune_edges(self, threshold: float = 0.001, min_outgoing: int = 1, min_incoming: int = 1) -> int:
        """
        가중치 절대값이 임계값 미만인 간선 제거 (벡터화).
        노드 고립 방지를 위해 최소 incoming/outgoing 간선 수 보장.

        Args:
            threshold: 이 값 미만의 |W|인 간선을 제거 후보로 선정
            min_outgoing: 각 비출력 노드가 유지해야 할 최소 outgoing 간선 수
            min_incoming: 각 비입력 노드가 유지해야 할 최소 incoming 간선 수

        Returns:
            제거된 간선 수
        """
        N = self._n_nodes
        with torch.no_grad():
            W_abs = (self.W.data[:N, :N] * self.mask[:N, :N]).abs()

            # 후보: mask > 0 이고 |W| < threshold
            candidate_mask = (self.mask[:N, :N] > 0) & (W_abs < threshold)
            if not candidate_mask.any():
                return 0

            # 현재 degree 계산
            out_degree = self.mask[:N, :N].sum(dim=1)  # (N,)
            in_degree = self.mask[:N, :N].sum(dim=0)   # (N,)

            # 후보 간선의 (src, tgt) 인덱스
            candidates = candidate_mask.nonzero(as_tuple=False)  # (K, 2)

            # |W|가 작은 순서대로 정렬 (가장 약한 간선부터 제거)
            cand_weights = W_abs[candidates[:, 0], candidates[:, 1]]
            sorted_idx = cand_weights.argsort()
            candidates = candidates[sorted_idx]

            removed = 0
            for i in range(candidates.shape[0]):
                s, t = candidates[i, 0].item(), candidates[i, 1].item()
                if out_degree[s] <= min_outgoing:
                    continue
                if in_degree[t] <= min_incoming:
                    continue
                self.mask[s, t] = 0.0
                self.W.data[s, t] = 0.0
                out_degree[s] -= 1
                in_degree[t] -= 1
                removed += 1

        return removed

    def remove_nodes(self, indices_to_remove: List[int]) -> int:
        """
        HIDDEN 노드를 제거. 기존 capacity 내에서 compact (재할당 없음).

        Args:
            indices_to_remove: 제거할 HIDDEN 노드 인덱스 리스트

        Returns:
            실제로 제거된 노드 수
        """
        if not indices_to_remove:
            return 0

        N = self._n_nodes

        # HIDDEN 노드만 제거 가능
        indices_to_remove = [
            i for i in indices_to_remove
            if i < N and self.node_types[i] == self.HIDDEN
        ]
        if not indices_to_remove:
            return 0

        remove_set = set(indices_to_remove)
        keep_indices = [i for i in range(N) if i not in remove_set]
        new_N = len(keep_indices)

        if new_N == N:
            return 0

        keep_idx = torch.tensor(keep_indices, device=self.W.device, dtype=torch.long)

        with torch.no_grad():
            # 유지할 데이터 추출 (clone으로 안전하게 복사)
            kept_W = self.W.data[keep_idx][:, keep_idx].clone()
            kept_mask = self.mask[keep_idx][:, keep_idx].clone()
            kept_bias = self.bias.data[keep_idx].clone()
            kept_types = self.node_types[keep_idx].clone()

            # Compact: 앞쪽에 데이터 배치, 나머지 영역 0으로 초기화
            self.W.data[:new_N, :new_N] = kept_W
            self.W.data[new_N:, :] = 0
            self.W.data[:new_N, new_N:] = 0

            self.mask[:new_N, :new_N] = kept_mask
            self.mask[new_N:, :] = 0
            self.mask[:new_N, new_N:] = 0

            self.bias.data[:new_N] = kept_bias
            self.bias.data[new_N:] = 0

        self.node_types[:new_N] = kept_types
        self.node_types[new_N:] = -1

        self._n_nodes = new_N

        # 위상 레벨 재계산
        self._compute_levels()

        return N - new_N

    def get_graph_state(self) -> torch.Tensor:
        """
        그래프 상태 인코딩 (RL 호환용).

        Returns:
            (6,) tensor: [n_nodes_norm, n_hidden_norm, n_edges_norm, density, avg_level, max_level]
        """
        n = self._n_nodes
        max_possible_edges = n * (n - 1)
        state = torch.tensor([
            self.n_hidden / max(n, 1),
            self.n_edges / max(max_possible_edges, 1),
            len(self._levels) / max(n, 1),
            self.n_hidden / 256.0,  # 정규화 (max_neurons 기준)
            self.n_inputs / max(n, 1),
            self.n_outputs / max(n, 1),
        ], dtype=torch.float32)
        return state

    def remove_edges(self, edge_list: List[Tuple[int, int]],
                     min_outgoing: int = 1, min_incoming: int = 1) -> int:
        """
        특정 간선 리스트를 제거 (RL 프루닝 결정용).
        최소 연결성 보장 (min_outgoing/min_incoming).

        Args:
            edge_list: (src, tgt) 튜플 리스트
            min_outgoing: 각 비출력 노드가 유지해야 할 최소 outgoing 간선 수
            min_incoming: 각 비입력 노드가 유지해야 할 최소 incoming 간선 수

        Returns:
            실제로 제거된 간선 수
        """
        N = self._n_nodes
        removed = 0
        with torch.no_grad():
            for src, tgt in edge_list:
                if src < 0 or src >= N or tgt < 0 or tgt >= N:
                    continue
                if self.mask[src, tgt] == 0:
                    continue

                # 최소 outgoing 보장
                src_outgoing = int(self.mask[src, :N].sum().item())
                if src_outgoing <= min_outgoing:
                    continue

                # 최소 incoming 보장
                tgt_incoming = int(self.mask[:N, tgt].sum().item())
                if tgt_incoming <= min_incoming:
                    continue

                self.mask[src, tgt] = 0.0
                self.W.data[src, tgt] = 0.0
                removed += 1

        return removed

    def get_node_degrees(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        노드별 (out_degree, in_degree) 반환.

        Returns:
            (out_degree, in_degree): 각각 (N,) 텐서
        """
        N = self._n_nodes
        out_degree = self.mask[:N, :N].sum(dim=1)
        in_degree = self.mask[:N, :N].sum(dim=0)
        return out_degree, in_degree

    def _validate_dag(self) -> bool:
        """DAG 제약 조건 검증."""
        N = self._n_nodes
        adj = self.mask[:N, :N].detach()
        node_types = self.node_types[:N]

        # 자기 연결 금지
        if adj.diag().sum() > 0:
            return False

        # 입력→입력 연결 금지
        input_nodes = (node_types == self.INPUT).nonzero(as_tuple=True)[0]
        for i in input_nodes:
            for j in input_nodes:
                if adj[i, j] > 0:
                    return False

        # 출력 노드에서 나가는 연결 금지
        output_nodes = (node_types == self.OUTPUT).nonzero(as_tuple=True)[0]
        for o in output_nodes:
            if adj[o].sum() > 0:
                return False

        return True

    def __repr__(self) -> str:
        return (
            f"GraphNet(inputs={self.n_inputs}, hidden={self.n_hidden}, "
            f"outputs={self.n_outputs}, edges={self.n_edges}, "
            f"levels={len(self._levels)})"
        )
