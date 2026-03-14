"""
DynamicNet: PyTorch 네트워크에서 학습 중 노드를 삽입/삭제할 수 있는 모듈

핵심 아이디어:
- 각 레이어를 ModuleList로 관리해 런타임에 교체 가능
- 노드 삽입 시 기존 가중치를 보존하고 새 뉴런을 0에 가깝게 초기화
  (Net2Net 방식: 네트워크 기능을 유지하면서 확장)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import copy


class DynamicLayer(nn.Module):
    """크기를 동적으로 변경할 수 있는 Linear 레이어."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def expand_output(self, n_new: int = 1) -> 'DynamicLayer':
        """출력 뉴런 n_new개 추가 (다음 레이어의 입력도 같이 늘어야 함)."""
        old_w = self.linear.weight.data   # (out, in)
        old_b = self.linear.bias.data if self.linear.bias is not None else None

        new_out = self.out_features + n_new
        new_layer = nn.Linear(self.in_features, new_out, bias=(old_b is not None))

        # 기존 가중치 복사 (기능 보존)
        new_layer.weight.data[:self.out_features] = old_w
        # 새 뉴런은 작은 노이즈로 초기화 (Net2Net-style)
        nn.init.normal_(new_layer.weight.data[self.out_features:], mean=0, std=0.01)

        if old_b is not None:
            new_layer.bias.data[:self.out_features] = old_b
            new_layer.bias.data[self.out_features:] = 0.0

        self.linear = new_layer
        self.out_features = new_out
        return self

    def expand_input(self, n_new: int = 1) -> 'DynamicLayer':
        """입력 채널 n_new개 추가 (이전 레이어에서 출력 확장 후 호출)."""
        old_w = self.linear.weight.data   # (out, in)
        old_b = self.linear.bias.data if self.linear.bias is not None else None

        new_in = self.in_features + n_new
        new_layer = nn.Linear(new_in, self.out_features, bias=(old_b is not None))

        new_layer.weight.data[:, :self.in_features] = old_w
        # 새 입력 가중치는 0으로 (처음엔 새 뉴런 무시)
        new_layer.weight.data[:, self.in_features:] = 0.0

        if old_b is not None:
            new_layer.bias.data = old_b.clone()

        self.linear = new_layer
        self.in_features = new_in
        return self


class DynamicNet(nn.Module):
    """
    동적으로 노드를 추가할 수 있는 MLP.

    구조: Input → [Hidden Layers] → Output
    각 hidden layer는 독립적으로 크기 조절 가능.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout

        # 히든 레이어 구성
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(len(sizes) - 1):
            self.layers.append(DynamicLayer(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:  # 출력 레이어 직전까지 BN
                self.batch_norms.append(nn.BatchNorm1d(sizes[i + 1]))

        self.dropout = nn.Dropout(dropout)

        # 그래프 구조 추적 (RL 에이전트용 상태)
        self.growth_history: List[dict] = []

    @property
    def hidden_sizes(self) -> List[int]:
        """현재 히든 레이어 크기 반환."""
        return [self.layers[i].out_features for i in range(len(self.layers) - 1)]

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if needed (for image input)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if i < len(self.batch_norms):
                # BN은 배치 크기 > 1일 때만 (eval 중 배치 1도 처리)
                if x.size(0) > 1 or not self.training:
                    x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.layers[-1](x)
        return x

    def add_neuron_to_layer(self, layer_idx: int, n_new: int = 1) -> None:
        """
        layer_idx번 히든 레이어에 n_new개 뉴런 추가.
        연결된 다음 레이어의 입력도 자동으로 확장.
        """
        assert 0 <= layer_idx < len(self.layers) - 1, "히든 레이어 인덱스만 가능"

        old_size = self.layers[layer_idx].out_features

        # 1) 해당 레이어 출력 확장
        self.layers[layer_idx].expand_output(n_new)

        # 2) 다음 레이어 입력 확장
        self.layers[layer_idx + 1].expand_input(n_new)

        # 3) BatchNorm도 확장 (새 레이어로 교체)
        if layer_idx < len(self.batch_norms):
            new_size = old_size + n_new
            old_bn = self.batch_norms[layer_idx]
            new_bn = nn.BatchNorm1d(new_size)
            # 기존 통계 복사
            new_bn.weight.data[:old_size] = old_bn.weight.data
            new_bn.bias.data[:old_size] = old_bn.bias.data
            new_bn.running_mean[:old_size] = old_bn.running_mean
            new_bn.running_var[:old_size] = old_bn.running_var
            # 새 뉴런 초기화
            new_bn.weight.data[old_size:] = 1.0
            new_bn.bias.data[old_size:] = 0.0
            new_bn.running_mean[old_size:] = 0.0
            new_bn.running_var[old_size:] = 1.0
            self.batch_norms[layer_idx] = new_bn

        # 성장 기록
        self.growth_history.append({
            'layer_idx': layer_idx,
            'n_new': n_new,
            'old_size': old_size,
            'new_size': old_size + n_new,
        })

    def get_graph_state(self) -> torch.Tensor:
        """
        RL 에이전트용 그래프 상태 인코딩.
        Returns: (n_layers * 2,) tensor — [크기, 파라미터 비율]
        """
        sizes = self.hidden_sizes
        max_size = max(sizes) if sizes else 1
        state = []
        for s in sizes:
            state.append(s / max_size)             # 정규화된 크기
            state.append(s / (self.input_size + 1e-6))  # 입력 대비 비율
        return torch.tensor(state, dtype=torch.float32)

    def __repr__(self) -> str:
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        return f"DynamicNet({' → '.join(map(str, sizes))})"
