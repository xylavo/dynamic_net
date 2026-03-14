"""
GraphNet 구조 시각화: HIDDEN + OUTPUT 노드와 간선만 표시.
INPUT 노드는 연결 수를 요약하여 표시.

사용법: python visualize_graph.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from models.graph_net import GraphNet


def load_graphnet(path: str) -> GraphNet:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    n_inputs = checkpoint['n_inputs']
    n_outputs = checkpoint['n_outputs']
    state = checkpoint['state_dict']

    # node_types에서 실제 노드 수 파악
    node_types = state['node_types']
    n_hidden = int((node_types == GraphNet.HIDDEN).sum().item())

    net = GraphNet(n_inputs, n_outputs, initial_hidden=0)
    # 필요한 만큼 더미 노드 추가하여 크기 맞추기
    if n_hidden > 0:
        input_indices = list(range(n_inputs))
        output_indices = (node_types == GraphNet.OUTPUT).nonzero(as_tuple=True)[0].tolist()
        for _ in range(n_hidden):
            net.add_node(connect_from=[], connect_to=[], init_std=0.0)

    # state_dict 로드
    net.load_state_dict(state)
    return net


def visualize_graphnet(net: GraphNet, save_path: str = 'graph_structure.png'):
    W_eff = (net.W.data * net.mask).detach()
    N = net.n_nodes

    # HIDDEN, OUTPUT 노드만 추출
    hidden_indices = (net.node_types == net.HIDDEN).nonzero(as_tuple=True)[0].tolist()
    output_indices = (net.node_types == net.OUTPUT).nonzero(as_tuple=True)[0].tolist()
    input_indices = (net.node_types == net.INPUT).nonzero(as_tuple=True)[0].tolist()

    if not hidden_indices:
        print("HIDDEN 노드가 없어 시각화할 간선이 없습니다.")
        return

    # 표시할 노드: HIDDEN + OUTPUT
    display_nodes = hidden_indices + output_indices

    G = nx.DiGraph()

    # 노드 추가
    for idx in display_nodes:
        if net.node_types[idx] == net.HIDDEN:
            G.add_node(idx, ntype='hidden')
        else:
            G.add_node(idx, ntype='output')

    # 간선 추가 (display_nodes 사이의 간선만)
    display_set = set(display_nodes)
    edges_data = []
    for i in display_nodes:
        for j in display_nodes:
            w = W_eff[i, j].item()
            if net.mask[i, j].item() > 0 and abs(w) > 1e-8:
                G.add_edge(i, j, weight=w)
                edges_data.append((i, j, w))

    # INPUT → HIDDEN 연결 수 계산 (노드 라벨에 표시)
    input_conn_count = {}
    for h in hidden_indices:
        count = sum(1 for inp in input_indices if net.mask[inp, h].item() > 0)
        input_conn_count[h] = count

    # 레이아웃: 레벨 기반 (좌→우)
    pos = {}
    node_levels = net._node_levels.detach()

    # 레벨별 그룹화
    level_groups = {}
    for idx in display_nodes:
        lv = node_levels[idx].item()
        if lv not in level_groups:
            level_groups[lv] = []
        level_groups[lv].append(idx)

    sorted_levels = sorted(level_groups.keys())
    for li, lv in enumerate(sorted_levels):
        nodes_at_lv = level_groups[lv]
        n = len(nodes_at_lv)
        for ni, idx in enumerate(nodes_at_lv):
            x = li * 3.0
            y = (ni - (n - 1) / 2.0) * 1.2
            pos[idx] = (x, y)

    # 그리기
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_title('GraphNet Structure (Hidden + Output nodes)', fontsize=14, fontweight='bold')

    # 노드 색상
    node_colors = []
    node_labels = {}
    for idx in G.nodes():
        ntype = G.nodes[idx]['ntype']
        if ntype == 'hidden':
            node_colors.append('#4CAF50')  # green
            ic = input_conn_count.get(idx, 0)
            node_labels[idx] = f'H{idx}\n({ic} in)'
        else:
            node_colors.append('#2196F3')  # blue
            node_labels[idx] = f'O{idx - output_indices[0]}'

    # 간선 색상/두께
    if edges_data:
        weights = [abs(w) for _, _, w in edges_data]
        max_w = max(weights) if weights else 1.0
        edge_colors = []
        edge_widths = []
        for i, j, w in edges_data:
            edge_colors.append('#E53935' if w < 0 else '#1E88E5')
            edge_widths.append(0.5 + 2.5 * abs(w) / max_w)
    else:
        edge_colors = []
        edge_widths = []

    # 간선 그리기
    edge_list = [(i, j) for i, j, _ in edges_data]
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list,
        edge_color=edge_colors, width=edge_widths,
        alpha=0.6, arrows=True, arrowsize=12,
        connectionstyle='arc3,rad=0.1',
        ax=ax,
    )

    # 노드 그리기
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(G.nodes()),
        node_color=node_colors, node_size=600,
        edgecolors='black', linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos, labels=node_labels,
        font_size=7, font_weight='bold',
        ax=ax,
    )

    # 범례
    legend_elements = [
        mpatches.Patch(color='#4CAF50', label=f'Hidden ({len(hidden_indices)})'),
        mpatches.Patch(color='#2196F3', label=f'Output ({len(output_indices)})'),
        plt.Line2D([0], [0], color='#1E88E5', linewidth=2, label='Positive weight'),
        plt.Line2D([0], [0], color='#E53935', linewidth=2, label='Negative weight'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # INPUT 요약 텍스트
    ax.text(
        0.01, 0.01,
        f'Input nodes: {len(input_indices)} (not shown)\n'
        f'Total edges: {net.n_edges} | Displayed edges: {len(edges_data)}',
        transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    ax.axis('off')
    plt.tight_layout()

    save_full = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_path)
    plt.savefig(save_full, dpi=150, bbox_inches='tight')
    print(f"그래프 구조 저장: {save_full}")
    plt.close()


if __name__ == '__main__':
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graphnet_model.pt')
    if not os.path.exists(model_path):
        print("graphnet_model.pt가 없습니다. 먼저 train.py를 실행하세요.")
        sys.exit(1)

    net = load_graphnet(model_path)
    print(f"로드 완료: {net}")
    visualize_graphnet(net)
