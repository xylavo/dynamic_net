"""
실험 메인 스크립트: 고정 MLP vs 그래프 기반 동적 뉴로제네시스 네트워크 비교

실험 조건:
- Dataset: MNIST (기본) / CIFAR-10 (선택)
- Baseline: 고정 구조 MLP [64, 64]
- GraphNet-Growth: 그래프 기반 동적 노드 추가
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from typing import Dict, List

from models.dynamic_net import DynamicNet
from models.graph_net import GraphNet
from models.neurogenesis_controller import NeurogenesisController, GraphNeurogenesisController
from models.rl_controller import RLEdgeController


# ──────────────────────────────────────────────
# 1. 데이터 로더
# ──────────────────────────────────────────────
def get_dataloaders(dataset: str = 'mnist', batch_size: int = 256, data_root: str = '/tmp/data'):
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=transform)
        test_ds  = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=transform)
        input_size = 784
        n_classes = 10

    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=transform_train)
        test_ds  = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=transform_test)
        input_size = 3072
        n_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    return train_loader, test_loader, input_size, n_classes


# ──────────────────────────────────────────────
# 2. 학습/평가 함수
# ──────────────────────────────────────────────
def train_epoch(net, loader, optimizer, criterion, device, neuro_ctrl=None):
    net.train()
    total_loss, correct, total = 0., 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = net(X)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)

        # GraphNet용: 배치별 활성화 기록
        if neuro_ctrl is not None and hasattr(neuro_ctrl, 'record_activations'):
            neuro_ctrl.record_activations()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(net, loader, criterion, device):
    net.eval()
    total_loss, correct, total = 0., 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = net(X)
        loss = criterion(out, y)
        total_loss += loss.item() * y.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


# ──────────────────────────────────────────────
# 3. Baseline 실험 (고정 MLP)
# ──────────────────────────────────────────────
def run_baseline(
    name: str,
    net: DynamicNet,
    train_loader,
    test_loader,
    device: torch.device,
    n_epochs: int = 30,
    verbose: bool = True,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"  실험: {name}")
    print(f"  초기 구조: {net}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {
        'name': name,
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'n_params': [], 'architecture': [],
        'growth_events': [],
        'time_per_epoch': [],
    }

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(net, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(net, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['n_params'].append(net.total_params)
        history['architecture'].append(str(net))
        history['time_per_epoch'].append(elapsed)

        if verbose and epoch % 5 == 0:
            print(
                f"  Epoch {epoch:3d}/{n_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
                f"params={net.total_params:,} | {elapsed:.1f}s"
            )

    print(f"\n  최종 결과: val_acc={max(history['val_acc']):.4f} | params={net.total_params:,}")
    return history


# ──────────────────────────────────────────────
# 4. GraphNet-Growth 실험
# ──────────────────────────────────────────────
def run_graphnet_growth(
    name: str,
    net: GraphNet,
    train_loader,
    test_loader,
    device: torch.device,
    n_epochs: int = 30,
    verbose: bool = True,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"  실험: {name}")
    print(f"  초기 구조: {net}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    neuro_ctrl = GraphNeurogenesisController(
        net=net,
        patience=2,
        min_delta=5e-4,
        growth_neurons=16,
        growth_cooldown=1,
        incoming_k=8,
        edges_per_epoch=32,
        verbose=verbose,
    )

    history = {
        'name': name,
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'n_params': [], 'n_nodes': [], 'n_edges': [],
        'architecture': [],
        'growth_events': [],
        'prune_events': [],
        'edge_events': [],
        'edge_prune_events': [],
        'time_per_epoch': [],
    }

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            net, train_loader, optimizer, criterion, device,
            neuro_ctrl=neuro_ctrl,
        )
        val_loss, val_acc = evaluate(net, test_loader, criterion, device)
        scheduler.step()

        # 뉴로제네시스 체크 (프루닝 + 간선 프루닝 + 간선 추가 + 성장)
        old_n_prune = len(neuro_ctrl.prune_events)
        old_n_edge = len(neuro_ctrl.edge_events)
        old_n_edge_prune = len(neuro_ctrl.edge_prune_events)
        grew = neuro_ctrl.step(val_loss, epoch)

        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['n_params'].append(net.total_params)
        history['n_nodes'].append(net.n_nodes)
        history['n_edges'].append(net.n_edges)
        history['architecture'].append(str(net))
        history['time_per_epoch'].append(elapsed)

        if verbose and epoch % 5 == 0:
            print(
                f"  Epoch {epoch:3d}/{n_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
                f"nodes={net.n_nodes} edges={net.n_edges} | {elapsed:.1f}s"
            )

        # 프루닝 이벤트 기록
        if len(neuro_ctrl.prune_events) > old_n_prune:
            latest_prune = neuro_ctrl.prune_events[-1]
            history['prune_events'].append({
                'epoch': epoch,
                'n_pruned': latest_prune['n_pruned'],
                'n_nodes': net.n_nodes,
                'n_edges': net.n_edges,
                'val_acc': val_acc,
            })
            # 프루닝으로 파라미터가 변경됨 → 옵티마이저 재구성
            optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            remaining = max(1, n_epochs - epoch)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

        # 간선 프루닝 이벤트 기록
        if len(neuro_ctrl.edge_prune_events) > old_n_edge_prune:
            latest_ep = neuro_ctrl.edge_prune_events[-1]
            history['edge_prune_events'].append({
                'epoch': epoch,
                'n_removed': latest_ep['n_removed'],
                'n_edges': net.n_edges,
                'val_acc': val_acc,
            })

        # 간선 추가 이벤트 기록
        if len(neuro_ctrl.edge_events) > old_n_edge:
            latest_edge = neuro_ctrl.edge_events[-1]
            history['edge_events'].append({
                'epoch': epoch,
                'n_added': latest_edge['n_added'],
                'n_edges': net.n_edges,
                'val_acc': val_acc,
            })

        if grew is not None:
            # 새 파라미터를 옵티마이저에 등록
            optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            remaining = max(1, n_epochs - epoch)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

            history['growth_events'].append({
                'epoch': epoch,
                'n_added': grew,
                'n_nodes': net.n_nodes,
                'n_edges': net.n_edges,
                'val_acc': val_acc,
            })

    summary = neuro_ctrl.get_summary()
    print(f"\n  최종 결과: val_acc={max(history['val_acc']):.4f} | "
          f"nodes={net.n_nodes} | edges={net.n_edges} | params={net.total_params:,}")
    print(f"  성장 이벤트: {len(history['growth_events'])}회 | "
          f"프루닝 이벤트: {len(history['prune_events'])}회 | "
          f"간선 추가: {len(history['edge_events'])}회 | "
          f"간선 프루닝: {len(history['edge_prune_events'])}회")
    print(f"  최종 구조: {net}")

    return history


# ──────────────────────────────────────────────
# 5. GraphNet-RL 실험
# ──────────────────────────────────────────────
def run_graphnet_rl(
    name: str,
    net: GraphNet,
    train_loader,
    test_loader,
    device: torch.device,
    n_epochs: int = 30,
    verbose: bool = True,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"  실험: {name}")
    print(f"  초기 구조: {net}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    rl_ctrl = RLEdgeController(
        net=net,
        lr=3e-4,
        alpha=10.0,
        beta=0.005,
        epsilon_start=0.3,
        epsilon_end=0.05,
        epsilon_decay_epochs=100,
        entropy_coeff=0.1,
        update_every=5,
        growth_incoming_candidates=48,   # 32→48: 더 많은 incoming 후보
        growth_outgoing_candidates=16,
        edge_add_candidates=400,         # 200→400: 간선 추가 후보 2배
        prune_weight_threshold=0.01,     # 0.05→0.01: 프루닝 임계값 5배 낮춤
        prune_warmup_epochs=10,          # 처음 10 epoch 프루닝 비활성화
        verbose=verbose,
    )

    neuro_ctrl = GraphNeurogenesisController(
        net=net,
        patience=2,
        min_delta=5e-4,
        growth_neurons=32,               # 16→32: 한번에 더 많은 노드 추가
        growth_cooldown=1,
        incoming_k=12,                   # 8→12: 노드당 더 많은 incoming 연결
        edges_per_epoch=64,              # 32→64: 비RL 경로 간선 추가도 공격적으로
        verbose=verbose,
        rl_controller=rl_ctrl,
    )

    history = {
        'name': name,
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'n_params': [], 'n_nodes': [], 'n_edges': [],
        'architecture': [],
        'growth_events': [],
        'prune_events': [],
        'edge_events': [],
        'edge_prune_events': [],
        'time_per_epoch': [],
        'rl_policy_loss': [],
        'rl_avg_connect_prob': [],
        'rl_epsilon': [],
    }

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            net, train_loader, optimizer, criterion, device,
            neuro_ctrl=neuro_ctrl,
        )
        val_loss, val_acc = evaluate(net, test_loader, criterion, device)
        scheduler.step()

        # 뉴로제네시스 체크 (RL 기반 간선 결정 포함)
        old_n_prune = len(neuro_ctrl.prune_events)
        old_n_edge = len(neuro_ctrl.edge_events)
        old_n_edge_prune = len(neuro_ctrl.edge_prune_events)
        grew = neuro_ctrl.step(val_loss, epoch, val_acc=val_acc)

        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['n_params'].append(net.total_params)
        history['n_nodes'].append(net.n_nodes)
        history['n_edges'].append(net.n_edges)
        history['architecture'].append(str(net))
        history['time_per_epoch'].append(elapsed)

        if verbose and epoch % 5 == 0:
            print(
                f"  Epoch {epoch:3d}/{n_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
                f"nodes={net.n_nodes} edges={net.n_edges} | {elapsed:.1f}s"
            )

        # 프루닝 이벤트 기록
        if len(neuro_ctrl.prune_events) > old_n_prune:
            latest_prune = neuro_ctrl.prune_events[-1]
            history['prune_events'].append({
                'epoch': epoch,
                'n_pruned': latest_prune['n_pruned'],
                'n_nodes': net.n_nodes,
                'n_edges': net.n_edges,
                'val_acc': val_acc,
            })
            optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            remaining = max(1, n_epochs - epoch)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

        # 간선 프루닝 이벤트 기록
        if len(neuro_ctrl.edge_prune_events) > old_n_edge_prune:
            latest_ep = neuro_ctrl.edge_prune_events[-1]
            history['edge_prune_events'].append({
                'epoch': epoch,
                'n_removed': latest_ep['n_removed'],
                'n_edges': net.n_edges,
                'val_acc': val_acc,
            })

        # 간선 추가 이벤트 기록
        if len(neuro_ctrl.edge_events) > old_n_edge:
            latest_edge = neuro_ctrl.edge_events[-1]
            history['edge_events'].append({
                'epoch': epoch,
                'n_added': latest_edge['n_added'],
                'n_edges': net.n_edges,
                'val_acc': val_acc,
            })

        if grew is not None:
            optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            remaining = max(1, n_epochs - epoch)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

            history['growth_events'].append({
                'epoch': epoch,
                'n_added': grew,
                'n_nodes': net.n_nodes,
                'n_edges': net.n_edges,
                'val_acc': val_acc,
            })

        # RL 메트릭 기록
        rl_summary = rl_ctrl.get_summary()
        history['rl_policy_loss'].append(
            rl_summary['policy_losses'][-1] if rl_summary['policy_losses'] else 0.0
        )
        history['rl_avg_connect_prob'].append(
            rl_summary['avg_connect_probs'][-1] if rl_summary['avg_connect_probs'] else 0.5
        )
        history['rl_epsilon'].append(
            rl_summary['epsilons'][-1] if rl_summary['epsilons'] else 0.3
        )

    summary = neuro_ctrl.get_summary()
    print(f"\n  최종 결과: val_acc={max(history['val_acc']):.4f} | "
          f"nodes={net.n_nodes} | edges={net.n_edges} | params={net.total_params:,}")
    print(f"  성장 이벤트: {len(history['growth_events'])}회 | "
          f"프루닝 이벤트: {len(history['prune_events'])}회 | "
          f"간선 추가: {len(history['edge_events'])}회 | "
          f"간선 프루닝: {len(history['edge_prune_events'])}회")
    print(f"  RL 정책 업데이트: {len(rl_ctrl.policy_losses)}회")
    print(f"  최종 구조: {net}")

    return history


# ──────────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = 'mnist'
    n_epochs = 150

    train_loader, test_loader, input_size, n_classes = get_dataloaders(dataset)
    print(f"Dataset: {dataset.upper()} | input_size={input_size} | classes={n_classes}")

    results = {}

    # ── 실험 1: 고정 MLP 베이스라인 (이전 결과 재사용) ──
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            prev_results = json.load(f)
        if 'Baseline (Fixed)' in prev_results and len(prev_results['Baseline (Fixed)']['val_acc']) > 0:
            results['Baseline (Fixed)'] = prev_results['Baseline (Fixed)']
            print("Baseline: 이전 결과 재사용")
        else:
            prev_results = None
    else:
        prev_results = None

    if 'Baseline (Fixed)' not in results:
        baseline_net = DynamicNet(
            input_size=input_size,
            hidden_sizes=[64, 64],
            output_size=n_classes,
        ).to(device)

        results['Baseline (Fixed)'] = run_baseline(
            name='Baseline (Fixed MLP)',
            net=baseline_net,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            n_epochs=n_epochs,
        )

    # ── 실험 2: GraphNet-Growth ──
    graph_net = GraphNet(
        n_inputs=input_size,
        n_outputs=n_classes,
        initial_hidden=0,
    ).to(device)

    results['GraphNet-Growth'] = run_graphnet_growth(
        name='GraphNet-Growth (Graph Neurogenesis)',
        net=graph_net,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        n_epochs=n_epochs,
    )

    # ── 실험 3: GraphNet-RL ──
    graph_net_rl = GraphNet(
        n_inputs=input_size,
        n_outputs=n_classes,
        initial_hidden=0,
    ).to(device)

    results['GraphNet-RL'] = run_graphnet_rl(
        name='GraphNet-RL (RL Edge Decisions)',
        net=graph_net_rl,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        n_epochs=n_epochs,
    )

    # ── GraphNet 모델 저장 (시각화용) ──
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graphnet_model.pt')
    torch.save({
        'n_inputs': graph_net.n_inputs,
        'n_outputs': graph_net.n_outputs,
        'state_dict': graph_net.cpu().state_dict(),
    }, model_path)
    print(f"GraphNet 모델 저장: {model_path}")

    # ── 결과 저장 ──
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n결과 저장: {save_path}")

    # ── 요약 출력 ──
    print("\n" + "="*60)
    print("  최종 비교 요약")
    print("="*60)
    print(f"{'모델':<30} {'최고 val_acc':>12} {'최종 params':>14} {'성장':>6} {'프루닝':>6} {'간선+':>6} {'간선-':>6}")
    print("-"*82)
    for name, hist in results.items():
        prune_count = len(hist.get('prune_events', []))
        edge_count = len(hist.get('edge_events', []))
        edge_prune_count = len(hist.get('edge_prune_events', []))
        print(
            f"{name:<30} "
            f"{max(hist['val_acc']):>11.4f} "
            f"{hist['n_params'][-1]:>14,} "
            f"{len(hist['growth_events']):>6} "
            f"{prune_count:>6} "
            f"{edge_count:>6} "
            f"{edge_prune_count:>6}"
        )

    # ── 그래프 출력 ──
    plot_results(results)

    return results


def plot_results(results: Dict):
    """epoch별 val_acc, val_loss, 파라미터 수, 노드 수, 간선 수, RL 메트릭 비교 그래프."""
    # RL 메트릭이 있으면 3x3, 없으면 2x3
    has_rl = any('rl_policy_loss' in hist for hist in results.values())
    if has_rl:
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = {'Baseline (Fixed)': 'tab:blue', 'GraphNet-Growth': 'tab:orange', 'GraphNet-RL': 'tab:green'}

    # 1) Val Accuracy
    ax = axes[0, 0]
    growth_line_added = False
    prune_line_added = False
    for name, hist in results.items():
        c = colors.get(name, None)
        epochs = range(1, len(hist['val_acc']) + 1)
        ax.plot(epochs, hist['val_acc'], label=name, linewidth=2, color=c)
        for ev in hist['growth_events']:
            ax.axvline(x=ev['epoch'], color='red', linestyle='--', alpha=0.2,
                       label='Growth' if not growth_line_added else '')
            growth_line_added = True
        for ev in hist.get('prune_events', []):
            ax.axvline(x=ev['epoch'], color='blue', linestyle=':', alpha=0.2,
                       label='Pruning' if not prune_line_added else '')
            prune_line_added = True
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Val Accuracy over Epochs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2) Val Loss
    ax = axes[0, 1]
    for name, hist in results.items():
        c = colors.get(name, None)
        epochs = range(1, len(hist['val_loss']) + 1)
        ax.plot(epochs, hist['val_loss'], label=name, linewidth=2, color=c)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Val Loss over Epochs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3) Parameter Count
    ax = axes[0, 2]
    for name, hist in results.items():
        c = colors.get(name, None)
        epochs = range(1, len(hist['n_params']) + 1)
        ax.plot(epochs, hist['n_params'], label=name, linewidth=2, color=c)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Parameters')
    ax.set_title('Parameter Count over Epochs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4) Node Count
    ax = axes[1, 0]
    for name, hist in results.items():
        if 'n_nodes' in hist:
            c = colors.get(name, None)
            epochs = range(1, len(hist['n_nodes']) + 1)
            ax.plot(epochs, hist['n_nodes'], label=name, linewidth=2, color=c)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Nodes')
    ax.set_title('Node Count over Epochs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5) Edge Count
    ax = axes[1, 1]
    for name, hist in results.items():
        if 'n_edges' in hist:
            c = colors.get(name, None)
            epochs = range(1, len(hist['n_edges']) + 1)
            ax.plot(epochs, hist['n_edges'], label=name, linewidth=2, color=c)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Edges')
    ax.set_title('Edge Count over Epochs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6) Train vs Val Accuracy (과적합 확인)
    ax = axes[1, 2]
    for name, hist in results.items():
        c = colors.get(name, None)
        epochs = range(1, len(hist['val_acc']) + 1)
        ax.plot(epochs, hist['train_acc'], linestyle='--', alpha=0.5, label=f'{name} (train)', color=c)
        ax.plot(epochs, hist['val_acc'], linewidth=2, label=f'{name} (val)', color=c)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Train vs Val Accuracy')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # RL 메트릭 서브플롯 (3행)
    if has_rl:
        # 7) RL Policy Loss
        ax = axes[2, 0]
        for name, hist in results.items():
            if 'rl_policy_loss' in hist:
                epochs = range(1, len(hist['rl_policy_loss']) + 1)
                ax.plot(epochs, hist['rl_policy_loss'], label=name, linewidth=2, color=colors.get(name))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Policy Loss')
        ax.set_title('RL Policy Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 8) RL Avg Connect Prob
        ax = axes[2, 1]
        for name, hist in results.items():
            if 'rl_avg_connect_prob' in hist:
                epochs = range(1, len(hist['rl_avg_connect_prob']) + 1)
                ax.plot(epochs, hist['rl_avg_connect_prob'], label=name, linewidth=2, color=colors.get(name))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Avg Connect Prob')
        ax.set_title('RL Average Connection Probability')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 9) RL Epsilon
        ax = axes[2, 2]
        for name, hist in results.items():
            if 'rl_epsilon' in hist:
                epochs = range(1, len(hist['rl_epsilon']) + 1)
                ax.plot(epochs, hist['rl_epsilon'], label=name, linewidth=2, color=colors.get(name))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Epsilon')
        ax.set_title('RL Exploration (Epsilon)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.png')
    plt.savefig(save_path, dpi=150)
    print(f"그래프 저장: {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
