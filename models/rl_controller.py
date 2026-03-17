"""
RLConnectionController: Policy Gradient (REINFORCE)로 연결 구조를 학습.

아이디어:
- 새 뉴런이 추가될 때마다 RL 에이전트가 "어떤 기존 뉴런과 연결할지" 결정
- State: 현재 그래프 임베딩 (레이어 크기, 활성화 통계)
- Action: 연결 마스크 (skip connection 포함 여부, 초기 가중치 크기)
- Reward: 다음 N step 후 accuracy 향상 - λ * 파라미터 증가량

이 구현은 단순화된 버전:
- Action = (skip_connection_add: bool, init_scale: float)
- Policy = 2층 MLP (상태 → 행동 확률)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Optional
import numpy as np


class PolicyNetwork(nn.Module):
    """RL 정책 네트워크: 그래프 상태 → 연결 결정."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 행동 1: skip connection 추가 여부 (binary)
        self.skip_head = nn.Linear(hidden_dim, 2)
        # 행동 2: 초기화 스케일 (작게/중간/크게)
        self.scale_head = nn.Linear(hidden_dim, 3)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            skip_logits: (2,) - skip 연결 추가/안함
            scale_logits: (3,) - 가중치 초기화 크기 {0.001, 0.01, 0.1}
        """
        h = self.net(state)
        return self.skip_head(h), self.scale_head(h)

    def sample_action(self, state: torch.Tensor) -> Tuple[dict, torch.Tensor]:
        """
        정책에서 행동 샘플링.
        Returns:
            action: {'add_skip': bool, 'init_scale': float}
            log_prob: 행동의 로그 확률 (REINFORCE 업데이트용)
        """
        skip_logits, scale_logits = self.forward(state)

        skip_dist = torch.distributions.Categorical(logits=skip_logits)
        scale_dist = torch.distributions.Categorical(logits=scale_logits)

        skip_act = skip_dist.sample()
        scale_act = scale_dist.sample()

        log_prob = skip_dist.log_prob(skip_act) + scale_dist.log_prob(scale_act)

        scales = [0.001, 0.01, 0.1]
        action = {
            'add_skip': bool(skip_act.item()),
            'init_scale': scales[scale_act.item()],
            'skip_idx': skip_act.item(),
            'scale_idx': scale_act.item(),
        }
        return action, log_prob


class RLConnectionController:
    """
    REINFORCE 알고리즘으로 연결 구조 학습.

    사용 방식:
    1. 뉴런 추가 직후 decide_connection() 호출
    2. 몇 epoch 학습 후 observe_reward() 호출
    3. update_policy() 로 정책 업데이트
    """

    def __init__(
        self,
        state_dim: int = 8,       # 그래프 상태 차원 (레이어 수에 맞게 조정)
        lr: float = 3e-4,
        gamma: float = 0.99,      # 할인율
        lambda_param: float = 0.01,  # 파라미터 패널티 계수
        verbose: bool = True,
    ):
        self.gamma = gamma
        self.lambda_param = lambda_param
        self.verbose = verbose

        self.policy = PolicyNetwork(state_dim=state_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 에피소드 버퍼 (뉴런 추가 이벤트 하나 = 에피소드)
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.actions: List[dict] = []
        self.states: List[torch.Tensor] = []

        # 성능 추적
        self.policy_losses: List[float] = []
        self.decision_history: List[dict] = []

    def decide_connection(
        self,
        graph_state: torch.Tensor,
        n_params_before: int,
    ) -> dict:
        """
        새 뉴런 추가 시 연결 방식 결정.

        Args:
            graph_state: 현재 네트워크 상태 (DynamicNet.get_graph_state())
            n_params_before: 변경 전 파라미터 수

        Returns:
            action dict: {'add_skip': bool, 'init_scale': float}
        """
        # 상태 차원 맞추기 (패딩/트런케이션)
        state = self._normalize_state(graph_state)

        action, log_prob = self.policy.sample_action(state)

        self.log_probs.append(log_prob)
        self.states.append(state)
        self.actions.append({**action, 'n_params_before': n_params_before})

        if self.verbose:
            print(
                f"  [RL] skip={'YES' if action['add_skip'] else 'NO'} | "
                f"init_scale={action['init_scale']}"
            )

        return action

    def observe_reward(
        self,
        acc_before: float,
        acc_after: float,
        n_params_before: int,
        n_params_after: int,
    ) -> float:
        """
        행동 결과 관찰 후 보상 계산.

        보상 = Δ정확도 - λ * Δ파라미터 비율
        """
        delta_acc = acc_after - acc_before
        delta_param_ratio = (n_params_after - n_params_before) / (n_params_before + 1)
        reward = delta_acc - self.lambda_param * delta_param_ratio

        self.rewards.append(reward)

        if self.verbose:
            print(
                f"  [RL Reward] Δacc={delta_acc:+.4f} | "
                f"Δparams={n_params_after - n_params_before:+d} | "
                f"reward={reward:.4f}"
            )

        return reward

    def update_policy(self) -> Optional[float]:
        """
        REINFORCE 알고리즘으로 정책 업데이트.
        에피소드가 최소 1개 이상 있을 때만 업데이트.
        """
        if len(self.rewards) == 0:
            return None

        # 할인 누적 보상 계산
        returns = self._compute_returns()

        # 정규화 (분산 감소)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        policy_loss = 0
        for log_prob, ret in zip(self.log_probs, returns):
            policy_loss -= log_prob * ret  # REINFORCE: -E[log π * R]

        self.optimizer.zero_grad()
        policy_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = policy_loss.item()
        self.policy_losses.append(loss_val)

        # 기록 후 버퍼 초기화
        for i, (action, reward) in enumerate(zip(self.actions, self.rewards)):
            self.decision_history.append({**action, 'reward': reward, 'return': returns[i].item()})

        self.log_probs.clear()
        self.rewards.clear()
        self.actions.clear()
        self.states.clear()

        return loss_val

    def _compute_returns(self) -> torch.Tensor:
        """할인 누적 보상 계산."""
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """정책 네트워크 입력 차원에 맞게 상태 패딩/트런케이션."""
        target_dim = self.policy.net[0].in_features
        current_dim = state.shape[0]
        if current_dim < target_dim:
            padding = torch.zeros(target_dim - current_dim)
            state = torch.cat([state, padding])
        elif current_dim > target_dim:
            state = state[:target_dim]
        return state.detach()

    def get_summary(self) -> dict:
        return {
            'n_decisions': len(self.decision_history),
            'avg_reward': np.mean([d['reward'] for d in self.decision_history]) if self.decision_history else 0,
            'policy_losses': self.policy_losses,
            'decisions': self.decision_history,
        }


# ──────────────────────────────────────────────────────────────
# EdgePolicyNetwork & RLEdgeController: 간선 수준 RL 결정
# ──────────────────────────────────────────────────────────────

class EdgePolicyNetwork(nn.Module):
    """
    간선 수준 정책 네트워크.

    입력: 18차원 간선 특징 벡터 (고정 크기, 그래프 크기 무관)
      [Edge-local 12차원]
        0: src_activation_mean
        1: src_activation_var
        2: tgt_activation_mean
        3: tgt_activation_var
        4: src_out_degree_norm    (/ N)
        5: tgt_in_degree_norm     (/ N)
        6: src_level_norm         (/ max_level)
        7: tgt_level_norm         (/ max_level)
        8: level_gap_norm         ((tgt-src) / max_level)
        9: src_is_input           (0 or 1)
        10: tgt_is_output         (0 or 1)
        11: abs_weight            (|W[s,t]|, 프루닝 시 사용, 추가 시 0.0)
      [Global context 6차원 — get_graph_state() 재사용]
        12-17: n_hidden_norm, n_edges_norm, levels_norm, hidden_ratio, inputs_ratio, outputs_ratio

    출력:
      - connect_prob: (B,) 연결/유지 확률 (Sigmoid)
      - scale_logits: (B, 3) 초기화 스케일 로짓 (추가 시만 사용)
    """

    def __init__(self, feature_dim: int = 18, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.connect_head = nn.Linear(hidden_dim, 1)
        self.scale_head = nn.Linear(hidden_dim, 3)

    def forward(self, edge_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            edge_features: (B, 18)

        Returns:
            connect_probs: (B,) — 연결 확률
            scale_logits: (B, 3) — 초기화 스케일 로짓
        """
        h = self.shared(edge_features)
        connect_probs = torch.sigmoid(self.connect_head(h)).squeeze(-1)
        scale_logits = self.scale_head(h)
        return connect_probs, scale_logits


class RLEdgeController:
    """
    간선 수준 RL 컨트롤러: REINFORCE로 간선 결정을 학습.

    모든 간선 결정(성장 연결, 간선 추가, 간선 프루닝)을 통합 관리.
    """

    SCALE_VALUES = [0.001, 0.01, 0.1]

    def __init__(
        self,
        net,  # GraphNet
        lr: float = 3e-4,
        alpha: float = 10.0,       # 정확도 보상 계수
        beta: float = 0.005,       # 파라미터 페널티 계수
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.05,
        epsilon_decay_epochs: int = 100,
        entropy_coeff: float = 0.1,
        entropy_decay: float = 0.995,
        entropy_min: float = 0.01,
        update_every: int = 5,     # N epoch마다 정책 업데이트
        growth_incoming_candidates: int = 32,
        growth_outgoing_candidates: int = 16,
        edge_add_candidates: int = 200,
        prune_weight_threshold: float = 0.05,
        prune_warmup_epochs: int = 0,  # 이 epoch까지 프루닝 비활성화
        verbose: bool = True,
    ):
        self.net = net
        self.alpha = alpha
        self.beta = beta
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_epochs = epsilon_decay_epochs
        self.entropy_coeff = entropy_coeff
        self.entropy_decay = entropy_decay
        self.entropy_min = entropy_min
        self.update_every = update_every
        self.growth_incoming_candidates = growth_incoming_candidates
        self.growth_outgoing_candidates = growth_outgoing_candidates
        self.edge_add_candidates = edge_add_candidates
        self.prune_weight_threshold = prune_weight_threshold
        self.prune_warmup_epochs = prune_warmup_epochs
        self.verbose = verbose

        self.policy = EdgePolicyNetwork(feature_dim=18, hidden_dim=64)
        # 정책 네트워크를 net과 같은 device로 이동
        device = next(net.parameters()).device
        self.policy = self.policy.to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 노드별 활성화 통계 (epoch 시작 시 precompute)
        self._act_mean: Optional[torch.Tensor] = None  # (N,)
        self._act_var: Optional[torch.Tensor] = None    # (N,)

        # 에피소드 버퍼 (한 epoch 내 모든 간선 결정)
        self._epoch_log_probs: List[torch.Tensor] = []
        self._epoch_entropies: List[torch.Tensor] = []

        # Epoch 결과 추적 (지연 보상용)
        self._prev_acc: Optional[float] = None
        self._prev_param_ratio: Optional[float] = None
        self._epoch_rewards: List[float] = []
        self._epoch_log_prob_sums: List[torch.Tensor] = []
        self._epoch_entropy_sums: List[torch.Tensor] = []

        # 성능 추적
        self.policy_losses: List[float] = []
        self.avg_connect_probs: List[float] = []
        self.epsilons: List[float] = []
        self._current_epoch_connect_probs: List[float] = []

    @property
    def _epsilon(self) -> float:
        """현재 epsilon (선형 감소)."""
        if not self.epsilons:
            return self.epsilon_start
        epoch = len(self.epsilons)
        frac = min(epoch / max(self.epsilon_decay_epochs, 1), 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    @property
    def _current_entropy_coeff(self) -> float:
        """현재 엔트로피 보너스 계수."""
        epoch = len(self.policy_losses)
        coeff = self.entropy_coeff * (self.entropy_decay ** epoch)
        return max(coeff, self.entropy_min)

    def precompute_activation_stats(self, activation_buffer: List[torch.Tensor]):
        """
        epoch 시작 시 노드별 mean/var 계산.

        Args:
            activation_buffer: list of (batch, N) 활성화 텐서
        """
        if not activation_buffer:
            N = self.net.n_nodes
            self._act_mean = torch.zeros(N)
            self._act_var = torch.zeros(N)
            return

        all_acts = torch.cat(activation_buffer, dim=0)  # (total_samples, N)
        self._act_mean = all_acts.mean(dim=0)
        self._act_var = all_acts.var(dim=0)

    def compute_edge_features_batch(
        self, edges: List[Tuple[int, int]], is_pruning: bool = False
    ) -> torch.Tensor:
        """
        후보 간선들의 18차원 특징 벡터 일괄 생성.

        Args:
            edges: (src, tgt) 리스트
            is_pruning: True이면 abs_weight를 현재 가중치로 설정

        Returns:
            (B, 18) 특징 텐서
        """
        net = self.net
        device = next(net.parameters()).device
        N = net.n_nodes
        max_level = max(net._node_levels.max().item(), 1)
        out_degree, in_degree = net.get_node_degrees()
        graph_state = net.get_graph_state()  # (6,)

        # 활성화 통계가 없으면 0으로 채움
        act_mean = self._act_mean if self._act_mean is not None else torch.zeros(N)
        act_var = self._act_var if self._act_var is not None else torch.zeros(N)

        B = len(edges)
        features = torch.zeros(B, 18, device=device)

        for i, (src, tgt) in enumerate(edges):
            # Edge-local 12차원
            features[i, 0] = act_mean[src] if src < len(act_mean) else 0.0
            features[i, 1] = act_var[src] if src < len(act_var) else 0.0
            features[i, 2] = act_mean[tgt] if tgt < len(act_mean) else 0.0
            features[i, 3] = act_var[tgt] if tgt < len(act_var) else 0.0
            features[i, 4] = out_degree[src].item() / max(N, 1)
            features[i, 5] = in_degree[tgt].item() / max(N, 1)
            features[i, 6] = net._node_levels[src].item() / max_level
            features[i, 7] = net._node_levels[tgt].item() / max_level
            features[i, 8] = (net._node_levels[tgt].item() - net._node_levels[src].item()) / max_level
            features[i, 9] = 1.0 if net.node_types[src] == net.INPUT else 0.0
            features[i, 10] = 1.0 if net.node_types[tgt] == net.OUTPUT else 0.0
            if is_pruning:
                features[i, 11] = abs(net.W.data[src, tgt].item())
            else:
                features[i, 11] = 0.0

            # Global context 6차원
            features[i, 12:18] = graph_state

        return features

    def _sample_decisions(
        self, edge_features: torch.Tensor
    ) -> Tuple[List[bool], List[float], torch.Tensor, torch.Tensor]:
        """
        간선별 연결/유지 결정 샘플링.

        Returns:
            decisions: 각 간선의 연결 여부 리스트
            scale_values: 각 간선의 초기화 스케일 값 리스트
            log_prob_sum: 배치 전체의 로그 확률 합
            entropy_sum: 배치 전체의 엔트로피 합
        """
        if edge_features.shape[0] == 0:
            return [], [], torch.tensor(0.0), torch.tensor(0.0)

        connect_probs, scale_logits = self.policy(edge_features)

        epsilon = self._epsilon
        decisions = []
        log_probs = []
        entropies = []
        scale_values = []

        for i in range(edge_features.shape[0]):
            p = connect_probs[i]
            self._current_epoch_connect_probs.append(p.item())

            # Epsilon-greedy 탐색
            if np.random.random() < epsilon:
                decision = np.random.random() < 0.5
            else:
                decision = p.item() > 0.5

            # 로그 확률 계산
            if decision:
                lp = torch.log(p + 1e-8)
            else:
                lp = torch.log(1 - p + 1e-8)
            log_probs.append(lp)

            # 엔트로피: -p*log(p) - (1-p)*log(1-p)
            ent = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
            entropies.append(ent)

            # 스케일 결정
            scale_dist = torch.distributions.Categorical(logits=scale_logits[i])
            scale_idx = scale_dist.sample()
            log_probs.append(scale_dist.log_prob(scale_idx))
            entropies.append(scale_dist.entropy())
            scale_values.append(self.SCALE_VALUES[scale_idx.item()])

            decisions.append(decision)

        log_prob_sum = torch.stack(log_probs).sum()
        entropy_sum = torch.stack(entropies).sum()

        return decisions, scale_values, log_prob_sum, entropy_sum

    def decide_growth_connections(
        self,
        new_node_indices: List[int],
        n_growth: int,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        새 노드의 incoming/outgoing 연결을 RL로 결정.

        Args:
            new_node_indices: 새로 추가될 노드들의 예상 인덱스
            n_growth: 추가할 노드 수

        Returns:
            (connect_from_list, connect_to_list): 각 노드별 incoming/outgoing 연결
        """
        net = self.net
        output_indices = (net.node_types == net.OUTPUT).nonzero(as_tuple=True)[0].tolist()
        non_output_indices = (net.node_types != net.OUTPUT).nonzero(as_tuple=True)[0].tolist()

        connect_from_list = []
        connect_to_list = []

        for node_i in range(n_growth):
            # ── Incoming 후보 생성 ──
            n_in_cand = min(self.growth_incoming_candidates, len(non_output_indices))
            if n_in_cand == 0:
                connect_from_list.append([])
                connect_to_list.append(output_indices[:1] if output_indices else [])
                continue

            perm = torch.randperm(len(non_output_indices))[:n_in_cand]
            in_candidates = [non_output_indices[p.item()] for p in perm]

            # 새 노드 인덱스 (아직 추가 전이므로, 가상 인덱스 사용)
            # 특징 벡터에서 tgt 관련 값은 새 노드이므로 기본값 사용
            in_edges = [(src, net.n_nodes) for src in in_candidates]
            in_features = self._compute_growth_features(in_edges, is_incoming=True)

            in_decisions, in_scales, lp, ent = self._sample_decisions(in_features)
            self._epoch_log_probs.append(lp)
            self._epoch_entropies.append(ent)

            selected_from = [src for src, dec in zip(in_candidates, in_decisions) if dec]

            # 최소 연결 보장: RL이 모두 거부하면 가장 높은 확률의 후보 1개 강제 연결
            if not selected_from and in_candidates:
                with torch.no_grad():
                    probs, _ = self.policy(in_features)
                best_idx = probs.argmax().item()
                selected_from = [in_candidates[best_idx]]

            connect_from_list.append(selected_from)

            # ── Outgoing 후보 생성 ──
            n_out_cand = min(self.growth_outgoing_candidates, len(output_indices))
            if n_out_cand == 0:
                connect_to_list.append([])
                continue

            perm = torch.randperm(len(output_indices))[:n_out_cand]
            out_candidates = [output_indices[p.item()] for p in perm]

            out_edges = [(net.n_nodes, tgt) for tgt in out_candidates]
            out_features = self._compute_growth_features(out_edges, is_incoming=False)

            out_decisions, out_scales, lp, ent = self._sample_decisions(out_features)
            self._epoch_log_probs.append(lp)
            self._epoch_entropies.append(ent)

            selected_to = [tgt for tgt, dec in zip(out_candidates, out_decisions) if dec]

            # 최소 연결 보장
            if not selected_to and out_candidates:
                with torch.no_grad():
                    probs, _ = self.policy(out_features)
                best_idx = probs.argmax().item()
                selected_to = [out_candidates[best_idx]]

            connect_to_list.append(selected_to)

        return connect_from_list, connect_to_list

    def _compute_growth_features(
        self, edges: List[Tuple[int, int]], is_incoming: bool
    ) -> torch.Tensor:
        """
        성장 시 후보 간선의 특징 벡터 계산.
        새 노드는 아직 그래프에 없으므로 기본값 사용.
        """
        net = self.net
        device = next(net.parameters()).device
        N = net.n_nodes
        max_level = max(net._node_levels.max().item(), 1)
        out_degree, in_degree = net.get_node_degrees()
        graph_state = net.get_graph_state()

        act_mean = self._act_mean if self._act_mean is not None else torch.zeros(N)
        act_var = self._act_var if self._act_var is not None else torch.zeros(N)

        B = len(edges)
        features = torch.zeros(B, 18, device=device)

        for i, (src, tgt) in enumerate(edges):
            if is_incoming:
                # src는 기존 노드, tgt는 새 노드
                features[i, 0] = act_mean[src].item() if src < len(act_mean) else 0.0
                features[i, 1] = act_var[src].item() if src < len(act_var) else 0.0
                features[i, 2] = 0.0  # 새 노드: 활성화 없음
                features[i, 3] = 0.0
                features[i, 4] = out_degree[src].item() / max(N, 1) if src < N else 0.0
                features[i, 5] = 0.0  # 새 노드: in_degree 0
                features[i, 6] = net._node_levels[src].item() / max_level if src < N else 0.0
                features[i, 7] = 0.5  # 새 노드: 중간 레벨 추정
                features[i, 8] = 0.5  # level gap 추정
                features[i, 9] = 1.0 if src < N and net.node_types[src] == net.INPUT else 0.0
                features[i, 10] = 0.0  # 새 노드는 HIDDEN
            else:
                # src는 새 노드, tgt는 기존 노드
                features[i, 0] = 0.0  # 새 노드
                features[i, 1] = 0.0
                features[i, 2] = act_mean[tgt].item() if tgt < len(act_mean) else 0.0
                features[i, 3] = act_var[tgt].item() if tgt < len(act_var) else 0.0
                features[i, 4] = 0.0  # 새 노드: out_degree 0
                features[i, 5] = in_degree[tgt].item() / max(N, 1) if tgt < N else 0.0
                features[i, 6] = 0.5  # 새 노드: 중간 레벨 추정
                features[i, 7] = net._node_levels[tgt].item() / max_level if tgt < N else 0.0
                features[i, 8] = 0.5
                features[i, 9] = 0.0  # 새 노드는 HIDDEN
                features[i, 10] = 1.0 if tgt < N and net.node_types[tgt] == net.OUTPUT else 0.0

            features[i, 11] = 0.0  # 성장 시 weight 없음
            features[i, 12:18] = graph_state

        return features

    def decide_edge_additions(self) -> List[Tuple[int, int]]:
        """
        기존 노드 간 간선 추가를 RL로 결정.

        Returns:
            추가할 (src, tgt) 리스트
        """
        net = self.net
        non_output = (net.node_types != net.OUTPUT).nonzero(as_tuple=True)[0].tolist()
        non_input = (net.node_types != net.INPUT).nonzero(as_tuple=True)[0].tolist()

        if not non_output or not non_input:
            return []

        # 랜덤 후보 쌍 생성 (DAG 제약 만족하는 것만)
        candidates = []
        attempts = 0
        max_attempts = self.edge_add_candidates * 3
        while len(candidates) < self.edge_add_candidates and attempts < max_attempts:
            src = non_output[np.random.randint(len(non_output))]
            tgt = non_input[np.random.randint(len(non_input))]
            attempts += 1
            if src == tgt:
                continue
            if net.mask[src, tgt] > 0:
                continue
            if net._node_levels[src] >= net._node_levels[tgt]:
                continue
            candidates.append((src, tgt))

        if not candidates:
            return []

        # 특징 벡터 계산
        features = self.compute_edge_features_batch(candidates, is_pruning=False)
        decisions, scale_values, lp, ent = self._sample_decisions(features)
        self._epoch_log_probs.append(lp)
        self._epoch_entropies.append(ent)

        # 선택된 간선 반환
        selected = [(s, t) for (s, t), dec in zip(candidates, decisions) if dec]
        return selected

    def decide_edge_pruning(self, current_epoch: int = 0) -> List[Tuple[int, int]]:
        """
        약한 간선 제거를 RL로 결정.

        Args:
            current_epoch: 현재 epoch (warmup 체크용)

        Returns:
            제거할 (src, tgt) 리스트
        """
        # Warmup 기간 동안 프루닝 비활성화
        if current_epoch <= self.prune_warmup_epochs:
            return []

        net = self.net
        N = net.n_nodes
        W_abs = (net.W.data * net.mask).abs()

        # |W| < threshold인 간선만 후보
        candidates = []
        for src in range(N):
            for tgt in range(N):
                if net.mask[src, tgt] == 0:
                    continue
                if W_abs[src, tgt] >= self.prune_weight_threshold:
                    continue
                candidates.append((src, tgt))

        if not candidates:
            return []

        # 특징 벡터 계산
        features = self.compute_edge_features_batch(candidates, is_pruning=True)
        decisions, _, lp, ent = self._sample_decisions(features)
        self._epoch_log_probs.append(lp)
        self._epoch_entropies.append(ent)

        # decision=True → 제거 (프루닝에서는 "연결"이 "제거" 의미)
        selected = [(s, t) for (s, t), dec in zip(candidates, decisions) if dec]
        return selected

    def record_epoch_result(self, val_acc: float, n_params: int):
        """
        epoch 결과 기록 (지연 보상 계산용).

        Args:
            val_acc: 검증 정확도
            n_params: 현재 파라미터 수
        """
        max_params = 256 * 256  # 정규화 기준
        param_ratio = n_params / max(max_params, 1)

        if self._prev_acc is not None and self._epoch_log_probs:
            delta_acc = val_acc - self._prev_acc
            delta_param_ratio = param_ratio - self._prev_param_ratio
            reward = self.alpha * delta_acc - self.beta * delta_param_ratio

            self._epoch_rewards.append(reward)

            # 이 epoch의 로그 확률 / 엔트로피 합산
            if self._epoch_log_probs:
                lp_sum = torch.stack(self._epoch_log_probs).sum()
                ent_sum = torch.stack(self._epoch_entropies).sum()
                self._epoch_log_prob_sums.append(lp_sum)
                self._epoch_entropy_sums.append(ent_sum)
        elif self._epoch_log_probs:
            # 첫 epoch: 보상 0으로 기록
            self._epoch_rewards.append(0.0)
            lp_sum = torch.stack(self._epoch_log_probs).sum()
            ent_sum = torch.stack(self._epoch_entropies).sum()
            self._epoch_log_prob_sums.append(lp_sum)
            self._epoch_entropy_sums.append(ent_sum)

        self._prev_acc = val_acc
        self._prev_param_ratio = param_ratio

        # 평균 연결 확률 기록
        if self._current_epoch_connect_probs:
            avg_prob = np.mean(self._current_epoch_connect_probs)
        else:
            avg_prob = 0.5
        self.avg_connect_probs.append(avg_prob)
        self.epsilons.append(self._epsilon)

        # epoch 버퍼 초기화
        self._epoch_log_probs.clear()
        self._epoch_entropies.clear()
        self._current_epoch_connect_probs.clear()

    def maybe_update_policy(self, epoch: int) -> Optional[float]:
        """
        N epoch마다 REINFORCE 업데이트.

        Args:
            epoch: 현재 epoch

        Returns:
            policy loss 값 (업데이트 시), None (스킵 시)
        """
        if epoch % self.update_every != 0:
            return None

        if len(self._epoch_rewards) == 0 or len(self._epoch_log_prob_sums) == 0:
            return None

        # 보상 정규화
        rewards = torch.tensor(self._epoch_rewards, dtype=torch.float32)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # REINFORCE loss + 엔트로피 보너스
        policy_loss = torch.tensor(0.0)
        entropy_bonus = torch.tensor(0.0)
        ent_coeff = self._current_entropy_coeff

        n = min(len(self._epoch_log_prob_sums), len(rewards))
        for i in range(n):
            policy_loss = policy_loss - self._epoch_log_prob_sums[i] * rewards[i]
            entropy_bonus = entropy_bonus + self._epoch_entropy_sums[i]

        total_loss = policy_loss - ent_coeff * entropy_bonus

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = policy_loss.item()
        self.policy_losses.append(loss_val)

        if self.verbose:
            print(
                f"  [RL Update] epoch={epoch} | "
                f"policy_loss={loss_val:.4f} | "
                f"entropy_coeff={ent_coeff:.4f} | "
                f"epsilon={self._epsilon:.3f} | "
                f"avg_connect_prob={self.avg_connect_probs[-1]:.3f}"
            )

        # 버퍼 초기화
        self._epoch_rewards.clear()
        self._epoch_log_prob_sums.clear()
        self._epoch_entropy_sums.clear()

        return loss_val

    def get_summary(self) -> dict:
        return {
            'policy_losses': self.policy_losses,
            'avg_connect_probs': self.avg_connect_probs,
            'epsilons': self.epsilons,
        }
