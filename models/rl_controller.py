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
