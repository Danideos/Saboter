"""PPO update for graph-action Saboteur policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from saboter.training.graph_rollout import GraphTransition
from saboter.training.graph_tensorize import BatchedGraphTensors, collate_graph_tensors
from saboter.training.returns import role_aware_discounted_returns


@dataclass(frozen=True)
class GraphPPOConfig:
    epochs: int = 4
    batch_size: int = 64
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    advantage_epsilon: float = 1e-8
    gamma: float = 0.99
    role_belief_coef: float = 0.05
    goal_belief_coef: float = 0.05


@dataclass(frozen=True)
class GraphPPOMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    loss: float
    role_belief_loss: float
    goal_belief_loss: float
    transitions: int
    updates: int


def graph_ppo_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    transitions: list[GraphTransition],
    config: GraphPPOConfig,
    device: str | torch.device = "cpu",
) -> GraphPPOMetrics:
    if not transitions:
        raise ValueError("Cannot run graph PPO update with no transitions")
    if config.epochs <= 0:
        raise ValueError("PPO epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("PPO batch_size must be positive")

    resolved_device = torch.device(device)
    returns = role_aware_discounted_returns(
        roles=[transition.role for transition in transitions],
        terminal_rewards=[transition.terminal_reward for transition in transitions],
        shaping_rewards=[transition.shaping_reward for transition in transitions],
        dones=[transition.done for transition in transitions],
        gamma=config.gamma,
    )
    old_values = torch.tensor([transition.value for transition in transitions], dtype=torch.float32)
    old_log_probs = torch.tensor(
        [transition.old_log_prob for transition in transitions],
        dtype=torch.float32,
    )
    advantages = returns - old_values
    if len(transitions) > 1:
        advantage_std = advantages.std(unbiased=False)
        if float(advantage_std) > config.advantage_epsilon:
            advantages = (advantages - advantages.mean()) / (advantage_std + config.advantage_epsilon)

    model.train()
    metric_sums = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
        "loss": 0.0,
        "role_belief_loss": 0.0,
        "goal_belief_loss": 0.0,
    }
    sample_count = 0
    update_count = 0
    transition_count = len(transitions)
    for _epoch in range(config.epochs):
        order = torch.randperm(transition_count).tolist()
        for start in range(0, transition_count, config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            batch_size = len(batch_indices)
            batch_old_log_probs = old_log_probs[batch_indices].to(resolved_device)
            batch_returns = returns[batch_indices].to(resolved_device)
            batch_advantages = advantages[batch_indices].to(resolved_device)
            batch = _collate_transitions([transitions[index] for index in batch_indices], resolved_device)
            output = model.score_graph_batches(batch)
            new_log_probs, entropies = _selected_log_probs_and_entropies(
                output.action_logits,
                batch.action_lengths,
                batch.action_indices,
                resolved_device,
            )
            log_ratio = new_log_probs - batch_old_log_probs
            ratio = torch.exp(log_ratio)
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(
                ratio,
                1.0 - config.clip_epsilon,
                1.0 + config.clip_epsilon,
            ) * batch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(output.values, batch_returns)
            entropy = entropies.mean()
            role_loss = _bce_or_zero(output.role_logits, batch.role_labels)
            goal_loss = _bce_or_zero(output.goal_logits, batch.goal_labels)
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
                + config.role_belief_coef * role_loss
                + config.goal_belief_coef * goal_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Graph PPO loss became non-finite")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > config.clip_epsilon).float().mean()
                )
            metric_sums["policy_loss"] += float(policy_loss.detach().cpu()) * batch_size
            metric_sums["value_loss"] += float(value_loss.detach().cpu()) * batch_size
            metric_sums["entropy"] += float(entropy.detach().cpu()) * batch_size
            metric_sums["approx_kl"] += float(approx_kl.detach().cpu()) * batch_size
            metric_sums["clip_fraction"] += float(clip_fraction.detach().cpu()) * batch_size
            metric_sums["grad_norm"] += float(grad_norm.detach().cpu()) * batch_size
            metric_sums["loss"] += float(loss.detach().cpu()) * batch_size
            metric_sums["role_belief_loss"] += float(role_loss.detach().cpu()) * batch_size
            metric_sums["goal_belief_loss"] += float(goal_loss.detach().cpu()) * batch_size
            sample_count += batch_size
            update_count += 1

    return GraphPPOMetrics(
        policy_loss=metric_sums["policy_loss"] / sample_count,
        value_loss=metric_sums["value_loss"] / sample_count,
        entropy=metric_sums["entropy"] / sample_count,
        approx_kl=metric_sums["approx_kl"] / sample_count,
        clip_fraction=metric_sums["clip_fraction"] / sample_count,
        grad_norm=metric_sums["grad_norm"] / sample_count,
        loss=metric_sums["loss"] / sample_count,
        role_belief_loss=metric_sums["role_belief_loss"] / sample_count,
        goal_belief_loss=metric_sums["goal_belief_loss"] / sample_count,
        transitions=transition_count,
        updates=update_count,
    )


def _collate_transitions(
    transitions: list[GraphTransition],
    device: torch.device,
) -> BatchedGraphTensors:
    return collate_graph_tensors(
        [transition.graph for transition in transitions],
        [transition.action_index for transition in transitions],
        device,
    )


def _selected_log_probs_and_entropies(
    flat_logits: torch.Tensor,
    action_lengths: list[int],
    action_indices: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    new_log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    offset = 0
    for length, action_index in zip(action_lengths, action_indices):
        logits = flat_logits[offset : offset + length]
        log_probs = torch.log_softmax(logits, dim=0)
        probs = log_probs.exp()
        new_log_probs.append(log_probs[torch.tensor(action_index, device=device)])
        entropies.append(-(probs * log_probs).sum())
        offset += length
    return torch.stack(new_log_probs), torch.stack(entropies)


def _bce_or_zero(logits: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
    if labels is None or labels.numel() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits, labels)
