"""Minimal PPO update for variable legal-action batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from saboter.training.rollout import Transition
from saboter.training.returns import discounted_returns


@dataclass(frozen=True)
class PPOConfig:
    epochs: int = 4
    batch_size: int = 64
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    advantage_epsilon: float = 1e-8
    gamma: float = 0.99


@dataclass(frozen=True)
class PPOMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    loss: float
    transitions: int
    updates: int


def ppo_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    config: PPOConfig,
    device: str | torch.device = "cpu",
) -> PPOMetrics:
    if not transitions:
        raise ValueError("Cannot run PPO update with no transitions")
    if config.epochs <= 0:
        raise ValueError("PPO epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("PPO batch_size must be positive")

    resolved_device = torch.device(device)
    returns = discounted_returns(
        [transition.reward for transition in transitions],
        [transition.done for transition in transitions],
        config.gamma,
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
    }
    update_count = 0
    sample_count = 0
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
            flat_logits, values_tensor = model.score_action_batches(
                batch.boards,
                batch.nonboards,
                batch.actions,
                batch.action_owner,
            )
            new_log_probs_tensor, entropy_tensor = _selected_log_probs_and_entropies(
                flat_logits,
                batch.action_lengths,
                batch.action_indices,
                resolved_device,
            )
            log_ratio = new_log_probs_tensor - batch_old_log_probs
            ratio = torch.exp(log_ratio)
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(
                ratio,
                1.0 - config.clip_epsilon,
                1.0 + config.clip_epsilon,
            ) * batch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(values_tensor, batch_returns)
            entropy = entropy_tensor.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            if not torch.isfinite(loss):
                raise RuntimeError("PPO loss became non-finite")

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
            sample_count += batch_size
            update_count += 1

    return PPOMetrics(
        policy_loss=metric_sums["policy_loss"] / sample_count,
        value_loss=metric_sums["value_loss"] / sample_count,
        entropy=metric_sums["entropy"] / sample_count,
        approx_kl=metric_sums["approx_kl"] / sample_count,
        clip_fraction=metric_sums["clip_fraction"] / sample_count,
        grad_norm=metric_sums["grad_norm"] / sample_count,
        loss=metric_sums["loss"] / sample_count,
        transitions=transition_count,
        updates=update_count,
    )


@dataclass(frozen=True)
class _PPOBatch:
    boards: torch.Tensor
    nonboards: torch.Tensor
    actions: torch.Tensor
    action_owner: torch.Tensor
    action_lengths: list[int]
    action_indices: list[int]


def _collate_transitions(transitions: list[Transition], device: torch.device) -> _PPOBatch:
    action_lengths = [int(transition.actions.shape[0]) for transition in transitions]
    if any(length <= 0 for length in action_lengths):
        raise ValueError("All PPO transitions must have at least one legal action")
    owners = [
        owner_index
        for owner_index, length in enumerate(action_lengths)
        for _item in range(length)
    ]
    return _PPOBatch(
        boards=torch.cat([transition.board for transition in transitions], dim=0).to(device),
        nonboards=torch.cat([transition.nonboard for transition in transitions], dim=0).to(device),
        actions=torch.cat([transition.actions for transition in transitions], dim=0).to(device),
        action_owner=torch.tensor(owners, dtype=torch.long, device=device),
        action_lengths=action_lengths,
        action_indices=[transition.action_index for transition in transitions],
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
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs.append(dist.log_prob(torch.tensor(action_index, device=device)))
        entropies.append(dist.entropy())
        offset += length
    return torch.stack(new_log_probs), torch.stack(entropies)
