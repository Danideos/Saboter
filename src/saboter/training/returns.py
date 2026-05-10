"""Return/advantage helpers for PPO training."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def discounted_returns(
    rewards: Sequence[float],
    dones: Sequence[bool],
    gamma: float,
) -> torch.Tensor:
    """Compute episode-aware discounted reward-to-go values."""

    if len(rewards) != len(dones):
        raise ValueError("rewards and dones must have the same length")
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in [0, 1]")

    returns = torch.empty(len(rewards), dtype=torch.float32)
    running_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if dones[index]:
            running_return = 0.0
        running_return = float(rewards[index]) + gamma * running_return
        returns[index] = running_return
    return returns


def role_aware_discounted_returns(
    *,
    roles: Sequence[str],
    terminal_rewards: Sequence[float],
    shaping_rewards: Sequence[float],
    dones: Sequence[bool],
    gamma: float,
) -> torch.Tensor:
    """Compute role-aware reward-to-go values for interleaved mixed-role episodes.

    Each transition gets the final terminal outcome from *its own* role
    perspective, plus shaping rewards from future transitions of the same role.
    Discounting is still measured in full environment decision steps, so gaps
    across opposing-role turns still count toward the exponent.
    """

    count = len(roles)
    if len(terminal_rewards) != count or len(shaping_rewards) != count or len(dones) != count:
        raise ValueError("roles, terminal_rewards, shaping_rewards, and dones must have the same length")
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if count == 0:
        return torch.empty(0, dtype=torch.float32)

    returns = torch.empty(count, dtype=torch.float32)
    episode_start = 0
    for index, done in enumerate(dones):
        if not done:
            continue
        _fill_role_aware_episode_returns(
            returns,
            episode_start,
            index + 1,
            roles,
            terminal_rewards,
            shaping_rewards,
            gamma,
        )
        episode_start = index + 1
    if episode_start != count:
        _fill_role_aware_episode_returns(
            returns,
            episode_start,
            count,
            roles,
            terminal_rewards,
            shaping_rewards,
            gamma,
        )
    return returns


def _fill_role_aware_episode_returns(
    returns: torch.Tensor,
    start: int,
    end: int,
    roles: Sequence[str],
    terminal_rewards: Sequence[float],
    shaping_rewards: Sequence[float],
    gamma: float,
) -> None:
    final_index = end - 1
    future_returns_by_role: dict[str, float] = {}
    future_indices_by_role: dict[str, int] = {}
    for index in range(final_index, start - 1, -1):
        role = str(roles[index])
        if role not in future_returns_by_role:
            future_returns_by_role[role] = float(terminal_rewards[index])
            future_indices_by_role[role] = final_index
        gap = future_indices_by_role[role] - index
        running_return = float(shaping_rewards[index]) + (gamma ** gap) * future_returns_by_role[role]
        returns[index] = running_return
        future_returns_by_role[role] = running_return
        future_indices_by_role[role] = index
