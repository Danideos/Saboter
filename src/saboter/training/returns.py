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
