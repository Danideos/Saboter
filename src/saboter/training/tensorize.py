"""Convert dependency-free encoder outputs into PyTorch tensors."""

from __future__ import annotations

import torch

from saboter.action_encoding import ActionFeatures
from saboter.observation import ObservationFeatures


def tensorize_observation(
    obs_features: ObservationFeatures,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    board = torch.tensor(obs_features.board, dtype=torch.float32, device=device).unsqueeze(0)
    nonboard_values: list[float] = []
    nonboard_values.extend(_flatten_2d(obs_features.hand))
    nonboard_values.extend(_flatten_2d(obs_features.players))
    nonboard_values.extend(obs_features.global_features)
    nonboard_values.extend(_flatten_2d(obs_features.history))
    nonboard = torch.tensor(nonboard_values, dtype=torch.float32, device=device).unsqueeze(0)
    return board, nonboard


def tensorize_actions(
    action_features: list[ActionFeatures],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    vectors = [features.vector for features in action_features]
    if not vectors:
        raise ValueError("Cannot tensorize empty action feature list")
    return torch.tensor(vectors, dtype=torch.float32, device=device)


def _flatten_2d(values: list[list[float]]) -> list[float]:
    return [item for row in values for item in row]
