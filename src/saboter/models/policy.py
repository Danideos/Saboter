"""PyTorch action-scoring policy for Saboteur."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from saboter.observation import ObservationFeatures


@dataclass(frozen=True)
class ObservationSizes:
    board_channels: int
    board_height: int
    board_width: int
    flat_nonboard_size: int

    @classmethod
    def from_features(cls, features: ObservationFeatures) -> "ObservationSizes":
        return cls(
            board_channels=features.board_shape[0],
            board_height=features.board_shape[1],
            board_width=features.board_shape[2],
            flat_nonboard_size=(
                features.hand_shape[0] * features.hand_shape[1]
                + features.players_shape[0] * features.players_shape[1]
                + features.global_shape[0]
                + features.history_shape[0] * features.history_shape[1]
            ),
        )


class SaboteurPolicy(nn.Module):
    """Small non-recurrent policy/value network for legal action batches."""

    def __init__(
        self,
        obs_sizes: ObservationSizes,
        action_size: int,
        *,
        board_embedding_size: int = 256,
        obs_embedding_size: int = 256,
        action_embedding_size: int = 128,
        hidden_size: int = 256,
    ):
        super().__init__()
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        if obs_sizes.flat_nonboard_size <= 0:
            raise ValueError("flat_nonboard_size must be positive")

        self.obs_sizes = obs_sizes
        self.action_size = action_size

        board_flat_size = 64 * obs_sizes.board_height * obs_sizes.board_width
        self.board_encoder = nn.Sequential(
            nn.Conv2d(obs_sizes.board_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_flat_size, board_embedding_size),
            nn.ReLU(),
        )
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_sizes.flat_nonboard_size, obs_embedding_size),
            nn.ReLU(),
            nn.Linear(obs_embedding_size, obs_embedding_size),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_size, action_embedding_size),
            nn.ReLU(),
            nn.Linear(action_embedding_size, action_embedding_size),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(board_embedding_size + obs_embedding_size + action_embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(board_embedding_size + obs_embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    @classmethod
    def from_features(
        cls,
        obs_features: ObservationFeatures,
        action_size: int,
        **kwargs: object,
    ) -> "SaboteurPolicy":
        return cls(ObservationSizes.from_features(obs_features), action_size, **kwargs)

    def forward(
        self,
        board: torch.Tensor,
        nonboard_obs: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.score_actions(board, nonboard_obs, action_batch)

    def score_actions(
        self,
        board: torch.Tensor,
        nonboard_obs: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if board.ndim != 4 or board.shape[0] != 1:
            raise ValueError("board must have shape [1, C, H, W]")
        if nonboard_obs.ndim != 2 or nonboard_obs.shape[0] != 1:
            raise ValueError("nonboard_obs must have shape [1, F]")
        if action_batch.ndim != 2:
            raise ValueError("action_batch must have shape [A, action_size]")
        if action_batch.shape[0] == 0:
            raise ValueError("action_batch must contain at least one legal action")
        if action_batch.shape[1] != self.action_size:
            raise ValueError(
                f"action_batch width {action_batch.shape[1]} does not match action_size {self.action_size}"
            )

        action_owner = torch.zeros(action_batch.shape[0], dtype=torch.long, device=action_batch.device)
        logits, value = self.score_action_batches(board, nonboard_obs, action_batch, action_owner)
        return logits, value

    def score_action_batches(
        self,
        boards: torch.Tensor,
        nonboard_obs: torch.Tensor,
        actions: torch.Tensor,
        action_owner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if boards.ndim != 4:
            raise ValueError("boards must have shape [B, C, H, W]")
        if nonboard_obs.ndim != 2:
            raise ValueError("nonboard_obs must have shape [B, F]")
        if boards.shape[0] == 0:
            raise ValueError("boards must contain at least one state")
        if nonboard_obs.shape[0] != boards.shape[0]:
            raise ValueError(
                f"nonboard batch size {nonboard_obs.shape[0]} does not match boards batch size {boards.shape[0]}"
            )
        if actions.ndim != 2:
            raise ValueError("actions must have shape [total_A, action_size]")
        if actions.shape[0] == 0:
            raise ValueError("actions must contain at least one legal action")
        if actions.shape[1] != self.action_size:
            raise ValueError(
                f"actions width {actions.shape[1]} does not match action_size {self.action_size}"
            )
        if action_owner.ndim != 1 or action_owner.shape[0] != actions.shape[0]:
            raise ValueError("action_owner must have shape [total_A]")
        if action_owner.dtype != torch.long:
            raise ValueError("action_owner must be a torch.long tensor")
        if int(action_owner.min().item()) < 0 or int(action_owner.max().item()) >= boards.shape[0]:
            raise ValueError("action_owner values must be in 0..B-1")

        board_emb = self.board_encoder(boards)
        obs_emb = self.obs_encoder(nonboard_obs)
        action_emb = self.action_encoder(actions)
        owner_board = board_emb[action_owner]
        owner_obs = obs_emb[action_owner]
        flat_logits = self.scorer(
            torch.cat([owner_board, owner_obs, action_emb], dim=-1)
        ).squeeze(-1)
        values = self.value_head(torch.cat([board_emb, obs_emb], dim=-1)).squeeze(-1)
        return flat_logits, values
