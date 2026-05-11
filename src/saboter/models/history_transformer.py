"""Structured public-history Transformer encoder for graph policies."""

from __future__ import annotations

import torch
from torch import nn


class GatedResidual(nn.Module):
    """Blend a residual input with a candidate update using a learned gate."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([x, y], dim=-1)))
        return gate * y + (1.0 - gate) * x


class GatedTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        ff_mult: int,
        dropout: float,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_gate = GatedResidual(hidden_dim)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ff_gate = GatedResidual(hidden_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~valid_mask
        attn_input = self.attn_norm(x)
        attn_output, _weights = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.attn_gate(x, attn_output)
        ff_output = self.ff(self.ff_norm(x))
        return self.ff_gate(x, ff_output)


class HistoryTransformerEncoder(nn.Module):
    def __init__(
        self,
        event_feature_size: int,
        hidden_dim: int,
        max_events: int,
        max_players: int,
        num_goals: int = 3,
        layers: int = 2,
        heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if event_feature_size <= 0:
            raise ValueError("event_feature_size must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        if max_players <= 0:
            raise ValueError("max_players must be positive")
        if num_goals <= 0:
            raise ValueError("num_goals must be positive")
        if layers <= 0:
            raise ValueError("layers must be positive")

        self.event_feature_size = event_feature_size
        self.hidden_dim = hidden_dim
        self.max_events = max_events
        self.max_players = max_players
        self.num_goals = num_goals

        self.input = nn.Linear(event_feature_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_events, hidden_dim)
        self.actor_embedding = nn.Embedding(max_players + 1, hidden_dim, padding_idx=0)
        self.target_player_embedding = nn.Embedding(max_players + 1, hidden_dim, padding_idx=0)
        self.goal_embedding = nn.Embedding(num_goals + 1, hidden_dim, padding_idx=0)
        self.layers = nn.ModuleList(
            GatedTransformerLayer(hidden_dim, heads, ff_mult, dropout)
            for _ in range(layers)
        )
        self.pool_score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        history_features: torch.Tensor,
        history_valid_mask: torch.Tensor,
        history_actor: torch.Tensor,
        history_target_player: torch.Tensor,
        history_goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if history_features.ndim != 3:
            raise ValueError("history_features must have shape [B, H, F]")
        batch_size, event_count, feature_size = history_features.shape
        if event_count != self.max_events:
            raise ValueError(f"history event count must be {self.max_events}, got {event_count}")
        if feature_size != self.event_feature_size:
            raise ValueError(
                f"history feature size must be {self.event_feature_size}, got {feature_size}"
            )
        if history_valid_mask.shape != (batch_size, event_count):
            raise ValueError("history_valid_mask must have shape [B, H]")

        positions = torch.arange(event_count, device=history_features.device)
        x = self.input(history_features)
        x = x + self.position_embedding(positions).unsqueeze(0)
        x = x + self.actor_embedding(_embedding_indices(history_actor, self.max_players))
        x = x + self.target_player_embedding(
            _embedding_indices(history_target_player, self.max_players)
        )
        x = x + self.goal_embedding(_embedding_indices(history_goal, self.num_goals))

        effective_mask = history_valid_mask.clone()
        empty_rows = ~effective_mask.any(dim=1)
        if bool(empty_rows.any()):
            effective_mask[empty_rows, 0] = True
        for layer in self.layers:
            x = layer(x, effective_mask)

        global_history = self._attention_pool(x, history_valid_mask)
        player_history = self._entity_pool(
            x,
            history_valid_mask,
            self.max_players,
            history_actor,
            history_target_player,
        )
        goal_history = self._entity_pool(
            x,
            history_valid_mask,
            self.num_goals,
            history_goal,
            None,
        )
        return global_history, player_history, goal_history

    def _attention_pool(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.pool_score(encoded).squeeze(-1)
        weights = torch.softmax(scores.masked_fill(~mask, -1.0e9), dim=1)
        weights = weights * mask.to(dtype=encoded.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return torch.sum(weights.unsqueeze(-1) * encoded, dim=1)

    def _entity_pool(
        self,
        encoded: torch.Tensor,
        valid_mask: torch.Tensor,
        entity_count: int,
        primary: torch.Tensor,
        secondary: torch.Tensor | None,
    ) -> torch.Tensor:
        entities = torch.arange(entity_count, device=encoded.device).view(1, 1, entity_count)
        entity_mask = primary.unsqueeze(-1) == entities
        if secondary is not None:
            entity_mask = entity_mask | (secondary.unsqueeze(-1) == entities)
        entity_mask = entity_mask & valid_mask.unsqueeze(-1)
        scores = self.pool_score(encoded).expand(-1, -1, entity_count)
        weights = torch.softmax(scores.masked_fill(~entity_mask, -1.0e9), dim=1)
        weights = weights * entity_mask.to(dtype=encoded.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return torch.einsum("bhe,bhd->bed", weights, encoded)


def _embedding_indices(indices: torch.Tensor, limit: int) -> torch.Tensor:
    result = indices.clamp(min=-1, max=limit - 1) + 1
    return result.to(dtype=torch.long)
