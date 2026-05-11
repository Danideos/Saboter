"""Pure-PyTorch graph action-scoring policy for Saboteur."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from saboter.graph_encoding import (
    EDGE_TYPE_NAMES,
    GRAPH_F,
    HISTORY_EVENT_FEATURE_NAMES,
    GraphFeatures,
    NODE_TYPE_NAMES,
)
from saboter.models.history_transformer import HistoryTransformerEncoder
from saboter.observation import MAX_PLAYERS
from saboter.training.graph_tensorize import (
    BatchedGraphTensors,
    GraphTensors,
    collate_graph_tensors,
)

NUM_GOALS = 3


@dataclass(frozen=True)
class GraphPolicyOutput:
    action_logits: torch.Tensor
    values: torch.Tensor
    role_logits: torch.Tensor
    goal_logits: torch.Tensor


class RelGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_edge_types: int):
        super().__init__()
        self.relation_transforms = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_edge_types)
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            agg = torch.zeros_like(h)
        else:
            src = edge_index[0]
            dst = edge_index[1]
            agg = torch.zeros_like(h)
            for relation_id, transform in enumerate(self.relation_transforms):
                mask = edge_type == relation_id
                if bool(mask.any()):
                    msg = transform(h[src[mask]])
                    agg.index_add_(0, dst[mask], msg)
        h2 = self.update(torch.cat([h, agg], dim=-1))
        return self.norm(h + h2)


class GraphPolicy(nn.Module):
    """Relation-aware policy/value/belief network over typed graph nodes."""

    def __init__(
        self,
        *,
        node_feature_size: int,
        num_node_types: int,
        num_edge_types: int,
        history_event_feature_size: int = len(HISTORY_EVENT_FEATURE_NAMES),
        hidden_dim: int = 256,
        graph_layers: int = 3,
        history_encoder: str = "none",
        history_max_events: int = 100,
        history_layers: int = 2,
        history_heads: int = 4,
        belief_injection: str = "none",
        belief_post_layers: int = 1,
        belief_detach: bool = False,
        role_conditioned_heads: bool = False,
    ):
        super().__init__()
        if node_feature_size <= 0:
            raise ValueError("node_feature_size must be positive")
        if num_node_types <= 0:
            raise ValueError("num_node_types must be positive")
        if num_edge_types <= 0:
            raise ValueError("num_edge_types must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if graph_layers <= 0:
            raise ValueError("graph_layers must be positive")
        if history_event_feature_size <= 0:
            raise ValueError("history_event_feature_size must be positive")
        if history_encoder not in {"none", "transformer"}:
            raise ValueError("history_encoder must be 'none' or 'transformer'")
        if history_max_events <= 0:
            raise ValueError("history_max_events must be positive")
        if history_layers <= 0:
            raise ValueError("history_layers must be positive")
        if history_heads <= 0:
            raise ValueError("history_heads must be positive")
        if belief_injection not in {"none", "add", "second_pass"}:
            raise ValueError("belief_injection must be 'none', 'add', or 'second_pass'")
        if belief_post_layers <= 0:
            raise ValueError("belief_post_layers must be positive")

        self.model_type = "graph"
        self.node_feature_size = node_feature_size
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.history_event_feature_size = history_event_feature_size
        self.hidden_dim = hidden_dim
        self.graph_layers = graph_layers
        self.history_encoder_name = history_encoder
        self.history_max_events = history_max_events
        self.history_layers = history_layers
        self.history_heads = history_heads
        self.belief_injection = belief_injection
        self.belief_post_layers = belief_post_layers
        self.belief_detach = belief_detach
        self.role_conditioned_heads = role_conditioned_heads
        self.goal_loss_type = "ce"

        self.node_input = nn.Linear(node_feature_size, hidden_dim)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.layers = nn.ModuleList(
            RelGraphLayer(hidden_dim, num_edge_types) for _ in range(graph_layers)
        )
        self.history_encoder = (
            HistoryTransformerEncoder(
                event_feature_size=history_event_feature_size,
                hidden_dim=hidden_dim,
                max_events=history_max_events,
                max_players=MAX_PLAYERS,
                num_goals=NUM_GOALS,
                layers=history_layers,
                heads=history_heads,
            )
            if history_encoder == "transformer"
            else None
        )
        if self.history_encoder is not None:
            self.global_history_proj = nn.Linear(hidden_dim, hidden_dim)
            self.player_history_proj = nn.Linear(hidden_dim, hidden_dim)
            self.goal_history_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.global_history_proj = None
            self.player_history_proj = None
            self.goal_history_proj = None
        self.post_layers = nn.ModuleList(
            RelGraphLayer(hidden_dim, num_edge_types)
            for _ in range(belief_post_layers if belief_injection == "second_pass" else 0)
        )
        if belief_injection != "none":
            self.player_belief_proj = nn.Linear(2, hidden_dim)
            self.goal_belief_proj = nn.Linear(5, hidden_dim)
            self.global_belief_proj = nn.Linear(4, hidden_dim)
            self.action_belief_proj = nn.Linear(4, hidden_dim)
        else:
            self.player_belief_proj = None
            self.goal_belief_proj = None
            self.global_belief_proj = None
            self.action_belief_proj = None
        self.own_role_embedding = (
            nn.Embedding(2, hidden_dim) if role_conditioned_heads else None
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.role_belief_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.goal_belief_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @classmethod
    def from_features(
        cls,
        graph: GraphFeatures,
        *,
        hidden_dim: int = 256,
        graph_layers: int = 3,
        history_encoder: str = "none",
        history_max_events: int = 100,
        history_layers: int = 2,
        history_heads: int = 4,
        belief_injection: str = "none",
        belief_post_layers: int = 1,
        belief_detach: bool = False,
        role_conditioned_heads: bool = False,
    ) -> "GraphPolicy":
        return cls(
            node_feature_size=len(graph.node_feature_names),
            num_node_types=len(graph.node_type_names),
            num_edge_types=len(graph.edge_type_names),
            history_event_feature_size=len(graph.history_feature_names),
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            history_encoder=history_encoder,
            history_max_events=history_max_events,
            history_layers=history_layers,
            history_heads=history_heads,
            belief_injection=belief_injection,
            belief_post_layers=belief_post_layers,
            belief_detach=belief_detach,
            role_conditioned_heads=role_conditioned_heads,
        )

    def checkpoint_metadata(self) -> dict[str, bool | int | str]:
        return {
            "model_type": self.model_type,
            "node_feature_size": self.node_feature_size,
            "num_node_types": self.num_node_types,
            "num_edge_types": self.num_edge_types,
            "history_event_feature_size": self.history_event_feature_size,
            "hidden_dim": self.hidden_dim,
            "graph_layers": self.graph_layers,
            "history_encoder": self.history_encoder_name,
            "history_max_events": self.history_max_events,
            "history_layers": self.history_layers,
            "history_heads": self.history_heads,
            "belief_injection": self.belief_injection,
            "belief_post_layers": self.belief_post_layers,
            "belief_detach": self.belief_detach,
            "role_conditioned_heads": self.role_conditioned_heads,
            "goal_loss_type": self.goal_loss_type,
        }

    def score_graph(self, graph: GraphTensors) -> GraphPolicyOutput:
        output = self.score_graph_batches(collate_graph_tensors([graph], [0], graph.x.device))
        player_count = int((graph.player_node_indices >= 0).sum().item())
        return GraphPolicyOutput(
            action_logits=output.action_logits,
            values=output.values,
            role_logits=output.role_logits[0, :player_count],
            goal_logits=output.goal_logits[0],
        )

    def score_graph_batches(self, batch: BatchedGraphTensors) -> GraphPolicyOutput:
        self._validate_batch(batch)
        h = self.node_input(batch.x) + self.node_type_embedding(batch.node_type)
        h = self._inject_history(h, batch)
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.edge_type)
        role_logits = self._role_logits(h, batch.player_node_indices)
        goal_logits = self._goal_logits(h, batch.goal_node_indices)
        if self.belief_injection != "none":
            h = self._inject_beliefs(h, batch, role_logits, goal_logits)
            for layer in self.post_layers:
                h = layer(h, batch.edge_index, batch.edge_type)
            role_logits = self._role_logits(h, batch.player_node_indices)
            goal_logits = self._goal_logits(h, batch.goal_node_indices)
        action_context = h[batch.action_node_indices]
        value_context = h[batch.global_node_indices]
        if self.own_role_embedding is not None:
            role_ids = self._own_role_ids(batch)
            action_batch_indices = _action_batch_indices(batch.action_lengths, h.device)
            action_context = action_context + self.own_role_embedding(role_ids[action_batch_indices])
            value_context = value_context + self.own_role_embedding(role_ids)
        return GraphPolicyOutput(
            action_logits=self.action_head(action_context).squeeze(-1),
            values=self.value_head(value_context).squeeze(-1),
            role_logits=role_logits,
            goal_logits=goal_logits,
        )

    def _inject_history(self, h: torch.Tensor, batch: BatchedGraphTensors) -> torch.Tensor:
        if self.history_encoder is None:
            return h
        if (
            self.global_history_proj is None
            or self.player_history_proj is None
            or self.goal_history_proj is None
        ):
            raise RuntimeError("History projections are not initialized")
        global_history, player_history, goal_history = self.history_encoder(
            batch.history_features,
            batch.history_valid_mask,
            batch.history_actor,
            batch.history_target_player,
            batch.history_goal,
        )
        result = h.clone()
        result[batch.global_node_indices] = (
            result[batch.global_node_indices] + self.global_history_proj(global_history)
        )
        result = _add_to_padded_indices(
            result,
            batch.player_node_indices,
            self.player_history_proj(player_history),
        )
        result = _add_to_padded_indices(
            result,
            batch.goal_node_indices,
            self.goal_history_proj(goal_history),
        )
        return result

    def _role_logits(self, h: torch.Tensor, player_indices: torch.Tensor) -> torch.Tensor:
        table = _node_table(h, player_indices)
        logits = self.role_belief_head(table).squeeze(-1)
        return logits.masked_fill(player_indices < 0, 0.0)

    def _goal_logits(self, h: torch.Tensor, goal_indices: torch.Tensor) -> torch.Tensor:
        table = _node_table(h, goal_indices)
        logits = self.goal_belief_head(table).squeeze(-1)
        return logits.masked_fill(goal_indices < 0, 0.0)

    def _inject_beliefs(
        self,
        h: torch.Tensor,
        batch: BatchedGraphTensors,
        role_logits: torch.Tensor,
        goal_logits: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.player_belief_proj is None
            or self.goal_belief_proj is None
            or self.global_belief_proj is None
            or self.action_belief_proj is None
        ):
            raise RuntimeError("Belief injection projections are not initialized")
        role_prob = torch.sigmoid(role_logits)
        goal_prob = torch.softmax(goal_logits, dim=-1)
        if self.belief_detach:
            role_prob = role_prob.detach()
            goal_prob = goal_prob.detach()

        player_present = batch.player_node_indices >= 0
        is_self = torch.zeros_like(role_prob)
        if role_prob.shape[1] > 0:
            is_self[:, 0] = player_present[:, 0].to(dtype=role_prob.dtype)
        player_input = torch.stack([role_prob, is_self], dim=-1)
        player_belief = self.player_belief_proj(player_input)
        player_belief = player_belief * player_present.unsqueeze(-1).to(dtype=player_belief.dtype)

        goal_flags = _node_feature_table(
            batch.x,
            batch.goal_node_indices,
            [
                GRAPH_F["public_known_gold"],
                GRAPH_F["public_known_stone"],
                GRAPH_F["private_known_gold"],
                GRAPH_F["private_known_stone"],
            ],
        )
        goal_input = torch.cat([goal_prob.unsqueeze(-1), goal_flags], dim=-1)
        goal_belief = self.goal_belief_proj(goal_input)

        mean_role_prob = (
            (role_prob * player_present.to(dtype=role_prob.dtype)).sum(dim=1)
            / player_present.sum(dim=1).clamp_min(1).to(dtype=role_prob.dtype)
        )
        belief_summary = torch.cat([mean_role_prob.unsqueeze(-1), goal_prob], dim=-1)

        result = h.clone()
        result = _add_to_padded_indices(result, batch.player_node_indices, player_belief)
        result = _add_to_padded_indices(result, batch.goal_node_indices, goal_belief)
        result[batch.global_node_indices] = (
            result[batch.global_node_indices] + self.global_belief_proj(belief_summary)
        )
        action_batch_indices = _action_batch_indices(batch.action_lengths, h.device)
        result[batch.action_node_indices] = (
            result[batch.action_node_indices]
            + self.action_belief_proj(belief_summary)[action_batch_indices]
        )
        return result

    def _own_role_ids(self, batch: BatchedGraphTensors) -> torch.Tensor:
        global_features = batch.x[batch.global_node_indices]
        return (global_features[:, GRAPH_F["own_role_saboteur"]] >= 0.5).to(dtype=torch.long)

    def _validate_batch(self, batch: BatchedGraphTensors) -> None:
        if batch.x.ndim != 2 or batch.x.shape[1] != self.node_feature_size:
            raise ValueError(
                f"x must have shape [N, {self.node_feature_size}], got {tuple(batch.x.shape)}"
            )
        if batch.node_type.ndim != 1 or batch.node_type.shape[0] != batch.x.shape[0]:
            raise ValueError("node_type must have shape [N]")
        if int(batch.node_type.min().item()) < 0 or int(batch.node_type.max().item()) >= self.num_node_types:
            raise ValueError("node_type values out of range")
        if batch.edge_index.ndim != 2 or batch.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if batch.edge_type.ndim != 1 or batch.edge_type.shape[0] != batch.edge_index.shape[1]:
            raise ValueError("edge_type must have shape [E]")
        if batch.edge_type.numel() > 0:
            if int(batch.edge_type.min().item()) < 0 or int(batch.edge_type.max().item()) >= self.num_edge_types:
                raise ValueError("edge_type values out of range")
        if batch.edge_index.numel() > 0:
            if int(batch.edge_index.min().item()) < 0 or int(batch.edge_index.max().item()) >= batch.x.shape[0]:
                raise ValueError("edge_index values out of range")
        if batch.action_node_indices.ndim != 1 or batch.action_node_indices.numel() == 0:
            raise ValueError("action_node_indices must be a non-empty one-dimensional tensor")
        if (
            int(batch.action_node_indices.min().item()) < 0
            or int(batch.action_node_indices.max().item()) >= batch.x.shape[0]
        ):
            raise ValueError("action_node_indices values out of range")
        if batch.global_node_indices.ndim != 1 or batch.global_node_indices.numel() == 0:
            raise ValueError("global_node_indices must be a non-empty one-dimensional tensor")
        if (
            int(batch.global_node_indices.min().item()) < 0
            or int(batch.global_node_indices.max().item()) >= batch.x.shape[0]
        ):
            raise ValueError("global_node_indices values out of range")
        if batch.player_node_indices.ndim != 2 or batch.player_node_indices.shape[1] != MAX_PLAYERS:
            raise ValueError(f"player_node_indices must have shape [B, {MAX_PLAYERS}]")
        if batch.goal_node_indices.ndim != 2 or batch.goal_node_indices.shape[1] != NUM_GOALS:
            raise ValueError(f"goal_node_indices must have shape [B, {NUM_GOALS}]")
        _validate_padded_indices("player_node_indices", batch.player_node_indices, batch.x.shape[0])
        _validate_padded_indices("goal_node_indices", batch.goal_node_indices, batch.x.shape[0])
        batch_size = int(batch.global_node_indices.shape[0])
        if len(batch.action_lengths) != batch_size or len(batch.action_indices) != batch_size:
            raise ValueError("action_lengths/action_indices must have one entry per graph")
        if batch.history_features.shape != (
            batch_size,
            self.history_max_events,
            self.history_event_feature_size,
        ):
            raise ValueError(
                "history_features must have shape "
                f"[B, {self.history_max_events}, {self.history_event_feature_size}]"
            )
        if batch.history_valid_mask.shape != (batch_size, self.history_max_events):
            raise ValueError("history_valid_mask must have shape [B, H]")
        if batch.history_actor.shape != (batch_size, self.history_max_events):
            raise ValueError("history_actor must have shape [B, H]")
        if batch.history_target_player.shape != (batch_size, self.history_max_events):
            raise ValueError("history_target_player must have shape [B, H]")
        if batch.history_goal.shape != (batch_size, self.history_max_events):
            raise ValueError("history_goal must have shape [B, H]")


def _node_table(h: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    safe_indices = indices.clamp_min(0)
    table = h[safe_indices]
    return table * (indices >= 0).unsqueeze(-1).to(dtype=table.dtype)


def _node_feature_table(
    x: torch.Tensor,
    indices: torch.Tensor,
    feature_indices: list[int],
) -> torch.Tensor:
    safe_indices = indices.clamp_min(0)
    table = x[safe_indices][:, :, feature_indices]
    return table * (indices >= 0).unsqueeze(-1).to(dtype=table.dtype)


def _add_to_padded_indices(
    h: torch.Tensor,
    indices: torch.Tensor,
    additions: torch.Tensor,
) -> torch.Tensor:
    flat_indices = indices.reshape(-1)
    flat_additions = additions.reshape(-1, additions.shape[-1])
    valid = flat_indices >= 0
    if bool(valid.any()):
        h[flat_indices[valid]] = h[flat_indices[valid]] + flat_additions[valid]
    return h


def _action_batch_indices(action_lengths: list[int], device: torch.device) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(len(action_lengths), dtype=torch.long, device=device),
        torch.tensor(action_lengths, dtype=torch.long, device=device),
    )


def _validate_padded_indices(name: str, indices: torch.Tensor, node_count: int) -> None:
    valid = indices >= 0
    if bool(valid.any()):
        valid_indices = indices[valid]
        if int(valid_indices.min().item()) < 0 or int(valid_indices.max().item()) >= node_count:
            raise ValueError(f"{name} values out of range")


def default_graph_policy_kwargs() -> dict[str, int]:
    return {
        "num_node_types": len(NODE_TYPE_NAMES),
        "num_edge_types": len(EDGE_TYPE_NAMES),
    }
