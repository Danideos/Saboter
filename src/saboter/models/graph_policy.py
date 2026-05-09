"""Pure-PyTorch graph action-scoring policy for Saboteur."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from saboter.graph_encoding import GraphFeatures, NODE_TYPE_NAMES, EDGE_TYPE_NAMES
from saboter.training.graph_tensorize import BatchedGraphTensors, GraphTensors


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
        hidden_dim: int = 256,
        graph_layers: int = 3,
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

        self.model_type = "graph"
        self.node_feature_size = node_feature_size
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.hidden_dim = hidden_dim
        self.graph_layers = graph_layers

        self.node_input = nn.Linear(node_feature_size, hidden_dim)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.layers = nn.ModuleList(
            RelGraphLayer(hidden_dim, num_edge_types) for _ in range(graph_layers)
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
    ) -> "GraphPolicy":
        return cls(
            node_feature_size=len(graph.node_feature_names),
            num_node_types=len(graph.node_type_names),
            num_edge_types=len(graph.edge_type_names),
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
        )

    def checkpoint_metadata(self) -> dict[str, int | str]:
        return {
            "model_type": self.model_type,
            "node_feature_size": self.node_feature_size,
            "num_node_types": self.num_node_types,
            "num_edge_types": self.num_edge_types,
            "hidden_dim": self.hidden_dim,
            "graph_layers": self.graph_layers,
        }

    def score_graph(self, graph: GraphTensors) -> GraphPolicyOutput:
        output = self.score_graph_batches(
            BatchedGraphTensors(
                x=graph.x,
                node_type=graph.node_type,
                edge_index=graph.edge_index,
                edge_type=graph.edge_type,
                action_node_indices=graph.action_node_indices,
                action_lengths=[int(graph.action_node_indices.shape[0])],
                action_indices=[0],
                global_node_indices=graph.global_node_index.reshape(1),
                player_node_indices=graph.player_node_indices,
                goal_node_indices=graph.goal_node_indices,
                role_labels=graph.role_labels,
                goal_labels=graph.goal_labels,
            )
        )
        return GraphPolicyOutput(
            action_logits=output.action_logits,
            values=output.values,
            role_logits=output.role_logits,
            goal_logits=output.goal_logits,
        )

    def score_graph_batches(self, batch: BatchedGraphTensors) -> GraphPolicyOutput:
        self._validate_batch(batch)
        h = self.node_input(batch.x) + self.node_type_embedding(batch.node_type)
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.edge_type)
        return GraphPolicyOutput(
            action_logits=self.action_head(h[batch.action_node_indices]).squeeze(-1),
            values=self.value_head(h[batch.global_node_indices]).squeeze(-1),
            role_logits=self.role_belief_head(h[batch.player_node_indices]).squeeze(-1),
            goal_logits=self.goal_belief_head(h[batch.goal_node_indices]).squeeze(-1),
        )

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
        for name, index_tensor in {
            "action_node_indices": batch.action_node_indices,
            "global_node_indices": batch.global_node_indices,
            "player_node_indices": batch.player_node_indices,
            "goal_node_indices": batch.goal_node_indices,
        }.items():
            if index_tensor.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if index_tensor.numel() == 0:
                raise ValueError(f"{name} must not be empty")
            if int(index_tensor.min().item()) < 0 or int(index_tensor.max().item()) >= batch.x.shape[0]:
                raise ValueError(f"{name} values out of range")


def default_graph_policy_kwargs() -> dict[str, int]:
    return {
        "num_node_types": len(NODE_TYPE_NAMES),
        "num_edge_types": len(EDGE_TYPE_NAMES),
    }
