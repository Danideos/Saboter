"""Tensor conversion and batching for graph policy inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from saboter.graph_encoding import GraphFeatures


@dataclass(frozen=True)
class GraphTensors:
    x: torch.Tensor
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    action_node_indices: torch.Tensor
    global_node_index: torch.Tensor
    player_node_indices: torch.Tensor
    goal_node_indices: torch.Tensor
    role_labels: torch.Tensor | None
    goal_labels: torch.Tensor | None

    def to(self, device: str | torch.device) -> "GraphTensors":
        resolved = torch.device(device)
        return GraphTensors(
            x=self.x.to(resolved),
            node_type=self.node_type.to(resolved),
            edge_index=self.edge_index.to(resolved),
            edge_type=self.edge_type.to(resolved),
            action_node_indices=self.action_node_indices.to(resolved),
            global_node_index=self.global_node_index.to(resolved),
            player_node_indices=self.player_node_indices.to(resolved),
            goal_node_indices=self.goal_node_indices.to(resolved),
            role_labels=None if self.role_labels is None else self.role_labels.to(resolved),
            goal_labels=None if self.goal_labels is None else self.goal_labels.to(resolved),
        )

    def detach_cpu(self) -> "GraphTensors":
        return GraphTensors(
            x=self.x.detach().cpu(),
            node_type=self.node_type.detach().cpu(),
            edge_index=self.edge_index.detach().cpu(),
            edge_type=self.edge_type.detach().cpu(),
            action_node_indices=self.action_node_indices.detach().cpu(),
            global_node_index=self.global_node_index.detach().cpu(),
            player_node_indices=self.player_node_indices.detach().cpu(),
            goal_node_indices=self.goal_node_indices.detach().cpu(),
            role_labels=None if self.role_labels is None else self.role_labels.detach().cpu(),
            goal_labels=None if self.goal_labels is None else self.goal_labels.detach().cpu(),
        )


@dataclass(frozen=True)
class BatchedGraphTensors:
    x: torch.Tensor
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    action_node_indices: torch.Tensor
    action_lengths: list[int]
    action_indices: list[int]
    global_node_indices: torch.Tensor
    player_node_indices: torch.Tensor
    goal_node_indices: torch.Tensor
    role_labels: torch.Tensor | None
    goal_labels: torch.Tensor | None


def tensorize_graph(
    graph: GraphFeatures,
    device: str | torch.device = "cpu",
) -> GraphTensors:
    if not graph.node_features:
        raise ValueError("Cannot tensorize graph with no nodes")
    if not graph.action_node_indices:
        raise ValueError("Cannot tensorize graph with no action nodes")
    resolved = torch.device(device)
    if graph.edge_index:
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=resolved).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=resolved)
    return GraphTensors(
        x=torch.tensor(graph.node_features, dtype=torch.float32, device=resolved),
        node_type=torch.tensor(graph.node_type_ids, dtype=torch.long, device=resolved),
        edge_index=edge_index,
        edge_type=torch.tensor(graph.edge_type_ids, dtype=torch.long, device=resolved),
        action_node_indices=torch.tensor(graph.action_node_indices, dtype=torch.long, device=resolved),
        global_node_index=torch.tensor(graph.global_node_index, dtype=torch.long, device=resolved),
        player_node_indices=torch.tensor(graph.player_node_indices, dtype=torch.long, device=resolved),
        goal_node_indices=torch.tensor(graph.goal_node_indices, dtype=torch.long, device=resolved),
        role_labels=(
            None
            if graph.role_labels is None
            else torch.tensor(graph.role_labels, dtype=torch.float32, device=resolved)
        ),
        goal_labels=(
            None
            if graph.goal_labels is None
            else torch.tensor(graph.goal_labels, dtype=torch.float32, device=resolved)
        ),
    )


def collate_graph_tensors(
    graphs: list[GraphTensors],
    action_indices: list[int],
    device: str | torch.device,
) -> BatchedGraphTensors:
    if not graphs:
        raise ValueError("Cannot collate empty graph batch")
    if len(graphs) != len(action_indices):
        raise ValueError("graph/action index count mismatch")
    resolved = torch.device(device)
    node_offset = 0
    xs: list[torch.Tensor] = []
    node_types: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    edge_types: list[torch.Tensor] = []
    action_node_indices: list[torch.Tensor] = []
    action_lengths: list[int] = []
    global_node_indices: list[torch.Tensor] = []
    player_node_indices: list[torch.Tensor] = []
    goal_node_indices: list[torch.Tensor] = []
    role_labels: list[torch.Tensor] = []
    goal_labels: list[torch.Tensor] = []
    for graph in graphs:
        graph = graph.to(resolved)
        xs.append(graph.x)
        node_types.append(graph.node_type)
        if graph.edge_index.numel() > 0:
            edge_indices.append(graph.edge_index + node_offset)
            edge_types.append(graph.edge_type)
        action_node_indices.append(graph.action_node_indices + node_offset)
        action_lengths.append(int(graph.action_node_indices.shape[0]))
        global_node_indices.append(graph.global_node_index.reshape(1) + node_offset)
        player_node_indices.append(graph.player_node_indices + node_offset)
        goal_node_indices.append(graph.goal_node_indices + node_offset)
        if graph.role_labels is not None:
            role_labels.append(graph.role_labels)
        if graph.goal_labels is not None:
            goal_labels.append(graph.goal_labels)
        node_offset += int(graph.x.shape[0])

    if edge_indices:
        batch_edge_index = torch.cat(edge_indices, dim=1)
        batch_edge_type = torch.cat(edge_types, dim=0)
    else:
        batch_edge_index = torch.empty((2, 0), dtype=torch.long, device=resolved)
        batch_edge_type = torch.empty((0,), dtype=torch.long, device=resolved)

    return BatchedGraphTensors(
        x=torch.cat(xs, dim=0),
        node_type=torch.cat(node_types, dim=0),
        edge_index=batch_edge_index,
        edge_type=batch_edge_type,
        action_node_indices=torch.cat(action_node_indices, dim=0),
        action_lengths=action_lengths,
        action_indices=action_indices,
        global_node_indices=torch.cat(global_node_indices, dim=0),
        player_node_indices=torch.cat(player_node_indices, dim=0),
        goal_node_indices=torch.cat(goal_node_indices, dim=0),
        role_labels=torch.cat(role_labels, dim=0) if role_labels else None,
        goal_labels=torch.cat(goal_labels, dim=0) if goal_labels else None,
    )
