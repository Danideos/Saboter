"""Tensor conversion and batching for graph policy inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from saboter.graph_encoding import (
    GRAPH_HISTORY_MAX_EVENTS,
    HISTORY_EVENT_FEATURE_NAMES,
    GraphFeatures,
)
from saboter.observation import MAX_PLAYERS

NUM_GOALS = 3


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
    history_features: torch.Tensor
    history_valid_mask: torch.Tensor
    history_actor: torch.Tensor
    history_target_player: torch.Tensor
    history_goal: torch.Tensor
    role_labels: torch.Tensor | None
    goal_labels: torch.Tensor | None
    role_label_mask: torch.Tensor | None = None

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
            history_features=self.history_features.to(resolved),
            history_valid_mask=self.history_valid_mask.to(resolved),
            history_actor=self.history_actor.to(resolved),
            history_target_player=self.history_target_player.to(resolved),
            history_goal=self.history_goal.to(resolved),
            role_labels=None if self.role_labels is None else self.role_labels.to(resolved),
            goal_labels=None if self.goal_labels is None else self.goal_labels.to(resolved),
            role_label_mask=None if self.role_label_mask is None else self.role_label_mask.to(resolved),
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
            history_features=self.history_features.detach().cpu(),
            history_valid_mask=self.history_valid_mask.detach().cpu(),
            history_actor=self.history_actor.detach().cpu(),
            history_target_player=self.history_target_player.detach().cpu(),
            history_goal=self.history_goal.detach().cpu(),
            role_labels=None if self.role_labels is None else self.role_labels.detach().cpu(),
            goal_labels=None if self.goal_labels is None else self.goal_labels.detach().cpu(),
            role_label_mask=None if self.role_label_mask is None else self.role_label_mask.detach().cpu(),
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
    history_features: torch.Tensor
    history_valid_mask: torch.Tensor
    history_actor: torch.Tensor
    history_target_player: torch.Tensor
    history_goal: torch.Tensor
    role_labels: torch.Tensor | None
    goal_labels: torch.Tensor | None
    role_label_mask: torch.Tensor | None = None


def tensorize_graph(
    graph: GraphFeatures,
    device: str | torch.device = "cpu",
    history_max_events: int = GRAPH_HISTORY_MAX_EVENTS,
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
    history_features, history_valid_mask, history_actor, history_target_player, history_goal = _history_tensors(
        graph,
        history_max_events,
        resolved,
    )
    player_node_indices = _padded_long_tensor(graph.player_node_indices, MAX_PLAYERS, -1, resolved)
    goal_node_indices = _padded_long_tensor(graph.goal_node_indices, NUM_GOALS, -1, resolved)
    role_labels = None
    role_label_mask = None
    if graph.role_labels is not None:
        role_labels = _padded_float_tensor(graph.role_labels, MAX_PLAYERS, 0.0, resolved)
        role_label_mask = torch.zeros((MAX_PLAYERS,), dtype=torch.bool, device=resolved)
        present_players = min(len(graph.role_labels), MAX_PLAYERS)
        if present_players > 1:
            role_label_mask[1:present_players] = True
    goal_labels = None
    if graph.goal_labels is not None:
        goal_labels = _padded_float_tensor(graph.goal_labels, NUM_GOALS, 0.0, resolved)
    return GraphTensors(
        x=torch.tensor(graph.node_features, dtype=torch.float32, device=resolved),
        node_type=torch.tensor(graph.node_type_ids, dtype=torch.long, device=resolved),
        edge_index=edge_index,
        edge_type=torch.tensor(graph.edge_type_ids, dtype=torch.long, device=resolved),
        action_node_indices=torch.tensor(graph.action_node_indices, dtype=torch.long, device=resolved),
        global_node_index=torch.tensor(graph.global_node_index, dtype=torch.long, device=resolved),
        player_node_indices=player_node_indices,
        goal_node_indices=goal_node_indices,
        history_features=history_features,
        history_valid_mask=history_valid_mask,
        history_actor=history_actor,
        history_target_player=history_target_player,
        history_goal=history_goal,
        role_labels=role_labels,
        goal_labels=goal_labels,
        role_label_mask=role_label_mask,
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
    history_features: list[torch.Tensor] = []
    history_valid_masks: list[torch.Tensor] = []
    history_actors: list[torch.Tensor] = []
    history_target_players: list[torch.Tensor] = []
    history_goals: list[torch.Tensor] = []
    role_labels: list[torch.Tensor] = []
    goal_labels: list[torch.Tensor] = []
    role_label_masks: list[torch.Tensor] = []
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
        player_node_indices.append(_offset_padded_indices(graph.player_node_indices, node_offset))
        goal_node_indices.append(_offset_padded_indices(graph.goal_node_indices, node_offset))
        history_features.append(graph.history_features)
        history_valid_masks.append(graph.history_valid_mask)
        history_actors.append(graph.history_actor)
        history_target_players.append(graph.history_target_player)
        history_goals.append(graph.history_goal)
        if graph.role_labels is not None:
            role_labels.append(graph.role_labels)
        if graph.role_label_mask is not None:
            role_label_masks.append(graph.role_label_mask)
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
        player_node_indices=torch.stack(player_node_indices, dim=0),
        goal_node_indices=torch.stack(goal_node_indices, dim=0),
        history_features=torch.stack(history_features, dim=0),
        history_valid_mask=torch.stack(history_valid_masks, dim=0),
        history_actor=torch.stack(history_actors, dim=0),
        history_target_player=torch.stack(history_target_players, dim=0),
        history_goal=torch.stack(history_goals, dim=0),
        role_labels=torch.stack(role_labels, dim=0) if role_labels else None,
        goal_labels=torch.stack(goal_labels, dim=0) if goal_labels else None,
        role_label_mask=torch.stack(role_label_masks, dim=0) if role_label_masks else None,
    )


def _history_tensors(
    graph: GraphFeatures,
    history_max_events: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if history_max_events <= 0:
        raise ValueError("history_max_events must be positive")
    feature_size = len(HISTORY_EVENT_FEATURE_NAMES)
    features = torch.zeros((history_max_events, feature_size), dtype=torch.float32, device=device)
    valid = torch.zeros((history_max_events,), dtype=torch.bool, device=device)
    actor = torch.full((history_max_events,), -1, dtype=torch.long, device=device)
    target_player = torch.full((history_max_events,), -1, dtype=torch.long, device=device)
    goal = torch.full((history_max_events,), -1, dtype=torch.long, device=device)
    source_rows = graph.history_features[-history_max_events:]
    source_valid = graph.history_valid_mask[-history_max_events:]
    source_actor = graph.history_actor[-history_max_events:]
    source_target = graph.history_target_player[-history_max_events:]
    source_goal = graph.history_goal[-history_max_events:]
    for index, row in enumerate(source_rows):
        if len(row) != feature_size:
            raise ValueError("Graph history feature width mismatch")
        features[index] = torch.tensor(row, dtype=torch.float32, device=device)
        valid[index] = bool(source_valid[index]) if index < len(source_valid) else True
        actor[index] = _bounded_index(source_actor[index] if index < len(source_actor) else -1, MAX_PLAYERS)
        target_player[index] = _bounded_index(
            source_target[index] if index < len(source_target) else -1,
            MAX_PLAYERS,
        )
        goal[index] = _bounded_index(source_goal[index] if index < len(source_goal) else -1, NUM_GOALS)
    return features, valid, actor, target_player, goal


def _padded_long_tensor(
    values: list[int],
    length: int,
    pad: int,
    device: torch.device,
) -> torch.Tensor:
    result = torch.full((length,), pad, dtype=torch.long, device=device)
    for index, value in enumerate(values[:length]):
        result[index] = int(value)
    return result


def _padded_float_tensor(
    values: list[float],
    length: int,
    pad: float,
    device: torch.device,
) -> torch.Tensor:
    result = torch.full((length,), pad, dtype=torch.float32, device=device)
    for index, value in enumerate(values[:length]):
        result[index] = float(value)
    return result


def _bounded_index(value: int, limit: int) -> int:
    return int(value) if 0 <= int(value) < limit else -1


def _offset_padded_indices(indices: torch.Tensor, offset: int) -> torch.Tensor:
    result = indices.clone()
    mask = result >= 0
    result[mask] += offset
    return result
