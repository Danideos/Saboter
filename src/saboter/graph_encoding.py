"""Typed graph encoder for action-node Saboteur policies."""

from __future__ import annotations

from dataclasses import dataclass

from saboter.actions import Action, Discard, MapGoal, PlayPath, RepairTool, Rockfall, SabotageTool
from saboter.board import GOAL_COORDS, START_COORD
from saboter.cards import CardType, GoalKind, Role, Tool
from saboter.encoding_utils import (
    CONNECTION_PAIRS,
    DIRECTIONS,
    connection_pairs_from_card,
    normalize_count,
    normalize_range,
    rotated_edges_from_card,
    rotated_groups_from_card,
    rotation_or_zero,
    validate_rotation,
)
from saboter.env import SaboteurEnv
from saboter.observation import (
    BOARD_MAX_X,
    BOARD_MAX_Y,
    BOARD_MIN_X,
    BOARD_MIN_Y,
    CARD_TYPE_NAMES,
    HISTORY_ACTION_TYPES,
    MAX_DECK_SIZE,
    MAX_HAND_SIZE,
    MAX_HISTORY,
    MAX_PLAYERS,
    TOOL_NAMES,
    _board_goal_knowledge,
    _board_structure_features,
    _known_goals,
)


NODE_TYPE_NAMES = (
    "global",
    "player",
    "card",
    "cell",
    "goal",
    "action",
    "history",
)
NODE_TYPE_IDS = {name: index for index, name in enumerate(NODE_TYPE_NAMES)}

EDGE_TYPE_NAMES = (
    "global_to_node",
    "node_to_global",
    "turn_next",
    "turn_prev",
    "card_to_action",
    "action_to_card",
    "action_to_cell",
    "cell_to_action",
    "action_to_player",
    "player_to_action",
    "action_to_goal",
    "goal_to_action",
    "cell_adj_n",
    "cell_adj_e",
    "cell_adj_s",
    "cell_adj_w",
    "cell_tunnel_connected",
    "goal_to_cell",
    "cell_to_goal",
    "history_to_actor",
    "actor_to_history",
    "history_to_target_player",
    "target_player_to_history",
    "history_to_cell",
    "cell_to_history",
    "history_to_goal",
    "goal_to_history",
)
EDGE_TYPE_IDS = {name: index for index, name in enumerate(EDGE_TYPE_NAMES)}

DIRECTION_DELTAS = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}
OPPOSITE_DIRECTION = {"N": "S", "E": "W", "S": "N", "W": "E"}

GRAPH_NODE_FEATURE_NAMES = (
    "present",
    "own_role_miner",
    "own_role_saboteur",
    "deck_size_norm",
    "discard_count_norm",
    "turn_number_norm",
    "num_players_norm",
    "terminal",
    "is_self",
    "relative_position_norm",
    "hand_size_norm",
    *(f"broken_{tool}" for tool in TOOL_NAMES),
    "card_slot_norm",
    *(f"card_slot_{slot}" for slot in range(MAX_HAND_SIZE)),
    *(f"card_type_{card_type}" for card_type in CARD_TYPE_NAMES),
    *(f"tool_{tool}" for tool in TOOL_NAMES),
    *(f"selected_tool_{tool}" for tool in TOOL_NAMES),
    *(f"edge_{direction}" for direction in DIRECTIONS),
    *(f"has_{direction}" for direction in DIRECTIONS),
    *(f"connects_{pair[0]}_{pair[1]}" for pair in CONNECTION_PAIRS),
    "group_count_norm",
    "is_dead",
    "is_split",
    "edge_count_norm",
    "x_norm",
    "y_norm",
    "empty",
    "occupied",
    "start",
    "path",
    "hidden_goal",
    "known_gold",
    "known_stone",
    "private_known_gold",
    "private_known_stone",
    "reachable_from_start",
    "frontier_empty",
    "distance_from_start_norm",
    "legal_candidate_position",
    "distance_to_goal_0",
    "distance_to_goal_1",
    "distance_to_goal_2",
    "goal_0",
    "goal_1",
    "goal_2",
    "public_unknown",
    "public_known_gold",
    "public_known_stone",
    "action_play_path",
    "action_sabotage_tool",
    "action_repair_tool",
    "action_map_goal",
    "action_rockfall",
    "action_discard",
    "rotation_180",
    "coord_present",
    "target_present",
    "target_relative_norm",
    "age_norm",
    *(f"history_action_{action_type}" for action_type in HISTORY_ACTION_TYPES),
    "actor_present",
    "actor_relative_norm",
    "revealed_gold",
    "revealed_stone",
)
GRAPH_F = {name: index for index, name in enumerate(GRAPH_NODE_FEATURE_NAMES)}
GRAPH_NODE_FEATURE_SIZE = len(GRAPH_NODE_FEATURE_NAMES)


@dataclass(frozen=True)
class GraphFeatures:
    node_features: list[list[float]]
    node_type_ids: list[int]
    edge_index: list[tuple[int, int]]
    edge_type_ids: list[int]
    action_node_indices: list[int]
    global_node_index: int
    player_node_indices: list[int]
    goal_node_indices: list[int]
    actions: list[Action]
    role_labels: list[float] | None = None
    goal_labels: list[float] | None = None
    node_feature_names: tuple[str, ...] = GRAPH_NODE_FEATURE_NAMES
    node_type_names: tuple[str, ...] = NODE_TYPE_NAMES
    edge_type_names: tuple[str, ...] = EDGE_TYPE_NAMES


class _GraphBuilder:
    def __init__(self) -> None:
        self.node_features: list[list[float]] = []
        self.node_type_ids: list[int] = []
        self.edge_index: list[tuple[int, int]] = []
        self.edge_type_ids: list[int] = []

    def add_node(self, node_type: str, features: list[float]) -> int:
        if len(features) != GRAPH_NODE_FEATURE_SIZE:
            raise ValueError("Graph node feature width mismatch")
        index = len(self.node_features)
        self.node_features.append(features)
        self.node_type_ids.append(NODE_TYPE_IDS[node_type])
        return index

    def add_edge(self, source: int, target: int, edge_type: str) -> None:
        self.edge_index.append((source, target))
        self.edge_type_ids.append(EDGE_TYPE_IDS[edge_type])


def encode_graph(
    env: SaboteurEnv,
    player_id: int | None = None,
    legal_actions: list[Action] | None = None,
    observation: dict[str, object] | None = None,
    include_labels: bool = True,
) -> GraphFeatures:
    resolved_player = env.agent_selection if player_id is None else player_id
    obs = env.observe(resolved_player) if observation is None else observation
    actions = env.legal_actions(resolved_player) if legal_actions is None else legal_actions
    role_labels = _role_labels(env, resolved_player) if include_labels else None
    goal_labels = _goal_labels(env) if include_labels else None
    return encode_graph_features(obs, actions, role_labels=role_labels, goal_labels=goal_labels)


def encode_graph_features(
    observation: dict[str, object],
    legal_actions: list[Action],
    *,
    role_labels: list[float] | None = None,
    goal_labels: list[float] | None = None,
) -> GraphFeatures:
    builder = _GraphBuilder()
    observer_id = _int_or_zero(observation.get("player_id"))
    num_players = _int_or_zero(observation.get("num_players"))
    if num_players <= 0:
        num_players = MAX_PLAYERS

    global_index = builder.add_node("global", _global_features(observation))
    player_indices, player_by_relative = _add_player_nodes(builder, observation, observer_id, num_players)
    card_indices = _add_card_nodes(builder, observation)
    cell_indices = _add_cell_nodes(builder, observation, legal_actions)
    goal_indices = _add_goal_nodes(builder, observation, cell_indices)
    action_indices = _add_action_nodes(
        builder,
        observation,
        legal_actions,
        card_indices,
        cell_indices,
        player_by_relative,
        goal_indices,
        observer_id,
        num_players,
    )
    _add_history_nodes(
        builder,
        observation,
        cell_indices,
        player_by_relative,
        goal_indices,
        observer_id,
        num_players,
    )
    _add_global_edges(builder, global_index)
    _add_turn_edges(builder, player_indices)
    _add_cell_edges(builder, observation, cell_indices)

    return GraphFeatures(
        node_features=builder.node_features,
        node_type_ids=builder.node_type_ids,
        edge_index=builder.edge_index,
        edge_type_ids=builder.edge_type_ids,
        action_node_indices=action_indices,
        global_node_index=global_index,
        player_node_indices=player_indices,
        goal_node_indices=goal_indices,
        actions=list(legal_actions),
        role_labels=role_labels,
        goal_labels=goal_labels,
    )


def _global_features(observation: dict[str, object]) -> list[float]:
    row = _empty_features()
    _set(row, "present", 1.0)
    own_role = observation.get("own_role")
    _set(row, "own_role_miner", 1.0 if own_role == Role.MINER.value else 0.0)
    _set(row, "own_role_saboteur", 1.0 if own_role == Role.SABOTEUR.value else 0.0)
    _set(row, "deck_size_norm", normalize_count(_int_or_zero(observation.get("deck_size")), MAX_DECK_SIZE))
    _set(
        row,
        "discard_count_norm",
        normalize_count(_int_or_zero(observation.get("discard_count")), MAX_DECK_SIZE * 2),
    )
    _set(row, "turn_number_norm", normalize_count(_int_or_zero(observation.get("turn_number")), 120))
    _set(row, "num_players_norm", normalize_count(_int_or_zero(observation.get("num_players")), MAX_PLAYERS))
    _set(row, "terminal", 1.0 if observation.get("terminal") else 0.0)
    return row


def _add_player_nodes(
    builder: _GraphBuilder,
    observation: dict[str, object],
    observer_id: int,
    num_players: int,
) -> tuple[list[int], dict[int, int]]:
    result = [0 for _ in range(num_players)]
    by_relative: dict[int, int] = {}
    players = observation.get("players", [])
    if not isinstance(players, list):
        return [], {}
    for player in players[:num_players]:
        if not isinstance(player, dict):
            continue
        player_id = player.get("player_id")
        if not isinstance(player_id, int):
            continue
        relative = (player_id - observer_id) % num_players
        row = _empty_features()
        _set(row, "present", 1.0)
        _set(row, "is_self", 1.0 if player.get("is_self") else 0.0)
        _set(row, "relative_position_norm", normalize_count(relative, max(1, num_players - 1)))
        hand_size = player.get("hand_size")
        if isinstance(hand_size, int):
            _set(row, "hand_size_norm", normalize_count(hand_size, MAX_HAND_SIZE))
        broken_tools = player.get("broken_tools", [])
        if isinstance(broken_tools, list):
            for tool in TOOL_NAMES:
                _set(row, f"broken_{tool}", 1.0 if tool in broken_tools else 0.0)
        node = builder.add_node("player", row)
        result[relative] = node
        by_relative[relative] = node
    return result, by_relative


def _add_card_nodes(builder: _GraphBuilder, observation: dict[str, object]) -> dict[int, int]:
    hand = observation.get("hand", [])
    result: dict[int, int] = {}
    if not isinstance(hand, list):
        return result
    for slot, card in enumerate(hand[:MAX_HAND_SIZE]):
        if not isinstance(card, dict):
            continue
        row = _card_features(card)
        _set(row, "card_slot_norm", normalize_count(slot, MAX_HAND_SIZE - 1))
        _set(row, f"card_slot_{slot}", 1.0)
        result[slot] = builder.add_node("card", row)
    return result


def _add_cell_nodes(
    builder: _GraphBuilder,
    observation: dict[str, object],
    legal_actions: list[Action],
) -> dict[tuple[int, int], int]:
    board_tiles = _public_tile_map(observation)
    _open_edges, frontier_distances, tile_distances = _board_structure_features(observation)
    coords = set(board_tiles)
    coords.update(frontier_distances)
    for coord in list(board_tiles):
        for dx, dy in DIRECTION_DELTAS.values():
            candidate = (coord[0] + dx, coord[1] + dy)
            if candidate not in board_tiles:
                coords.add(candidate)
    for action in legal_actions:
        if isinstance(action, (PlayPath, Rockfall)):
            coords.add((action.x, action.y))

    result: dict[tuple[int, int], int] = {}
    legal_candidate_coords = {
        (action.x, action.y) for action in legal_actions if isinstance(action, PlayPath)
    }
    for coord in sorted(coords):
        row = _cell_features(
            observation,
            board_tiles.get(coord),
            coord,
            frontier_distances,
            tile_distances,
            legal_candidate_coords,
        )
        result[coord] = builder.add_node("cell", row)
    return result


def _cell_features(
    observation: dict[str, object],
    tile: dict[str, object] | None,
    coord: tuple[int, int],
    frontier_distances: dict[tuple[int, int], int],
    tile_distances: dict[tuple[int, int], int],
    legal_candidate_coords: set[tuple[int, int]],
) -> list[float]:
    row = _empty_features()
    _set(row, "present", 1.0)
    _set(row, "x_norm", _normalize_x(coord[0]))
    _set(row, "y_norm", _normalize_y(coord[1]))
    for goal_index, (goal_x, goal_y) in enumerate(GOAL_COORDS):
        distance = abs(coord[0] - goal_x) + abs(coord[1] - goal_y)
        _set(row, f"distance_to_goal_{goal_index}", min(1.0, distance / 36.0))
    if tile is None:
        _set(row, "empty", 1.0)
    else:
        _set(row, "occupied", 1.0)
        kind = tile.get("kind")
        if kind == "start":
            _set(row, "start", 1.0)
        elif kind == "path":
            _set(row, "path", 1.0)
        elif kind == "goal":
            goal_index = tile.get("goal_index")
            goal_kind = tile.get("goal_kind")
            revealed = bool(tile.get("revealed"))
            if not revealed:
                _set(row, "hidden_goal", 1.0)
            elif goal_kind == GoalKind.GOLD.value:
                _set(row, "known_gold", 1.0)
            elif goal_kind == GoalKind.STONE.value:
                _set(row, "known_stone", 1.0)
            private_kind = _known_goals(observation).get(goal_index) if isinstance(goal_index, int) else None
            if private_kind == GoalKind.GOLD.value:
                _set(row, "private_known_gold", 1.0)
            elif private_kind == GoalKind.STONE.value:
                _set(row, "private_known_stone", 1.0)
        if tile.get("reachable"):
            _set(row, "reachable_from_start", 1.0)
        card = tile.get("card")
        rotation = rotation_or_zero(tile.get("rotation", 0))
        for direction in rotated_edges_from_card(card, rotation):
            _set(row, f"has_{direction}", 1.0)
        for pair in connection_pairs_from_card(card, rotation):
            _set(row, f"connects_{pair[0]}_{pair[1]}", 1.0)
    if coord in frontier_distances:
        _set(row, "frontier_empty", 1.0)
    distance_from_start = tile_distances.get(coord, frontier_distances.get(coord))
    if distance_from_start is not None:
        _set(row, "distance_from_start_norm", normalize_count(distance_from_start, 36))
    if coord in legal_candidate_coords:
        _set(row, "legal_candidate_position", 1.0)
    return row


def _add_goal_nodes(
    builder: _GraphBuilder,
    observation: dict[str, object],
    cell_indices: dict[tuple[int, int], int],
) -> list[int]:
    public_known = _board_goal_knowledge(observation)
    private_known = _known_goals(observation)
    result: list[int] = []
    for goal_index, coord in enumerate(GOAL_COORDS):
        row = _empty_features()
        _set(row, "present", 1.0)
        _set(row, f"goal_{goal_index}", 1.0)
        _set(row, "x_norm", _normalize_x(coord[0]))
        _set(row, "y_norm", _normalize_y(coord[1]))
        public_kind = public_known.get(goal_index)
        if public_kind == GoalKind.GOLD.value:
            _set(row, "public_known_gold", 1.0)
        elif public_kind == GoalKind.STONE.value:
            _set(row, "public_known_stone", 1.0)
        else:
            _set(row, "public_unknown", 1.0)
        private_kind = private_known.get(goal_index)
        if private_kind == GoalKind.GOLD.value:
            _set(row, "private_known_gold", 1.0)
        elif private_kind == GoalKind.STONE.value:
            _set(row, "private_known_stone", 1.0)
        node = builder.add_node("goal", row)
        result.append(node)
        cell = cell_indices.get(coord)
        if cell is not None:
            builder.add_edge(node, cell, "goal_to_cell")
            builder.add_edge(cell, node, "cell_to_goal")
    return result


def _add_action_nodes(
    builder: _GraphBuilder,
    observation: dict[str, object],
    legal_actions: list[Action],
    card_indices: dict[int, int],
    cell_indices: dict[tuple[int, int], int],
    player_by_relative: dict[int, int],
    goal_indices: list[int],
    observer_id: int,
    num_players: int,
) -> list[int]:
    result: list[int] = []
    for action in legal_actions:
        row = _action_features(observation, action, observer_id, num_players)
        node = builder.add_node("action", row)
        result.append(node)
        card_node = card_indices.get(action.card_slot)
        if card_node is not None:
            builder.add_edge(card_node, node, "card_to_action")
            builder.add_edge(node, card_node, "action_to_card")
        if isinstance(action, (PlayPath, Rockfall)):
            cell_node = cell_indices.get((action.x, action.y))
            if cell_node is not None:
                builder.add_edge(node, cell_node, "action_to_cell")
                builder.add_edge(cell_node, node, "cell_to_action")
        elif isinstance(action, (SabotageTool, RepairTool)):
            relative = (action.target_player - observer_id) % num_players
            player_node = player_by_relative.get(relative)
            if player_node is not None:
                builder.add_edge(node, player_node, "action_to_player")
                builder.add_edge(player_node, node, "player_to_action")
        elif isinstance(action, MapGoal):
            if 0 <= action.goal_index < len(goal_indices):
                goal_node = goal_indices[action.goal_index]
                builder.add_edge(node, goal_node, "action_to_goal")
                builder.add_edge(goal_node, node, "goal_to_action")
    return result


def _action_features(
    observation: dict[str, object],
    action: Action,
    observer_id: int,
    num_players: int,
) -> list[float]:
    row = _empty_features()
    _set(row, "present", 1.0)
    _set(row, "card_slot_norm", normalize_count(action.card_slot, MAX_HAND_SIZE - 1))
    if 0 <= action.card_slot < MAX_HAND_SIZE:
        _set(row, f"card_slot_{action.card_slot}", 1.0)
    card = _card_for_action(observation, action)
    if card is not None:
        _merge_card_features(row, card, getattr(action, "rotation", 0))
    if isinstance(action, PlayPath):
        validate_rotation(action.rotation)
        _set(row, "action_play_path", 1.0)
        _set_coord(row, action.x, action.y)
        _set(row, "rotation_180", 1.0 if action.rotation % 360 == 180 else 0.0)
    elif isinstance(action, SabotageTool):
        _set(row, "action_sabotage_tool", 1.0)
        _set_target(row, action.target_player, observer_id, num_players)
        _set(row, f"selected_tool_{action.tool.value}", 1.0)
    elif isinstance(action, RepairTool):
        _set(row, "action_repair_tool", 1.0)
        _set_target(row, action.target_player, observer_id, num_players)
        _set(row, f"selected_tool_{action.tool.value}", 1.0)
    elif isinstance(action, MapGoal):
        _set(row, "action_map_goal", 1.0)
        if 0 <= action.goal_index <= 2:
            _set(row, f"goal_{action.goal_index}", 1.0)
    elif isinstance(action, Rockfall):
        _set(row, "action_rockfall", 1.0)
        _set_coord(row, action.x, action.y)
    elif isinstance(action, Discard):
        _set(row, "action_discard", 1.0)
    return row


def _add_history_nodes(
    builder: _GraphBuilder,
    observation: dict[str, object],
    cell_indices: dict[tuple[int, int], int],
    player_by_relative: dict[int, int],
    goal_indices: list[int],
    observer_id: int,
    num_players: int,
) -> None:
    history = observation.get("history", [])
    if not isinstance(history, list):
        return
    recent = [event for event in history[-MAX_HISTORY:] if isinstance(event, dict)]
    for recent_index, event in enumerate(recent):
        age_index = len(recent) - 1 - recent_index
        node = builder.add_node("history", _history_features(event, observer_id, num_players, age_index))
        actor = event.get("actor")
        if isinstance(actor, int):
            actor_node = player_by_relative.get((actor - observer_id) % num_players)
            if actor_node is not None:
                builder.add_edge(node, actor_node, "history_to_actor")
                builder.add_edge(actor_node, node, "actor_to_history")
        target = event.get("target_player")
        if isinstance(target, int):
            target_node = player_by_relative.get((target - observer_id) % num_players)
            if target_node is not None:
                builder.add_edge(node, target_node, "history_to_target_player")
                builder.add_edge(target_node, node, "target_player_to_history")
        x = event.get("x")
        y = event.get("y")
        if isinstance(x, int) and isinstance(y, int):
            cell_node = cell_indices.get((x, y))
            if cell_node is not None:
                builder.add_edge(node, cell_node, "history_to_cell")
                builder.add_edge(cell_node, node, "cell_to_history")
        goal_index = event.get("goal_index")
        if isinstance(goal_index, int) and 0 <= goal_index < len(goal_indices):
            goal_node = goal_indices[goal_index]
            builder.add_edge(node, goal_node, "history_to_goal")
            builder.add_edge(goal_node, node, "goal_to_history")


def _history_features(
    event: dict[str, object],
    observer_id: int,
    num_players: int,
    age_index: int,
) -> list[float]:
    row = _empty_features()
    _set(row, "present", 1.0)
    _set(row, "age_norm", normalize_count(age_index, MAX_HISTORY - 1))
    action_type = event.get("action_type")
    if isinstance(action_type, str):
        _set(row, f"history_action_{action_type}", 1.0)
    card = event.get("card")
    if isinstance(card, dict):
        card_type = card.get("type")
        if isinstance(card_type, str):
            _set(row, f"card_type_{card_type}", 1.0)
    actor = event.get("actor")
    if isinstance(actor, int):
        _set(row, "actor_present", 1.0)
        _set(row, "actor_relative_norm", normalize_count((actor - observer_id) % num_players, max(1, num_players - 1)))
    target = event.get("target_player")
    if isinstance(target, int):
        _set(row, "target_present", 1.0)
        _set(row, "target_relative_norm", normalize_count((target - observer_id) % num_players, max(1, num_players - 1)))
    tool = event.get("tool")
    if isinstance(tool, str) and tool in TOOL_NAMES:
        _set(row, f"tool_{tool}", 1.0)
    x = event.get("x")
    y = event.get("y")
    if isinstance(x, int) and isinstance(y, int):
        _set_coord(row, x, y)
    goal_index = event.get("goal_index")
    if isinstance(goal_index, int) and 0 <= goal_index <= 2:
        _set(row, f"goal_{goal_index}", 1.0)
    revealed_goal_kind = event.get("revealed_goal_kind")
    if revealed_goal_kind == GoalKind.GOLD.value:
        _set(row, "revealed_gold", 1.0)
    elif revealed_goal_kind == GoalKind.STONE.value:
        _set(row, "revealed_stone", 1.0)
    return row


def _add_global_edges(builder: _GraphBuilder, global_index: int) -> None:
    for node in range(len(builder.node_features)):
        if node == global_index:
            continue
        builder.add_edge(global_index, node, "global_to_node")
        builder.add_edge(node, global_index, "node_to_global")


def _add_turn_edges(builder: _GraphBuilder, player_indices: list[int]) -> None:
    if len(player_indices) <= 1:
        return
    for index, node in enumerate(player_indices):
        next_node = player_indices[(index + 1) % len(player_indices)]
        prev_node = player_indices[(index - 1) % len(player_indices)]
        builder.add_edge(node, next_node, "turn_next")
        builder.add_edge(node, prev_node, "turn_prev")


def _add_cell_edges(
    builder: _GraphBuilder,
    observation: dict[str, object],
    cell_indices: dict[tuple[int, int], int],
) -> None:
    tile_map = _public_tile_map(observation)
    for coord, node in cell_indices.items():
        for direction, (dx, dy) in DIRECTION_DELTAS.items():
            neighbor_coord = (coord[0] + dx, coord[1] + dy)
            neighbor_node = cell_indices.get(neighbor_coord)
            if neighbor_node is not None:
                builder.add_edge(node, neighbor_node, f"cell_adj_{direction.lower()}")
            if _tiles_tunnel_connected(tile_map.get(coord), tile_map.get(neighbor_coord), direction):
                builder.add_edge(node, neighbor_node, "cell_tunnel_connected")


def _tiles_tunnel_connected(
    tile: dict[str, object] | None,
    neighbor: dict[str, object] | None,
    direction: str,
) -> bool:
    if tile is None or neighbor is None:
        return False
    opposite = OPPOSITE_DIRECTION[direction]
    for group in _tile_groups(tile):
        if direction not in group:
            continue
        for neighbor_group in _tile_groups(neighbor):
            if opposite in neighbor_group:
                return True
    return False


def _card_features(card: dict[str, object]) -> list[float]:
    row = _empty_features()
    _set(row, "present", 1.0)
    _merge_card_features(row, card, 0)
    return row


def _merge_card_features(row: list[float], card: dict[str, object], rotation: int) -> None:
    card_type = card.get("type")
    if isinstance(card_type, str):
        _set(row, f"card_type_{card_type}", 1.0)
    tools = card.get("tools", [])
    if isinstance(tools, list):
        for tool in TOOL_NAMES:
            _set(row, f"tool_{tool}", 1.0 if tool in tools else 0.0)
    edges = rotated_edges_from_card(card, rotation)
    for direction in DIRECTIONS:
        value = 1.0 if direction in edges else 0.0
        _set(row, f"edge_{direction}", value)
        _set(row, f"has_{direction}", value)
    for pair in connection_pairs_from_card(card, rotation):
        _set(row, f"connects_{pair[0]}_{pair[1]}", 1.0)
    groups = card.get("groups", [])
    group_count = len(groups) if isinstance(groups, list) else 0
    _set(row, "group_count_norm", normalize_count(group_count, 4))
    _set(row, "edge_count_norm", normalize_count(len(edges), 4))
    card_id = card.get("id")
    _set(row, "is_dead", 1.0 if isinstance(card_id, str) and card_id.startswith("dead") else 0.0)
    _set(row, "is_split", 1.0 if group_count > 1 else 0.0)


def _set_coord(row: list[float], x: int, y: int) -> None:
    _set(row, "coord_present", 1.0)
    _set(row, "x_norm", _normalize_x(x))
    _set(row, "y_norm", _normalize_y(y))


def _set_target(row: list[float], target_player: int, observer_id: int, num_players: int) -> None:
    _set(row, "target_present", 1.0)
    _set(
        row,
        "target_relative_norm",
        normalize_count((target_player - observer_id) % num_players, max(1, num_players - 1)),
    )


def _role_labels(env: SaboteurEnv, observer_id: int) -> list[float]:
    return [
        1.0 if env.players[(observer_id + relative) % env.num_players].role == Role.SABOTEUR else 0.0
        for relative in range(env.num_players)
    ]


def _goal_labels(env: SaboteurEnv) -> list[float]:
    board = env._board()
    return [
        1.0 if board.tile_at(coord).card.goal_kind == GoalKind.GOLD else 0.0
        for coord in GOAL_COORDS
    ]


def _card_for_action(observation: dict[str, object], action: Action) -> dict[str, object] | None:
    hand = observation.get("hand", [])
    if not isinstance(hand, list) or not 0 <= action.card_slot < len(hand):
        return None
    card = hand[action.card_slot]
    return card if isinstance(card, dict) else None


def _public_tile_map(observation: dict[str, object]) -> dict[tuple[int, int], dict[str, object]]:
    board = observation.get("board", [])
    if not isinstance(board, list):
        return {}
    result: dict[tuple[int, int], dict[str, object]] = {}
    for tile in board:
        if not isinstance(tile, dict):
            continue
        x = tile.get("x")
        y = tile.get("y")
        if isinstance(x, int) and isinstance(y, int):
            result[(x, y)] = tile
    return result


def _tile_groups(tile: dict[str, object] | None) -> tuple[frozenset[str], ...]:
    if tile is None:
        return ()
    card = tile.get("card")
    if tile.get("kind") == "goal" and not isinstance(card, dict):
        return (frozenset(DIRECTIONS),)
    return rotated_groups_from_card(card, rotation_or_zero(tile.get("rotation", 0)))


def _normalize_x(x: int) -> float:
    return normalize_range(x, BOARD_MIN_X, BOARD_MAX_X)


def _normalize_y(y: int) -> float:
    return normalize_range(y, BOARD_MIN_Y, BOARD_MAX_Y)


def _empty_features() -> list[float]:
    return [0.0 for _ in GRAPH_NODE_FEATURE_NAMES]


def _set(row: list[float], name: str, value: float) -> None:
    row[GRAPH_F[name]] = value


def _int_or_zero(value: object) -> int:
    return value if isinstance(value, int) else 0
