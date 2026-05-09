"""Neural-ready observation encoders.

The encoder intentionally returns plain Python lists plus explicit shape/name
metadata. Model code can convert these to NumPy/PyTorch tensors without making
the simulator core depend on a tensor library.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from saboter.actions import Action, PlayPath, Rockfall
from saboter.board import GOAL_COORDS, START_COORD
from saboter.cards import CardType, GoalKind, Role, Tool
from saboter.encoding_utils import (
    CONNECTION_PAIRS,
    DIRECTIONS,
    connection_pairs_from_card as _connection_pairs_from_card,
    normalize_count as _normalize_count,
    normalize_range as _normalize_range,
    rotated_edges_from_card as _rotated_edges_from_card,
    rotated_groups_from_card as _rotated_groups_from_card,
    rotation_or_zero as _rotation_or_zero,
    validate_rotation as _validate_rotation,
)
from saboter.env import SaboteurEnv


BOARD_MIN_X = -6
BOARD_MAX_X = 14
BOARD_MIN_Y = -7
BOARD_MAX_Y = 7
BOARD_WIDTH = BOARD_MAX_X - BOARD_MIN_X + 1
BOARD_HEIGHT = BOARD_MAX_Y - BOARD_MIN_Y + 1
MAX_HAND_SIZE = 6
MAX_PLAYERS = 10
MAX_HISTORY = 20
MAX_DECK_SIZE = 67
DISTANCE_FROM_START_MAX = BOARD_WIDTH + BOARD_HEIGHT

DIRECTION_DELTAS = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}
OPPOSITE_DIRECTION = {"N": "S", "E": "W", "S": "N", "W": "E"}

CARD_TYPE_NAMES = (
    CardType.PATH.value,
    CardType.SABOTAGE.value,
    CardType.REPAIR.value,
    CardType.MAP.value,
    CardType.ROCKFALL.value,
)
TOOL_NAMES = (Tool.PICKAXE.value, Tool.LANTERN.value, Tool.CART.value)
HISTORY_ACTION_TYPES = (
    "discard",
    "play_path",
    "sabotage",
    "repair",
    "map_goal",
    "rockfall",
    "reveal_goal",
)

BOARD_CHANNEL_NAMES = (
    "empty",
    "occupied",
    "start",
    "path",
    "hidden_goal",
    "known_gold",
    "known_stone",
    "private_known_gold",
    "private_known_stone",
    "has_N",
    "has_E",
    "has_S",
    "has_W",
    "connects_N_E",
    "connects_N_S",
    "connects_N_W",
    "connects_E_S",
    "connects_E_W",
    "connects_S_W",
    "reachable_from_start",
    "reachable_open_N",
    "reachable_open_E",
    "reachable_open_S",
    "reachable_open_W",
    "frontier_empty",
    "distance_from_start_norm",
    "legal_candidate_position",
    "x_norm",
    "y_norm",
    "distance_to_goal_0",
    "distance_to_goal_1",
    "distance_to_goal_2",
)

HAND_FEATURE_NAMES = (
    "present",
    *(f"card_type_{card_type}" for card_type in CARD_TYPE_NAMES),
    *(f"tool_{tool}" for tool in TOOL_NAMES),
    *(f"edge_{direction}" for direction in DIRECTIONS),
    *(f"connects_{pair[0]}_{pair[1]}" for pair in CONNECTION_PAIRS),
    "group_count_norm",
    "is_dead",
    "is_split",
    "edge_count_norm",
)

PLAYER_FEATURE_NAMES = (
    "present",
    "is_self",
    "relative_position_norm",
    "hand_size_norm",
    *(f"broken_{tool}" for tool in TOOL_NAMES),
)

GLOBAL_FEATURE_NAMES = (
    "own_role_miner",
    "own_role_saboteur",
    "deck_size_norm",
    "discard_count_norm",
    "turn_number_norm",
    "player_id_norm",
    "num_players_norm",
    "is_agent_selection",
    "terminal",
    "known_goal_0_unknown",
    "known_goal_0_gold",
    "known_goal_0_stone",
    "known_goal_1_unknown",
    "known_goal_1_gold",
    "known_goal_1_stone",
    "known_goal_2_unknown",
    "known_goal_2_gold",
    "known_goal_2_stone",
    "off_board_tile_count_norm",
    "off_board_legal_action_count_norm",
)

HISTORY_FEATURE_NAMES = (
    "present",
    "age_norm",
    *(f"action_{action_type}" for action_type in HISTORY_ACTION_TYPES),
    *(f"card_type_{card_type}" for card_type in CARD_TYPE_NAMES),
    "actor_present",
    "actor_relative_norm",
    "target_present",
    "target_relative_norm",
    *(f"tool_{tool}" for tool in TOOL_NAMES),
    "coord_present",
    "x_norm",
    "y_norm",
    "goal_0",
    "goal_1",
    "goal_2",
    "revealed_gold",
    "revealed_stone",
)

BOARD_CH = {name: index for index, name in enumerate(BOARD_CHANNEL_NAMES)}
HAND_F = {name: index for index, name in enumerate(HAND_FEATURE_NAMES)}
PLAYER_F = {name: index for index, name in enumerate(PLAYER_FEATURE_NAMES)}
GLOBAL_F = {name: index for index, name in enumerate(GLOBAL_FEATURE_NAMES)}
HISTORY_F = {name: index for index, name in enumerate(HISTORY_FEATURE_NAMES)}


@dataclass(frozen=True)
class ObservationFeatures:
    board: list[list[list[float]]]
    hand: list[list[float]]
    players: list[list[float]]
    global_features: list[float]
    history: list[list[float]]
    board_shape: tuple[int, int, int]
    hand_shape: tuple[int, int]
    players_shape: tuple[int, int]
    global_shape: tuple[int]
    history_shape: tuple[int, int]
    board_channel_names: tuple[str, ...] = BOARD_CHANNEL_NAMES
    hand_feature_names: tuple[str, ...] = HAND_FEATURE_NAMES
    player_feature_names: tuple[str, ...] = PLAYER_FEATURE_NAMES
    global_feature_names: tuple[str, ...] = GLOBAL_FEATURE_NAMES
    history_feature_names: tuple[str, ...] = HISTORY_FEATURE_NAMES

    def flat_vector(self) -> list[float]:
        values: list[float] = []
        values.extend(_flatten_3d(self.board))
        values.extend(_flatten_2d(self.hand))
        values.extend(_flatten_2d(self.players))
        values.extend(self.global_features)
        values.extend(_flatten_2d(self.history))
        return values


def encode_observation(
    env: SaboteurEnv,
    player_id: int | None = None,
    legal_actions: list[Action] | None = None,
) -> ObservationFeatures:
    resolved_player = env.agent_selection if player_id is None else player_id
    observation = env.observe(resolved_player)
    actions = env.legal_actions(resolved_player) if legal_actions is None else legal_actions
    return encode_observation_features(observation, actions)


def encode_observation_features(
    observation: dict[str, object],
    legal_actions: list[Action] | None = None,
) -> ObservationFeatures:
    """Encode an already-built legal observation into model features."""
    actions = legal_actions
    board = encode_board_tensor(observation, actions)
    hand = encode_hand(observation)
    players = encode_players(observation)
    global_features = encode_global_features(observation, actions)
    history = encode_history(observation)
    return ObservationFeatures(
        board=board,
        hand=hand,
        players=players,
        global_features=global_features,
        history=history,
        board_shape=(len(BOARD_CHANNEL_NAMES), BOARD_HEIGHT, BOARD_WIDTH),
        hand_shape=(MAX_HAND_SIZE, len(HAND_FEATURE_NAMES)),
        players_shape=(MAX_PLAYERS, len(PLAYER_FEATURE_NAMES)),
        global_shape=(len(GLOBAL_FEATURE_NAMES),),
        history_shape=(MAX_HISTORY, len(HISTORY_FEATURE_NAMES)),
    )


def encode_board_tensor(
    observation: dict[str, object],
    legal_actions: list[Action] | None = None,
) -> list[list[list[float]]]:
    tensor = [
        [[0.0 for _x in range(BOARD_WIDTH)] for _y in range(BOARD_HEIGHT)]
        for _channel in BOARD_CHANNEL_NAMES
    ]

    for y in range(BOARD_MIN_Y, BOARD_MAX_Y + 1):
        for x in range(BOARD_MIN_X, BOARD_MAX_X + 1):
            row, col = _coord_to_index(x, y)
            _set_channel(tensor, "empty", row, col, 1.0)
            _set_channel(tensor, "x_norm", row, col, _normalize_x(x))
            _set_channel(tensor, "y_norm", row, col, _normalize_y(y))
            for goal_index, (goal_x, goal_y) in enumerate(GOAL_COORDS):
                distance = abs(x - goal_x) + abs(y - goal_y)
                _set_channel(
                    tensor,
                    f"distance_to_goal_{goal_index}",
                    row,
                    col,
                    min(1.0, distance / float(BOARD_WIDTH + BOARD_HEIGHT)),
                )

    private_goal_knowledge = _known_goals(observation)
    board_tiles = observation.get("board", [])
    if isinstance(board_tiles, list):
        for tile in board_tiles:
            if not isinstance(tile, dict):
                continue
            coord = _coord_to_index_if_visible(tile.get("x"), tile.get("y"))
            if coord is None:
                continue
            row, col = coord
            _set_channel(tensor, "empty", row, col, 0.0)
            _set_channel(tensor, "occupied", row, col, 1.0)
            kind = tile.get("kind")
            if kind == "start":
                _set_channel(tensor, "start", row, col, 1.0)
            elif kind == "path":
                _set_channel(tensor, "path", row, col, 1.0)
            elif kind == "goal":
                goal_index = tile.get("goal_index")
                goal_kind = tile.get("goal_kind")
                revealed = bool(tile.get("revealed"))
                if not revealed:
                    _set_channel(tensor, "hidden_goal", row, col, 1.0)
                elif goal_kind == GoalKind.GOLD.value:
                    _set_channel(tensor, "known_gold", row, col, 1.0)
                elif goal_kind == GoalKind.STONE.value:
                    _set_channel(tensor, "known_stone", row, col, 1.0)
                if isinstance(goal_index, int):
                    private_kind = private_goal_knowledge.get(goal_index)
                    if private_kind == GoalKind.GOLD.value:
                        _set_channel(tensor, "private_known_gold", row, col, 1.0)
                    elif private_kind == GoalKind.STONE.value:
                        _set_channel(tensor, "private_known_stone", row, col, 1.0)

            if tile.get("reachable"):
                _set_channel(tensor, "reachable_from_start", row, col, 1.0)
            card = tile.get("card")
            rotation = _rotation_or_zero(tile.get("rotation", 0))
            edges = _rotated_edges_from_card(card, rotation)
            for direction in edges:
                _set_channel(tensor, f"has_{direction}", row, col, 1.0)
            for pair in _connection_pairs_from_card(card, rotation):
                _set_channel(tensor, f"connects_{pair[0]}_{pair[1]}", row, col, 1.0)

    open_edges, frontier_distances, tile_distances = _board_structure_features(observation)
    for coord, directions in open_edges.items():
        visible_coord = _coord_to_index_if_visible(coord[0], coord[1])
        if visible_coord is None:
            continue
        row, col = visible_coord
        for direction in directions:
            _set_channel(tensor, f"reachable_open_{direction}", row, col, 1.0)
    for coord, distance in {**tile_distances, **frontier_distances}.items():
        visible_coord = _coord_to_index_if_visible(coord[0], coord[1])
        if visible_coord is None:
            continue
        row, col = visible_coord
        _set_channel(
            tensor,
            "distance_from_start_norm",
            row,
            col,
            _normalize_count(distance, DISTANCE_FROM_START_MAX),
        )
    for coord in frontier_distances:
        visible_coord = _coord_to_index_if_visible(coord[0], coord[1])
        if visible_coord is None:
            continue
        row, col = visible_coord
        _set_channel(tensor, "frontier_empty", row, col, 1.0)

    for action in legal_actions or []:
        if not isinstance(action, PlayPath):
            continue
        _validate_rotation(action.rotation)
        coord = _coord_to_index_if_visible(action.x, action.y)
        if coord is not None:
            row, col = coord
            _set_channel(tensor, "legal_candidate_position", row, col, 1.0)

    return tensor


def encode_hand(observation: dict[str, object]) -> list[list[float]]:
    hand = observation.get("hand", [])
    encoded = [[0.0 for _ in HAND_FEATURE_NAMES] for _slot in range(MAX_HAND_SIZE)]
    if not isinstance(hand, list):
        return encoded
    for slot, card in enumerate(hand[:MAX_HAND_SIZE]):
        if isinstance(card, dict):
            encoded[slot] = _encode_card(card)
    return encoded


def encode_players(observation: dict[str, object]) -> list[list[float]]:
    players = observation.get("players", [])
    encoded = [[0.0 for _ in PLAYER_FEATURE_NAMES] for _player in range(MAX_PLAYERS)]
    if not isinstance(players, list):
        return encoded
    observer_id = _int_or_zero(observation.get("player_id"))
    num_players = _int_or_zero(observation.get("num_players"))
    if num_players <= 0:
        num_players = MAX_PLAYERS
    for player in players[:MAX_PLAYERS]:
        if not isinstance(player, dict):
            continue
        player_id = player.get("player_id")
        if not isinstance(player_id, int) or not 0 <= player_id < MAX_PLAYERS:
            continue
        row = [0.0 for _ in PLAYER_FEATURE_NAMES]
        _set_feature(row, PLAYER_F, "present", 1.0)
        _set_feature(row, PLAYER_F, "is_self", 1.0 if player.get("is_self") else 0.0)
        relative_position = player.get("relative_position")
        if isinstance(relative_position, int):
            _set_feature(
                row,
                PLAYER_F,
                "relative_position_norm",
                _normalize_count(relative_position, max(1, num_players - 1)),
            )
        hand_size = player.get("hand_size")
        if isinstance(hand_size, int):
            _set_feature(
                row,
                PLAYER_F,
                "hand_size_norm",
                _normalize_count(hand_size, MAX_HAND_SIZE),
            )
        broken_tools = player.get("broken_tools", [])
        if isinstance(broken_tools, list):
            for tool in TOOL_NAMES:
                _set_feature(
                    row,
                    PLAYER_F,
                    f"broken_{tool}",
                    1.0 if tool in broken_tools else 0.0,
                )
        relative_index = (player_id - observer_id) % num_players
        if 0 <= relative_index < MAX_PLAYERS:
            encoded[relative_index] = row
    return encoded


def encode_global_features(
    observation: dict[str, object],
    legal_actions: list[Action] | None = None,
) -> list[float]:
    features = [0.0 for _ in GLOBAL_FEATURE_NAMES]
    own_role = observation.get("own_role")
    _set_feature(features, GLOBAL_F, "own_role_miner", 1.0 if own_role == Role.MINER.value else 0.0)
    _set_feature(
        features,
        GLOBAL_F,
        "own_role_saboteur",
        1.0 if own_role == Role.SABOTEUR.value else 0.0,
    )
    _set_feature(
        features,
        GLOBAL_F,
        "deck_size_norm",
        _normalize_count(_int_or_zero(observation.get("deck_size")), MAX_DECK_SIZE),
    )
    _set_feature(
        features,
        GLOBAL_F,
        "discard_count_norm",
        _normalize_count(_int_or_zero(observation.get("discard_count")), MAX_DECK_SIZE * 2),
    )
    _set_feature(
        features,
        GLOBAL_F,
        "turn_number_norm",
        _normalize_count(_int_or_zero(observation.get("turn_number")), 120),
    )
    _set_feature(
        features,
        GLOBAL_F,
        "player_id_norm",
        _normalize_count(_int_or_zero(observation.get("player_id")), MAX_PLAYERS - 1),
    )
    _set_feature(
        features,
        GLOBAL_F,
        "num_players_norm",
        _normalize_count(_int_or_zero(observation.get("num_players")), MAX_PLAYERS),
    )
    _set_feature(
        features,
        GLOBAL_F,
        "is_agent_selection",
        1.0 if observation.get("agent_selection") == observation.get("player_id") else 0.0,
    )
    _set_feature(features, GLOBAL_F, "terminal", 1.0 if observation.get("terminal") else 0.0)

    known_goals = _known_goals(observation)
    board_known_goals = _board_goal_knowledge(observation)
    for goal_index in range(3):
        known_kind = known_goals.get(goal_index, board_known_goals.get(goal_index))
        if known_kind == GoalKind.GOLD.value:
            _set_feature(features, GLOBAL_F, f"known_goal_{goal_index}_gold", 1.0)
        elif known_kind == GoalKind.STONE.value:
            _set_feature(features, GLOBAL_F, f"known_goal_{goal_index}_stone", 1.0)
        else:
            _set_feature(features, GLOBAL_F, f"known_goal_{goal_index}_unknown", 1.0)
    _set_feature(
        features,
        GLOBAL_F,
        "off_board_tile_count_norm",
        _normalize_count(_off_board_tile_count(observation), 20),
    )
    _set_feature(
        features,
        GLOBAL_F,
        "off_board_legal_action_count_norm",
        _normalize_count(_off_board_legal_action_count(legal_actions or []), 50),
    )
    return features


def encode_history(observation: dict[str, object]) -> list[list[float]]:
    history = observation.get("history", [])
    encoded = [[0.0 for _ in HISTORY_FEATURE_NAMES] for _event in range(MAX_HISTORY)]
    if not isinstance(history, list):
        return encoded
    recent = [event for event in history[-MAX_HISTORY:] if isinstance(event, dict)]
    offset = MAX_HISTORY - len(recent)
    player_id = _int_or_zero(observation.get("player_id"))
    num_players = _int_or_zero(observation.get("num_players"))
    if num_players <= 0:
        num_players = MAX_PLAYERS
    for recent_index, event in enumerate(recent):
        row_index = offset + recent_index
        encoded[row_index] = _encode_event(
            event,
            player_id,
            num_players,
            age_index=len(recent) - 1 - recent_index,
        )
    return encoded


def _encode_event(
    event: dict[str, object],
    observer_id: int,
    num_players: int,
    age_index: int,
) -> list[float]:
    row = [0.0 for _ in HISTORY_FEATURE_NAMES]
    _set_feature(row, HISTORY_F, "present", 1.0)
    _set_feature(row, HISTORY_F, "age_norm", _normalize_count(age_index, MAX_HISTORY - 1))
    action_type = event.get("action_type")
    if isinstance(action_type, str):
        _set_feature(row, HISTORY_F, f"action_{action_type}", 1.0)
    card = event.get("card")
    if isinstance(card, dict):
        card_type = card.get("type")
        if isinstance(card_type, str):
            _set_feature(row, HISTORY_F, f"card_type_{card_type}", 1.0)
    actor = event.get("actor")
    if isinstance(actor, int):
        _set_feature(row, HISTORY_F, "actor_present", 1.0)
        _set_feature(
            row,
            HISTORY_F,
            "actor_relative_norm",
            _normalize_count((actor - observer_id) % num_players, max(1, num_players - 1)),
        )
    target = event.get("target_player")
    if isinstance(target, int):
        _set_feature(row, HISTORY_F, "target_present", 1.0)
        _set_feature(
            row,
            HISTORY_F,
            "target_relative_norm",
            _normalize_count((target - observer_id) % num_players, max(1, num_players - 1)),
        )
    tool = event.get("tool")
    if isinstance(tool, str):
        _set_feature(row, HISTORY_F, f"tool_{tool}", 1.0)
    x = event.get("x")
    y = event.get("y")
    if isinstance(x, int) and isinstance(y, int):
        _set_feature(row, HISTORY_F, "coord_present", 1.0)
        _set_feature(row, HISTORY_F, "x_norm", _normalize_x(x))
        _set_feature(row, HISTORY_F, "y_norm", _normalize_y(y))
    goal_index = event.get("goal_index")
    if isinstance(goal_index, int) and 0 <= goal_index <= 2:
        _set_feature(row, HISTORY_F, f"goal_{goal_index}", 1.0)
    revealed_goal_kind = event.get("revealed_goal_kind")
    if revealed_goal_kind == GoalKind.GOLD.value:
        _set_feature(row, HISTORY_F, "revealed_gold", 1.0)
    elif revealed_goal_kind == GoalKind.STONE.value:
        _set_feature(row, HISTORY_F, "revealed_stone", 1.0)
    return row


def _encode_card(card: dict[str, object]) -> list[float]:
    row = [0.0 for _ in HAND_FEATURE_NAMES]
    _set_feature(row, HAND_F, "present", 1.0)
    card_type = card.get("type")
    if isinstance(card_type, str):
        _set_feature(row, HAND_F, f"card_type_{card_type}", 1.0)
    tools = card.get("tools", [])
    if isinstance(tools, list):
        for tool in TOOL_NAMES:
            _set_feature(row, HAND_F, f"tool_{tool}", 1.0 if tool in tools else 0.0)
    edges = _rotated_edges_from_card(card, 0)
    for direction in DIRECTIONS:
        _set_feature(row, HAND_F, f"edge_{direction}", 1.0 if direction in edges else 0.0)
    for pair in _connection_pairs_from_card(card, 0):
        _set_feature(row, HAND_F, f"connects_{pair[0]}_{pair[1]}", 1.0)
    groups = card.get("groups", [])
    group_count = len(groups) if isinstance(groups, list) else 0
    _set_feature(row, HAND_F, "group_count_norm", _normalize_count(group_count, 4))
    card_id = card.get("id")
    _set_feature(
        row,
        HAND_F,
        "is_dead",
        1.0 if isinstance(card_id, str) and card_id.startswith("dead") else 0.0,
    )
    _set_feature(row, HAND_F, "is_split", 1.0 if group_count > 1 else 0.0)
    _set_feature(row, HAND_F, "edge_count_norm", _normalize_count(len(edges), 4))
    return row


def _coord_to_index(x: int, y: int) -> tuple[int, int]:
    return y - BOARD_MIN_Y, x - BOARD_MIN_X


def _coord_to_index_if_visible(x: object, y: object) -> tuple[int, int] | None:
    if not isinstance(x, int) or not isinstance(y, int):
        return None
    if not (BOARD_MIN_X <= x <= BOARD_MAX_X and BOARD_MIN_Y <= y <= BOARD_MAX_Y):
        return None
    return _coord_to_index(x, y)


def _board_structure_features(
    observation: dict[str, object],
) -> tuple[dict[tuple[int, int], set[str]], dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    tiles = _public_tile_map(observation)
    start_groups = _tile_groups(tiles.get(START_COORD))
    if not start_groups:
        return {}, {}, {}

    reachable_open_edges: dict[tuple[int, int], set[str]] = {}
    frontier_distances: dict[tuple[int, int], int] = {}
    tile_distances: dict[tuple[int, int], int] = {START_COORD: 0}
    reached_nodes = {(START_COORD, group_index) for group_index in range(len(start_groups))}
    frontier = deque((START_COORD, group_index, 0) for group_index in range(len(start_groups)))

    while frontier:
        coord, group_index, distance = frontier.popleft()
        groups = _tile_groups(tiles.get(coord))
        if group_index >= len(groups):
            continue
        for direction in groups[group_index]:
            delta = DIRECTION_DELTAS[direction]
            neighbor_coord = (coord[0] + delta[0], coord[1] + delta[1])
            neighbor = tiles.get(neighbor_coord)
            if neighbor is None:
                reachable_open_edges.setdefault(coord, set()).add(direction)
                next_distance = distance + 1
                old_distance = frontier_distances.get(neighbor_coord)
                if old_distance is None or next_distance < old_distance:
                    frontier_distances[neighbor_coord] = next_distance
                continue

            opposite = OPPOSITE_DIRECTION[direction]
            for neighbor_group_index, neighbor_group in enumerate(_tile_groups(neighbor)):
                if opposite not in neighbor_group:
                    continue
                node = (neighbor_coord, neighbor_group_index)
                next_distance = distance + 1
                old_tile_distance = tile_distances.get(neighbor_coord)
                if old_tile_distance is None or next_distance < old_tile_distance:
                    tile_distances[neighbor_coord] = next_distance
                if node in reached_nodes:
                    continue
                reached_nodes.add(node)
                frontier.append((neighbor_coord, neighbor_group_index, next_distance))

    return reachable_open_edges, frontier_distances, tile_distances


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
    rotation = _rotation_or_zero(tile.get("rotation", 0))
    return _rotated_groups_from_card(card, rotation)


def _normalize_x(x: int) -> float:
    return _normalize_range(x, BOARD_MIN_X, BOARD_MAX_X)


def _normalize_y(y: int) -> float:
    return _normalize_range(y, BOARD_MIN_Y, BOARD_MAX_Y)


def _set_channel(
    tensor: list[list[list[float]]],
    name: str,
    row: int,
    col: int,
    value: float,
) -> None:
    tensor[BOARD_CH[name]][row][col] = value


def _set_feature(row: list[float], index: dict[str, int], name: str, value: float) -> None:
    row[index[name]] = value


def _known_goals(observation: dict[str, object]) -> dict[int, str]:
    known = observation.get("known_goals", {})
    if not isinstance(known, dict):
        return {}
    result: dict[int, str] = {}
    for raw_index, raw_kind in known.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if raw_kind in {GoalKind.GOLD.value, GoalKind.STONE.value}:
            result[index] = str(raw_kind)
    return result


def _board_goal_knowledge(observation: dict[str, object]) -> dict[int, str]:
    board = observation.get("board", [])
    if not isinstance(board, list):
        return {}
    result: dict[int, str] = {}
    for tile in board:
        if not isinstance(tile, dict) or tile.get("kind") != "goal":
            continue
        if not tile.get("revealed"):
            continue
        goal_index = tile.get("goal_index")
        goal_kind = tile.get("goal_kind")
        if isinstance(goal_index, int) and goal_kind in {GoalKind.GOLD.value, GoalKind.STONE.value}:
            result[goal_index] = str(goal_kind)
    return result


def _off_board_tile_count(observation: dict[str, object]) -> int:
    board = observation.get("board", [])
    if not isinstance(board, list):
        return 0
    count = 0
    for tile in board:
        if not isinstance(tile, dict):
            continue
        if _coord_to_index_if_visible(tile.get("x"), tile.get("y")) is None:
            count += 1
    return count


def _off_board_legal_action_count(legal_actions: list[Action]) -> int:
    count = 0
    for action in legal_actions:
        if isinstance(action, (PlayPath, Rockfall)):
            if _coord_to_index_if_visible(action.x, action.y) is None:
                count += 1
    return count


def _int_or_zero(value: object) -> int:
    return value if isinstance(value, int) else 0


def _flatten_2d(values: list[list[float]]) -> list[float]:
    return [item for row in values for item in row]


def _flatten_3d(values: list[list[list[float]]]) -> list[float]:
    return [item for channel in values for row in channel for item in row]
