"""Fixed-width action encoders for action-scoring policies."""

from __future__ import annotations

from dataclasses import dataclass

from saboter.actions import (
    Action,
    Discard,
    MapGoal,
    PlayPath,
    RepairTool,
    Rockfall,
    SabotageTool,
)
from saboter.cards import Tool
from saboter.encoding_utils import (
    CONNECTION_PAIRS,
    DIRECTIONS,
    connection_pairs_from_card as _connection_pairs,
    normalize_count as _normalize_count,
    normalize_range as _normalize_range,
    rotated_edges_from_card as _rotated_edges,
)
from saboter.env import SaboteurEnv
from saboter.observation import (
    BOARD_MAX_X,
    BOARD_MAX_Y,
    BOARD_MIN_X,
    BOARD_MIN_Y,
    CARD_TYPE_NAMES,
    MAX_HAND_SIZE,
    MAX_PLAYERS,
    TOOL_NAMES,
)


ACTION_TYPE_NAMES = (
    "play_path",
    "sabotage_tool",
    "repair_tool",
    "map_goal",
    "rockfall",
    "discard",
)

ACTION_FEATURE_NAMES = (
    *(f"action_{action_type}" for action_type in ACTION_TYPE_NAMES),
    "card_slot_norm",
    *(f"card_slot_{slot}" for slot in range(MAX_HAND_SIZE)),
    *(f"card_type_{card_type}" for card_type in CARD_TYPE_NAMES),
    *(f"selected_tool_{tool}" for tool in TOOL_NAMES),
    *(f"card_tool_{tool}" for tool in TOOL_NAMES),
    "target_present",
    "target_relative_norm",
    *(f"target_relative_{relative_index}" for relative_index in range(MAX_PLAYERS)),
    "coord_present",
    "x_norm",
    "y_norm",
    "coord_in_board_window",
    "rotation_180",
    "goal_0",
    "goal_1",
    "goal_2",
    *(f"card_edge_{direction}" for direction in DIRECTIONS),
    *(f"card_connects_{pair[0]}_{pair[1]}" for pair in CONNECTION_PAIRS),
    "card_edge_count_norm",
    "card_is_dead",
    "card_is_split",
)

ACTION_F = {name: index for index, name in enumerate(ACTION_FEATURE_NAMES)}


@dataclass(frozen=True)
class ActionFeatures:
    action: Action
    vector: list[float]
    feature_names: tuple[str, ...] = ACTION_FEATURE_NAMES
    shape: tuple[int] = (len(ACTION_FEATURE_NAMES),)


def encode_action(env: SaboteurEnv, player_id: int, action: Action) -> ActionFeatures:
    observation = env.observe(player_id)
    return encode_action_feature(observation, action)


def encode_actions(
    env: SaboteurEnv,
    player_id: int | None = None,
    legal_actions: list[Action] | None = None,
) -> list[ActionFeatures]:
    resolved_player = env.agent_selection if player_id is None else player_id
    observation = env.observe(resolved_player)
    actions = env.legal_actions(resolved_player) if legal_actions is None else legal_actions
    return encode_action_features(observation, actions)


def encode_action_feature(observation: dict[str, object], action: Action) -> ActionFeatures:
    """Encode one action against an already-built legal observation."""
    vector = encode_action_vector(observation, action)
    return ActionFeatures(action=action, vector=vector)


def encode_action_features(
    observation: dict[str, object],
    legal_actions: list[Action],
) -> list[ActionFeatures]:
    """Encode a legal action batch without rebuilding the observation."""
    return [
        ActionFeatures(action=action, vector=encode_action_vector(observation, action))
        for action in legal_actions
    ]


def encode_legal_action_batch(env: SaboteurEnv, player_id: int | None = None) -> list[list[float]]:
    return [features.vector for features in encode_actions(env, player_id)]


def encode_action_vector(observation: dict[str, object], action: Action) -> list[float]:
    vector = [0.0 for _ in ACTION_FEATURE_NAMES]
    action_name = _action_name(action)
    _set_feature(vector, f"action_{action_name}", 1.0)
    _set_feature(vector, "card_slot_norm", _normalize_count(action.card_slot, MAX_HAND_SIZE - 1))
    if 0 <= action.card_slot < MAX_HAND_SIZE:
        _set_feature(vector, f"card_slot_{action.card_slot}", 1.0)

    card = _card_for_action(observation, action)
    if card is not None:
        card_type = card.get("type")
        if isinstance(card_type, str) and card_type in CARD_TYPE_NAMES:
            _set_feature(vector, f"card_type_{card_type}", 1.0)
        tools = card.get("tools", [])
        if isinstance(tools, list):
            for tool in TOOL_NAMES:
                _set_feature(vector, f"card_tool_{tool}", 1.0 if tool in tools else 0.0)
        edges = _rotated_edges(card, getattr(action, "rotation", 0))
        for direction in DIRECTIONS:
            _set_feature(vector, f"card_edge_{direction}", 1.0 if direction in edges else 0.0)
        for pair in _connection_pairs(card, getattr(action, "rotation", 0)):
            _set_feature(vector, f"card_connects_{pair[0]}_{pair[1]}", 1.0)
        _set_feature(vector, "card_edge_count_norm", _normalize_count(len(edges), 4))
        card_id = card.get("id")
        _set_feature(
            vector,
            "card_is_dead",
            1.0 if isinstance(card_id, str) and card_id.startswith("dead") else 0.0,
        )
        groups = card.get("groups", [])
        _set_feature(vector, "card_is_split", 1.0 if isinstance(groups, list) and len(groups) > 1 else 0.0)

    if isinstance(action, (SabotageTool, RepairTool)):
        _set_tool(vector, action.tool)
        _set_target(vector, observation, action.target_player)
    elif isinstance(action, PlayPath):
        _set_coord(vector, action.x, action.y)
        _set_feature(vector, "rotation_180", 1.0 if action.rotation % 360 == 180 else 0.0)
    elif isinstance(action, Rockfall):
        _set_coord(vector, action.x, action.y)
    elif isinstance(action, MapGoal):
        if 0 <= action.goal_index <= 2:
            _set_feature(vector, f"goal_{action.goal_index}", 1.0)
    elif isinstance(action, Discard):
        pass
    return vector


def _action_name(action: Action) -> str:
    if isinstance(action, PlayPath):
        return "play_path"
    if isinstance(action, SabotageTool):
        return "sabotage_tool"
    if isinstance(action, RepairTool):
        return "repair_tool"
    if isinstance(action, MapGoal):
        return "map_goal"
    if isinstance(action, Rockfall):
        return "rockfall"
    if isinstance(action, Discard):
        return "discard"
    raise TypeError(f"Unsupported action type: {action!r}")


def _card_for_action(observation: dict[str, object], action: Action) -> dict[str, object] | None:
    hand = observation.get("hand", [])
    if not isinstance(hand, list) or not 0 <= action.card_slot < len(hand):
        return None
    card = hand[action.card_slot]
    return card if isinstance(card, dict) else None


def _set_coord(vector: list[float], x: int, y: int) -> None:
    _set_feature(vector, "coord_present", 1.0)
    _set_feature(vector, "x_norm", _normalize_range(x, BOARD_MIN_X, BOARD_MAX_X))
    _set_feature(vector, "y_norm", _normalize_range(y, BOARD_MIN_Y, BOARD_MAX_Y))
    _set_feature(
        vector,
        "coord_in_board_window",
        1.0 if BOARD_MIN_X <= x <= BOARD_MAX_X and BOARD_MIN_Y <= y <= BOARD_MAX_Y else 0.0,
    )


def _set_target(vector: list[float], observation: dict[str, object], target_player: int) -> None:
    _set_feature(vector, "target_present", 1.0)
    observer = observation.get("player_id")
    observer_id = observer if isinstance(observer, int) else 0
    num_players = observation.get("num_players")
    player_count = num_players if isinstance(num_players, int) and num_players > 1 else MAX_PLAYERS
    relative = (target_player - observer_id) % player_count
    _set_feature(
        vector,
        "target_relative_norm",
        _normalize_count(relative, player_count - 1),
    )
    if 0 <= relative < MAX_PLAYERS:
        _set_feature(vector, f"target_relative_{relative}", 1.0)


def _set_tool(vector: list[float], tool: Tool) -> None:
    _set_feature(vector, f"selected_tool_{tool.value}", 1.0)


def _set_feature(vector: list[float], name: str, value: float) -> None:
    vector[ACTION_F[name]] = value
