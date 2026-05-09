"""Transparent heuristic baselines for simulator evaluation."""

from __future__ import annotations

from random import Random
from typing import Callable, Iterable

from saboter.actions import (
    Action,
    Discard,
    MapGoal,
    PlayPath,
    RepairTool,
    Rockfall,
    SabotageTool,
)
from saboter.board import GOAL_COORDS
from saboter.cards import CardType, GoalKind, Role
from saboter.env import SaboteurEnv


ScoreFn = Callable[[Action], float]


def _choose_best(actions: Iterable[Action], score_fn: ScoreFn, rng: Random) -> Action | None:
    best_score: float | None = None
    best_actions: list[Action] = []
    for action in actions:
        score = score_fn(action)
        if best_score is None or score > best_score:
            best_score = score
            best_actions = [action]
        elif score == best_score:
            best_actions.append(action)
    if not best_actions:
        return None
    return rng.choice(best_actions)


def _hand_card_type(obs: dict[str, object], action: Action) -> str | None:
    hand = obs["hand"]
    if not isinstance(hand, list):
        return None
    card = hand[action.card_slot]
    if not isinstance(card, dict):
        return None
    card_type = card.get("type")
    return card_type if isinstance(card_type, str) else None


def _hand_card_id(obs: dict[str, object], action: Action) -> str:
    card = _hand_card(obs, action)
    if card is None:
        return ""
    card_id = card.get("id")
    return card_id if isinstance(card_id, str) else ""


def _hand_card(obs: dict[str, object], action: Action) -> dict[str, object] | None:
    hand = obs["hand"]
    if not isinstance(hand, list):
        return None
    card = hand[action.card_slot]
    if not isinstance(card, dict):
        return None
    return card


def _known_goal_targets(obs: dict[str, object]) -> tuple[list[tuple[int, int]], bool]:
    known_goals = obs.get("known_goals", {})
    known_stones: set[int] = set()
    if isinstance(known_goals, dict):
        for raw_index, raw_kind in known_goals.items():
            index = int(raw_index)
            if raw_kind == GoalKind.GOLD.value:
                return [GOAL_COORDS[index]], True
            if raw_kind == GoalKind.STONE.value:
                known_stones.add(index)

    board = obs.get("board", [])
    targets: list[tuple[int, int]] = []
    if isinstance(board, list):
        for tile in board:
            if not isinstance(tile, dict) or tile.get("kind") != "goal":
                continue
            goal_index = tile.get("goal_index")
            if not isinstance(goal_index, int):
                continue
            if tile.get("goal_kind") == GoalKind.GOLD.value:
                return [GOAL_COORDS[goal_index]], True
            if tile.get("goal_kind") == GoalKind.STONE.value:
                known_stones.add(goal_index)
            elif goal_index not in known_stones:
                targets.append(GOAL_COORDS[goal_index])
    center_goal_index = 1
    if center_goal_index not in known_stones:
        return [GOAL_COORDS[center_goal_index]], False
    if targets:
        return targets, False
    return list(GOAL_COORDS), False


def _unknown_goal_indices(obs: dict[str, object]) -> list[int]:
    known_goals = obs.get("known_goals", {})
    known_by_map = {int(index) for index in known_goals} if isinstance(known_goals, dict) else set()
    unknown: list[int] = []
    board = obs.get("board", [])
    if not isinstance(board, list):
        return list(range(len(GOAL_COORDS)))
    for tile in board:
        if not isinstance(tile, dict) or tile.get("kind") != "goal":
            continue
        goal_index = tile.get("goal_index")
        if not isinstance(goal_index, int):
            continue
        if tile.get("revealed") or goal_index in known_by_map:
            continue
        unknown.append(goal_index)
    return sorted(unknown)


def _distance_to_targets(x: int, y: int, targets: list[tuple[int, int]]) -> int:
    return min(abs(x - target_x) + abs(y - target_y) for target_x, target_y in targets)


def _rotated_edges(card: dict[str, object] | None, rotation: int) -> set[str]:
    if card is None:
        return set()
    edges = card.get("edges", [])
    if not isinstance(edges, list):
        return set()
    edge_set = {edge for edge in edges if isinstance(edge, str)}
    if rotation % 360 != 180:
        return edge_set
    opposite = {"N": "S", "E": "W", "S": "N", "W": "E"}
    return {opposite[edge] for edge in edge_set if edge in opposite}


def _path_quality_penalty(card: dict[str, object] | None) -> float:
    if card is None:
        return 0.0
    card_id = str(card.get("id", ""))
    groups = card.get("groups", [])
    if card_id.startswith("dead"):
        return 500.0
    if isinstance(groups, list):
        if len(groups) != 1:
            return 420.0
        if groups and isinstance(groups[0], list) and len(groups[0]) <= 1:
            return 360.0
    return 0.0


def _path_openness(card: dict[str, object] | None) -> int:
    if card is None:
        return 0
    edges = card.get("edges", [])
    if not isinstance(edges, list):
        return 0
    return sum(1 for edge in edges if isinstance(edge, str))


def _target_direction_bonus(action: PlayPath, card: dict[str, object] | None, targets: list[tuple[int, int]]) -> float:
    edges = _rotated_edges(card, action.rotation)
    best = 0.0
    for target_x, target_y in targets:
        desired: set[str] = set()
        if target_x > action.x:
            desired.add("E")
        elif target_x < action.x:
            desired.add("W")
        if target_y > action.y:
            desired.add("S")
        elif target_y < action.y:
            desired.add("N")
        best = max(best, len(edges & desired) * 28.0)
    return best


def _player_public(obs: dict[str, object], player_id: int) -> dict[str, object]:
    players = obs.get("players", [])
    if not isinstance(players, list):
        return {}
    for player in players:
        if isinstance(player, dict) and player.get("player_id") == player_id:
            return player
    return {}


class GreedyMinerAgent:
    """Miner-style baseline: repair, learn goals, then extend the reachable path."""

    def __init__(self, seed: int | None = None):
        self.rng = Random(seed)

    def act(self, env: SaboteurEnv, player_id: int | None = None) -> Action | None:
        resolved_player = env.agent_selection if player_id is None else player_id
        actions = env.legal_actions(resolved_player)
        if not actions:
            return None
        obs = env.observe(resolved_player)
        targets, has_known_gold = _known_goal_targets(obs)
        unknown_goals = _unknown_goal_indices(obs)

        def score(action: Action) -> float:
            if isinstance(action, RepairTool):
                if action.target_player == resolved_player:
                    return 600.0
                target = _player_public(obs, action.target_player)
                return 360.0 + float(target.get("hand_size", 0))
            if isinstance(action, MapGoal):
                if has_known_gold:
                    return 80.0
                unknown_bonus = 120.0 if action.goal_index in unknown_goals else 0.0
                center_bonus = 18.0 if action.goal_index == 1 else 0.0
                return 610.0 + unknown_bonus + center_bonus - action.goal_index
            if isinstance(action, PlayPath):
                card = _hand_card(obs, action)
                distance = _distance_to_targets(action.x, action.y, targets)
                lane_penalty = min(abs(action.y - target_y) for _target_x, target_y in targets) * 18.0
                return (
                    760.0
                    - distance * 45.0
                    - lane_penalty
                    + action.x * 7.0
                    + _target_direction_bonus(action, card, targets)
                    - _path_quality_penalty(card)
                )
            if isinstance(action, Rockfall):
                return 180.0 if self._rockfall_looks_helpful(obs, action) else 25.0
            if isinstance(action, SabotageTool):
                return 20.0
            if isinstance(action, Discard):
                return self._discard_score(obs, action)
            return 0.0

        return _choose_best(actions, score, self.rng)

    def _rockfall_looks_helpful(self, obs: dict[str, object], action: Rockfall) -> bool:
        board = obs.get("board", [])
        if not isinstance(board, list):
            return False
        for tile in board:
            if not isinstance(tile, dict):
                continue
            if tile.get("x") != action.x or tile.get("y") != action.y:
                continue
            card = tile.get("card")
            if not isinstance(card, dict):
                return False
            card_id = card.get("id")
            return isinstance(card_id, str) and card_id.startswith("dead")
        return False

    def _discard_score(self, obs: dict[str, object], action: Discard) -> float:
        card_type = _hand_card_type(obs, action)
        card_id = _hand_card_id(obs, action)
        card = _hand_card(obs, action)
        if card_type == CardType.SABOTAGE.value:
            return 430.0
        if card_type == CardType.MAP.value:
            return 40.0 if _unknown_goal_indices(obs) else 170.0
        if card_type == CardType.ROCKFALL.value:
            return 155.0
        if card_type == CardType.REPAIR.value:
            return 45.0
        if card_id.startswith("dead"):
            return 380.0
        if card_type == CardType.PATH.value:
            penalty = _path_quality_penalty(card)
            if penalty >= 420.0:
                return 330.0
            if _path_openness(card) <= 1:
                return 300.0
        return 30.0


class GreedySaboteurAgent:
    """Saboteur-style baseline: break likely miners, collapse progress, misdirect paths."""

    def __init__(self, seed: int | None = None):
        self.rng = Random(seed)

    def act(self, env: SaboteurEnv, player_id: int | None = None) -> Action | None:
        resolved_player = env.agent_selection if player_id is None else player_id
        actions = env.legal_actions(resolved_player)
        if not actions:
            return None
        obs = env.observe(resolved_player)
        targets, _has_known_gold = _known_goal_targets(obs)

        def score(action: Action) -> float:
            if isinstance(action, SabotageTool):
                if action.target_player == resolved_player:
                    return 75.0
                target = _player_public(obs, action.target_player)
                return 620.0 + float(target.get("hand_size", 0)) * 4.0
            if isinstance(action, Rockfall):
                return 520.0 + action.x * 8.0 - abs(action.y) * 2.0
            if isinstance(action, RepairTool):
                if action.target_player == resolved_player:
                    return 260.0
                return 30.0
            if isinstance(action, PlayPath):
                distance = _distance_to_targets(action.x, action.y, targets)
                return 170.0 + distance * 9.0 - action.x * 4.0 + abs(action.y) * 3.0
            if isinstance(action, MapGoal):
                return 110.0
            if isinstance(action, Discard):
                return self._discard_score(obs, action)
            return 0.0

        return _choose_best(actions, score, self.rng)

    def _discard_score(self, obs: dict[str, object], action: Discard) -> float:
        card_type = _hand_card_type(obs, action)
        if card_type == CardType.REPAIR.value:
            return 14.0
        if card_type == CardType.MAP.value:
            return 8.0
        return 2.0


class HeuristicRoleInferenceAgent:
    """Role-aware heuristic with public-history suspicion scores.

    The suspicion model only reads observations, not hidden environment state.
    It is intentionally small and inspectable so it can serve as a baseline and
    a debugging signal for later neural belief heads.
    """

    def __init__(self, seed: int | None = None):
        self.rng = Random(seed)
        self.miner = GreedyMinerAgent(seed=seed)
        self.saboteur = GreedySaboteurAgent(seed=seed)

    def act(self, env: SaboteurEnv, player_id: int | None = None) -> Action | None:
        resolved_player = env.agent_selection if player_id is None else player_id
        obs = env.observe(resolved_player)
        if obs["own_role"] == Role.SABOTEUR.value:
            return self.saboteur.act(env, resolved_player)

        actions = env.legal_actions(resolved_player)
        if not actions:
            return None
        suspicion = self.suspicion_scores(obs)
        miner_fallback = self.miner.act(env, resolved_player)

        def score(action: Action) -> float:
            if isinstance(action, RepairTool):
                if action.target_player == resolved_player:
                    return 700.0
                return 480.0 - suspicion[action.target_player] * 250.0
            if isinstance(action, SabotageTool):
                target_suspicion = suspicion[action.target_player]
                if target_suspicion >= 0.55:
                    return 460.0 + target_suspicion * 120.0
                return 10.0
            if miner_fallback == action:
                return 450.0
            return 0.0

        return _choose_best(actions, score, self.rng)

    def suspicion_scores(self, observation: dict[str, object]) -> dict[int, float]:
        players = observation.get("players", [])
        player_ids = [
            int(player["player_id"])
            for player in players
            if isinstance(player, dict) and isinstance(player.get("player_id"), int)
        ]
        scores = {player_id: 0.25 for player_id in player_ids}
        history = observation.get("history", [])
        if not isinstance(history, list):
            return scores
        for event in history:
            if not isinstance(event, dict):
                continue
            actor = event.get("actor")
            if not isinstance(actor, int) or actor not in scores:
                continue
            action_type = event.get("action_type")
            if action_type == "sabotage":
                scores[actor] += 0.28
                target = event.get("target_player")
                if isinstance(target, int) and target in scores:
                    scores[target] -= 0.08
            elif action_type == "repair":
                scores[actor] -= 0.18
                target = event.get("target_player")
                if isinstance(target, int) and target in scores:
                    scores[target] -= 0.05
            elif action_type == "rockfall":
                scores[actor] += 0.18
            elif action_type == "play_path":
                card = event.get("card")
                if isinstance(card, dict) and str(card.get("id", "")).startswith("dead"):
                    scores[actor] += 0.16
                else:
                    x = event.get("x")
                    if isinstance(x, int) and x >= 4:
                        scores[actor] -= 0.04
            elif action_type == "map_goal":
                scores[actor] -= 0.03

        observer_id = observation.get("player_id")
        if isinstance(observer_id, int) and observation.get("own_role") == Role.MINER.value:
            scores[observer_id] = 0.0
        return {player_id: min(1.0, max(0.0, score)) for player_id, score in scores.items()}


class RoleAwareHeuristicAgent:
    """Delegates to the miner or saboteur heuristic based on the agent's own role."""

    def __init__(self, seed: int | None = None):
        self.miner = GreedyMinerAgent(seed=seed)
        self.saboteur = GreedySaboteurAgent(seed=seed)

    def act(self, env: SaboteurEnv, player_id: int | None = None) -> Action | None:
        resolved_player = env.agent_selection if player_id is None else player_id
        obs = env.observe(resolved_player)
        if obs["own_role"] == Role.SABOTEUR.value:
            return self.saboteur.act(env, resolved_player)
        return self.miner.act(env, resolved_player)
