"""Progress diagnostics for Saboteur rollout states and games."""

from __future__ import annotations

from dataclasses import dataclass

from saboter.board import GOAL_COORDS
from saboter.cards import GoalKind
from saboter.env import Outcome, SaboteurEnv
from saboter.observation import _board_structure_features


@dataclass(frozen=True)
class DecisionProgress:
    reachable_tiles: float
    frontier_empty_cells: float
    min_distance_to_goal: float
    private_goal_knowledge_count: float


@dataclass(frozen=True)
class GameProgress:
    public_stone_reaches: float
    gold_reaches: float


def decision_progress_from_observation(observation: dict[str, object]) -> DecisionProgress:
    board = observation.get("board", [])
    reachable_coords: set[tuple[int, int]] = set()
    if isinstance(board, list):
        for tile in board:
            if not isinstance(tile, dict) or not tile.get("reachable"):
                continue
            kind = tile.get("kind")
            if kind not in {"start", "path", "goal"}:
                continue
            x = tile.get("x")
            y = tile.get("y")
            if isinstance(x, int) and isinstance(y, int):
                reachable_coords.add((x, y))

    _open_edges, frontier_distances, _tile_distances = _board_structure_features(observation)
    frontier_coords = set(frontier_distances)
    candidate_coords = reachable_coords | frontier_coords
    min_distance = 0.0
    if candidate_coords:
        min_distance = float(
            min(
                abs(x - goal_x) + abs(y - goal_y)
                for x, y in candidate_coords
                for goal_x, goal_y in GOAL_COORDS
            )
        )

    known_goals = observation.get("known_goals", {})
    known_count = float(len(known_goals)) if isinstance(known_goals, dict) else 0.0
    return DecisionProgress(
        reachable_tiles=float(len(reachable_coords)),
        frontier_empty_cells=float(len(frontier_coords)),
        min_distance_to_goal=min_distance,
        private_goal_knowledge_count=known_count,
    )


def game_progress_from_env(env: SaboteurEnv) -> GameProgress:
    public_stone_reaches = sum(
        1
        for event in env.history
        if event.action_type == "reveal_goal" and event.revealed_goal_kind == GoalKind.STONE.value
    )
    return GameProgress(
        public_stone_reaches=float(public_stone_reaches),
        gold_reaches=1.0 if env.outcome == Outcome.MINERS_WIN else 0.0,
    )
