"""Incremental frontier cache for heuristic rollout reward shaping."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

from saboter.actions import Action, PlayPath, Rockfall
from saboter.board import Board, Coord, DIRECTION_DELTAS, GOAL_COORDS, START_COORD
from saboter.cards import OPPOSITE


GoalDistanceSummary = tuple[float, float, float]


def goal_missing_distance(coord: Coord, goal_coord: Coord) -> float:
    """Estimate how many more tiles are needed from a frontier cell to a goal."""

    x, y = coord
    goal_x, goal_y = goal_coord
    x_cost = abs(goal_x - x)
    if goal_y == 0:
        return float(x_cost + abs(goal_y - y))
    branch_weight = min(1.0, max(0.0, (x - 4) / 4.0))
    return float(x_cost + branch_weight * abs(goal_y - y))


def frontier_goal_distance_summary(frontier_cells: set[Coord]) -> GoalDistanceSummary:
    if not frontier_cells:
        return (math.inf, math.inf, math.inf)

    best = [math.inf, math.inf, math.inf]
    for coord in frontier_cells:
        for goal_index, goal_coord in enumerate(GOAL_COORDS):
            distance = goal_missing_distance(coord, goal_coord)
            if distance < best[goal_index]:
                best[goal_index] = distance
    return (best[0], best[1], best[2])


def heuristic_path_reward(
    before_distances: GoalDistanceSummary,
    after_distances: GoalDistanceSummary,
) -> float:
    total_reduction = 0.0
    for before_value, after_value in zip(before_distances, after_distances):
        if not math.isfinite(before_value) or not math.isfinite(after_value):
            continue
        total_reduction += max(0.0, before_value - after_value)
    return total_reduction * 0.025


@dataclass
class HeuristicRewardTracker:
    reachable_nodes: set[tuple[Coord, int]] = field(default_factory=set)
    frontier_cells: set[Coord] = field(default_factory=set)
    frontier_scores: dict[Coord, GoalDistanceSummary] = field(default_factory=dict)
    goal_distances: GoalDistanceSummary = (math.inf, math.inf, math.inf)
    recompute_count: int = 0

    @classmethod
    def from_board(cls, board: Board) -> "HeuristicRewardTracker":
        tracker = cls()
        tracker.recompute(board)
        return tracker

    def current_goal_distances(self) -> GoalDistanceSummary:
        return self.goal_distances

    def apply_action(self, board: Board, action: Action | None) -> None:
        if isinstance(action, PlayPath):
            self._update_after_play_path(board, (action.x, action.y))
            return
        if isinstance(action, Rockfall):
            self.recompute(board)

    def recompute(self, board: Board) -> None:
        self.recompute_count += 1
        self.reachable_nodes.clear()
        self.frontier_cells.clear()
        self.frontier_scores.clear()

        start_tile = board.tile_at(START_COORD)
        if start_tile is None:
            self.goal_distances = (math.inf, math.inf, math.inf)
            return

        self._expand_from_queue(
            board,
            deque((START_COORD, group_index) for group_index, _group in enumerate(start_tile.groups())),
        )
        self._refresh_goal_distances()

    def _update_after_play_path(self, board: Board, coord: Coord) -> None:
        self.frontier_cells.discard(coord)
        self.frontier_scores.pop(coord, None)

        start_nodes = self._reachable_entry_nodes(board, coord)
        if not start_nodes:
            # Path placements should normally connect to the current reachable component.
            # Fall back to a full recompute if we hit an unexpected board state.
            self.recompute(board)
            return

        self._expand_from_queue(
            board,
            deque(node for node in start_nodes if node not in self.reachable_nodes),
        )
        self._refresh_goal_distances()

    def _reachable_entry_nodes(self, board: Board, coord: Coord) -> set[tuple[Coord, int]]:
        tile = board.tile_at(coord)
        if tile is None:
            return set()

        entry_nodes: set[tuple[Coord, int]] = set()
        for group_index, group in enumerate(tile.groups()):
            for direction in group:
                delta = DIRECTION_DELTAS[direction]
                neighbor_coord = (coord[0] + delta[0], coord[1] + delta[1])
                neighbor = board.tile_at(neighbor_coord)
                if neighbor is None or (neighbor.is_goal and not neighbor.revealed):
                    continue
                opposite = OPPOSITE[direction]
                for neighbor_group_index, neighbor_group in enumerate(neighbor.groups()):
                    if opposite not in neighbor_group:
                        continue
                    if (neighbor_coord, neighbor_group_index) in self.reachable_nodes:
                        entry_nodes.add((coord, group_index))
                        break
                if (coord, group_index) in entry_nodes:
                    break
        return entry_nodes

    def _expand_from_queue(self, board: Board, queue: deque[tuple[Coord, int]]) -> None:
        while queue:
            coord, group_index = queue.popleft()
            node = (coord, group_index)
            if node in self.reachable_nodes:
                continue

            tile = board.tile_at(coord)
            if tile is None:
                continue
            groups = tile.groups()
            if group_index >= len(groups):
                continue

            self.reachable_nodes.add(node)
            if tile.is_goal and not tile.revealed:
                continue

            for direction in groups[group_index]:
                delta = DIRECTION_DELTAS[direction]
                neighbor_coord = (coord[0] + delta[0], coord[1] + delta[1])
                neighbor = board.tile_at(neighbor_coord)
                if neighbor is None:
                    self._add_frontier_cell(neighbor_coord)
                    continue
                if neighbor.is_goal and not neighbor.revealed:
                    for neighbor_group_index, _neighbor_group in enumerate(neighbor.groups()):
                        queue.append((neighbor_coord, neighbor_group_index))
                    continue

                opposite = OPPOSITE[direction]
                for neighbor_group_index, neighbor_group in enumerate(neighbor.groups()):
                    if opposite in neighbor_group:
                        queue.append((neighbor_coord, neighbor_group_index))

    def _add_frontier_cell(self, coord: Coord) -> None:
        if coord in self.frontier_cells:
            return
        self.frontier_cells.add(coord)
        self.frontier_scores[coord] = tuple(
            goal_missing_distance(coord, goal_coord) for goal_coord in GOAL_COORDS
        )

    def _refresh_goal_distances(self) -> None:
        if not self.frontier_scores:
            self.goal_distances = (math.inf, math.inf, math.inf)
            return
        best = [math.inf, math.inf, math.inf]
        for scores in self.frontier_scores.values():
            for goal_index, distance in enumerate(scores):
                if distance < best[goal_index]:
                    best[goal_index] = distance
        self.goal_distances = (best[0], best[1], best[2])
