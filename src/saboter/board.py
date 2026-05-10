"""Sparse Saboteur board and tunnel connectivity rules."""

from __future__ import annotations

from dataclasses import dataclass

from saboter.cards import Card, CardType, Direction, GoalKind, OPPOSITE, START_CARD


Coord = tuple[int, int]

DIRECTION_DELTAS: dict[Direction, Coord] = {
    Direction.NORTH: (0, -1),
    Direction.EAST: (1, 0),
    Direction.SOUTH: (0, 1),
    Direction.WEST: (-1, 0),
}

GOAL_COORDS: tuple[Coord, Coord, Coord] = ((8, -2), (8, 0), (8, 2))
START_COORD: Coord = (0, 0)


@dataclass
class Tile:
    card: Card
    rotation: int = 0
    revealed: bool = True
    goal_index: int | None = None

    @property
    def is_goal(self) -> bool:
        return self.card.type == CardType.GOAL

    @property
    def is_start(self) -> bool:
        return self.card.type == CardType.START

    @property
    def is_removable_path(self) -> bool:
        return self.card.type == CardType.PATH

    def edges(self) -> frozenset[Direction]:
        return self.card.rotated_edges(self.rotation)

    def groups(self) -> tuple[frozenset[Direction], ...]:
        return self.card.rotated_groups(self.rotation)


class Board:
    """Sparse coordinate board with hidden goal support."""

    def __init__(self, goal_cards: list[Card]):
        if len(goal_cards) != 3:
            raise ValueError("Exactly three goal cards are required")
        self.tiles: dict[Coord, Tile] = {
            START_COORD: Tile(START_CARD, revealed=True),
        }
        for index, coord in enumerate(GOAL_COORDS):
            self.tiles[coord] = Tile(goal_cards[index], revealed=False, goal_index=index)

    def copy(self) -> "Board":
        copied = object.__new__(Board)
        copied.tiles = {
            coord: Tile(tile.card, tile.rotation, tile.revealed, tile.goal_index)
            for coord, tile in self.tiles.items()
        }
        return copied

    def tile_at(self, coord: Coord) -> Tile | None:
        return self.tiles.get(coord)

    def is_empty(self, coord: Coord) -> bool:
        return coord not in self.tiles

    def neighbors(self, coord: Coord) -> list[tuple[Direction, Coord, Tile]]:
        result: list[tuple[Direction, Coord, Tile]] = []
        for direction, delta in DIRECTION_DELTAS.items():
            neighbor_coord = (coord[0] + delta[0], coord[1] + delta[1])
            neighbor = self.tiles.get(neighbor_coord)
            if neighbor is not None:
                result.append((direction, neighbor_coord, neighbor))
        return result

    def can_place_path(self, card: Card, coord: Coord, rotation: int) -> bool:
        if card.type != CardType.PATH:
            return False
        if rotation not in {0, 180}:
            return False
        if not self.is_empty(coord):
            return False

        adjacent_tiles = self.neighbors(coord)
        if not adjacent_tiles:
            return False

        edges = card.rotated_edges(rotation)
        for direction, _neighbor_coord, neighbor in adjacent_tiles:
            if neighbor.is_goal and not neighbor.revealed:
                continue
            neighbor_has_edge = OPPOSITE[direction] in neighbor.edges()
            if (direction in edges) != neighbor_has_edge:
                return False

        test_board = self.copy()
        test_board.tiles[coord] = Tile(card, rotation=rotation, revealed=True)
        return coord in test_board.reachable_path_coords()

    def place_path(self, card: Card, coord: Coord, rotation: int) -> list[int]:
        if not self.can_place_path(card, coord, rotation):
            raise ValueError(f"Illegal path placement at {coord}")
        self.tiles[coord] = Tile(card, rotation=rotation, revealed=True)
        return self.reveal_reached_goals_from(coord)

    def reveal_reached_goals_from(self, coord: Coord) -> list[int]:
        reached = self.reachable_goal_indices()
        revealed: list[int] = []
        for direction, neighbor_coord, neighbor in self.neighbors(coord):
            if neighbor.is_goal and not neighbor.revealed and neighbor.goal_index in reached:
                required_edge = OPPOSITE[direction]
                neighbor.rotation = self._revealed_goal_rotation(neighbor_coord, neighbor, required_edge)
                neighbor.revealed = True
                revealed.append(neighbor.goal_index)
        return revealed

    def reveal_goal(self, goal_index: int) -> None:
        coord = GOAL_COORDS[goal_index]
        tile = self.tiles[coord]
        if not tile.is_goal:
            raise ValueError(f"Coordinate {coord} is not a goal")
        tile.revealed = True

    def remove_path(self, coord: Coord) -> Card:
        tile = self.tiles.get(coord)
        if tile is None or not tile.is_removable_path:
            raise ValueError("Rockfall can only remove normal path cards")
        del self.tiles[coord]
        return tile.card

    def removable_path_coords(self) -> list[Coord]:
        return sorted(coord for coord, tile in self.tiles.items() if tile.is_removable_path)

    def _revealed_goal_rotation(
        self,
        coord: Coord,
        goal: Tile,
        required_edge: Direction,
    ) -> int:
        for rotation in (0, 180):
            edges = goal.card.rotated_edges(rotation)
            if required_edge not in edges:
                continue
            if self._goal_rotation_matches_revealed_neighbors(coord, edges):
                return rotation
        for rotation in (0, 180):
            if required_edge in goal.card.rotated_edges(rotation):
                return rotation
        return 0

    def _goal_rotation_matches_revealed_neighbors(
        self,
        coord: Coord,
        goal_edges: frozenset[Direction],
    ) -> bool:
        for direction, _neighbor_coord, neighbor in self.neighbors(coord):
            if neighbor.is_goal and not neighbor.revealed:
                continue
            neighbor_has_edge = OPPOSITE[direction] in neighbor.edges()
            if (direction in goal_edges) != neighbor_has_edge:
                return False
        return True

    def reachable_path_coords(self) -> set[Coord]:
        return {
            coord
            for coord, _group_index in self._reachable_group_nodes()
            if not self.tiles[coord].is_goal
        }

    def reachable_goal_indices(self) -> set[int]:
        return {
            self.tiles[coord].goal_index
            for coord, _group_index in self._reachable_group_nodes()
            if self.tiles[coord].is_goal and self.tiles[coord].goal_index is not None
        }

    def _reachable_group_nodes(self) -> set[tuple[Coord, int]]:
        start_tile = self.tiles[START_COORD]
        frontier = [(START_COORD, group_index) for group_index, _ in enumerate(start_tile.groups())]
        reached: set[tuple[Coord, int]] = set(frontier)

        while frontier:
            coord, group_index = frontier.pop()
            tile = self.tiles[coord]
            if tile.is_goal and not tile.revealed:
                continue
            group = tile.groups()[group_index]
            for direction in group:
                delta = DIRECTION_DELTAS[direction]
                neighbor_coord = (coord[0] + delta[0], coord[1] + delta[1])
                neighbor = self.tiles.get(neighbor_coord)
                if neighbor is None:
                    continue
                if neighbor.is_goal and not neighbor.revealed:
                    for neighbor_group_index, _neighbor_group in enumerate(neighbor.groups()):
                        node = (neighbor_coord, neighbor_group_index)
                        if node not in reached:
                            reached.add(node)
                            frontier.append(node)
                    continue
                opposite = OPPOSITE[direction]
                for neighbor_group_index, neighbor_group in enumerate(neighbor.groups()):
                    if opposite not in neighbor_group:
                        continue
                    node = (neighbor_coord, neighbor_group_index)
                    if node not in reached:
                        reached.add(node)
                        frontier.append(node)
        return reached

    def has_reached_gold(self) -> bool:
        for coord in GOAL_COORDS:
            tile = self.tiles[coord]
            if (
                tile.revealed
                and tile.card.goal_kind == GoalKind.GOLD
                and tile.goal_index in self.reachable_goal_indices()
            ):
                return True
        return False

    def public_tiles(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        reachable = self.reachable_path_coords()
        reachable_goals = self.reachable_goal_indices()
        for coord in sorted(self.tiles):
            tile = self.tiles[coord]
            item: dict[str, object] = {
                "x": coord[0],
                "y": coord[1],
                "rotation": tile.rotation,
                "revealed": tile.revealed,
                "reachable": coord in reachable
                or (tile.goal_index is not None and tile.goal_index in reachable_goals),
            }
            if tile.is_goal:
                item["kind"] = "goal"
                item["goal_index"] = tile.goal_index
                if tile.revealed:
                    item["goal_kind"] = tile.card.goal_kind.value
                    item["card"] = tile.card.public_dict()
                else:
                    item["goal_kind"] = None
                    item["card"] = None
            elif tile.is_start:
                item["kind"] = "start"
                item["card"] = tile.card.public_dict()
            else:
                item["kind"] = "path"
                item["card"] = tile.card.public_dict()
            result.append(item)
        return result
