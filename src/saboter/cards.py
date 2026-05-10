"""Card and deck definitions for base-game Saboteur."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import Iterable


class Direction(str, Enum):
    NORTH = "N"
    EAST = "E"
    SOUTH = "S"
    WEST = "W"


class CardType(str, Enum):
    PATH = "path"
    START = "start"
    GOAL = "goal"
    SABOTAGE = "sabotage"
    REPAIR = "repair"
    MAP = "map"
    ROCKFALL = "rockfall"


class GoalKind(str, Enum):
    GOLD = "gold"
    STONE = "stone"


class Role(str, Enum):
    MINER = "miner"
    SABOTEUR = "saboteur"


class Tool(str, Enum):
    PICKAXE = "pickaxe"
    LANTERN = "lantern"
    CART = "cart"


OPPOSITE: dict[Direction, Direction] = {
    Direction.NORTH: Direction.SOUTH,
    Direction.EAST: Direction.WEST,
    Direction.SOUTH: Direction.NORTH,
    Direction.WEST: Direction.EAST,
}

ROTATE_180: dict[Direction, Direction] = {
    Direction.NORTH: Direction.SOUTH,
    Direction.EAST: Direction.WEST,
    Direction.SOUTH: Direction.NORTH,
    Direction.WEST: Direction.EAST,
}


@dataclass(frozen=True)
class Card:
    """Immutable card specification.

    For path-like cards, ``groups`` describes internal tunnel connectivity.
    Each group contains the card edges connected by one uninterrupted tunnel
    component. A one-edge group is a dead end.
    """

    id: str
    type: CardType
    edges: frozenset[Direction] = frozenset()
    groups: tuple[frozenset[Direction], ...] = ()
    tools: frozenset[Tool] = frozenset()
    goal_kind: GoalKind | None = None

    def rotated_edges(self, rotation: int) -> frozenset[Direction]:
        return frozenset(_rotate_direction(direction, rotation) for direction in self.edges)

    def rotated_groups(self, rotation: int) -> tuple[frozenset[Direction], ...]:
        return tuple(
            frozenset(_rotate_direction(direction, rotation) for direction in group)
            for group in self.groups
        )

    def public_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.id,
            "type": self.type.value,
        }
        if self.type in {CardType.PATH, CardType.START, CardType.GOAL}:
            result["edges"] = sorted(direction.value for direction in self.edges)
            result["groups"] = [
                sorted(direction.value for direction in group) for group in self.groups
            ]
        if self.tools:
            result["tools"] = sorted(tool.value for tool in self.tools)
        if self.goal_kind is not None:
            result["goal_kind"] = self.goal_kind.value
        return result


def _rotate_direction(direction: Direction, rotation: int) -> Direction:
    normalized = rotation % 360
    if normalized == 0:
        return direction
    if normalized == 180:
        return ROTATE_180[direction]
    raise ValueError("Saboteur base cards only support 0 or 180 degree rotation")


def _path(
    card_id: str,
    groups: Iterable[Iterable[Direction]],
    count: int,
) -> tuple[Card, int]:
    normalized_groups = tuple(frozenset(group) for group in groups)
    edges = frozenset(direction for group in normalized_groups for direction in group)
    return Card(card_id, CardType.PATH, edges=edges, groups=normalized_groups), count


START_CARD = Card(
    "start",
    CardType.START,
    edges=frozenset(Direction),
    groups=(frozenset(Direction),),
)

GOAL_GOLD_CARD = Card(
    "goal_gold",
    CardType.GOAL,
    edges=frozenset(Direction),
    groups=(frozenset(Direction),),
    goal_kind=GoalKind.GOLD,
)

GOAL_STONE_NE_CARD = Card(
    "goal_stone_ne",
    CardType.GOAL,
    edges=frozenset((Direction.NORTH, Direction.EAST)),
    groups=(frozenset((Direction.NORTH, Direction.EAST)),),
    goal_kind=GoalKind.STONE,
)

GOAL_STONE_NW_CARD = Card(
    "goal_stone_nw",
    CardType.GOAL,
    edges=frozenset((Direction.NORTH, Direction.WEST)),
    groups=(frozenset((Direction.NORTH, Direction.WEST)),),
    goal_kind=GoalKind.STONE,
)

GOAL_STONE_CARDS = (GOAL_STONE_NE_CARD, GOAL_STONE_NW_CARD)

# Backward-compatible alias for older tests/scripts that imported one stone card.
GOAL_STONE_CARD = GOAL_STONE_NE_CARD


PATH_CARD_SPECS: tuple[tuple[Card, int], ...] = (
    _path("path_ns", ((Direction.NORTH, Direction.SOUTH),), 4),
    _path("path_nes", ((Direction.NORTH, Direction.EAST, Direction.SOUTH),), 5),
    _path("path_cross", ((Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST),), 5),
    _path("path_es", ((Direction.EAST, Direction.SOUTH),), 4),
    _path("path_sw", ((Direction.SOUTH, Direction.WEST),), 5),
    _path("dead_s", ((Direction.SOUTH,),), 1),
    _path("dead_ns_split", ((Direction.NORTH,), (Direction.SOUTH,)), 1),
    _path("dead_nw_split", ((Direction.NORTH,), (Direction.WEST,)), 1),
    _path("dead_es_split", ((Direction.EAST,), (Direction.SOUTH,)), 1),
    _path("dead_ew_split", ((Direction.EAST,), (Direction.WEST,)), 1),
    _path("dead_e", ((Direction.EAST,),), 1),
    _path("path_new", ((Direction.NORTH, Direction.EAST, Direction.WEST),), 5),
    _path("path_ew", ((Direction.EAST, Direction.WEST),), 3),
    _path("dead_n", ((Direction.NORTH,),), 1),
    _path("dead_w", ((Direction.WEST,),), 1),
    _path("path_ne", ((Direction.NORTH, Direction.EAST),), 1),
)


def path_card_by_id(card_id: str) -> Card:
    for card, _count in PATH_CARD_SPECS:
        if card.id == card_id:
            return card
    if card_id == START_CARD.id:
        return START_CARD
    if card_id == GOAL_GOLD_CARD.id:
        return GOAL_GOLD_CARD
    for goal_card in GOAL_STONE_CARDS:
        if card_id == goal_card.id:
            return goal_card
    if card_id == "goal_stone":
        return GOAL_STONE_CARD
    raise KeyError(card_id)


def action_card(card_type: CardType, tools: Iterable[Tool] = ()) -> Card:
    tool_set = frozenset(tools)
    suffix = "_".join(tool.value for tool in sorted(tool_set, key=lambda item: item.value))
    if card_type in {CardType.SABOTAGE, CardType.REPAIR}:
        if not tool_set:
            raise ValueError(f"{card_type.value} cards require at least one tool")
        card_id = f"{card_type.value}_{suffix}"
    elif card_type == CardType.MAP:
        card_id = "map"
    elif card_type == CardType.ROCKFALL:
        card_id = "rockfall"
    else:
        raise ValueError(f"Unsupported action card type: {card_type}")
    return Card(card_id, card_type, tools=tool_set)


SABOTAGE_CARDS: tuple[tuple[Card, int], ...] = tuple(
    (action_card(CardType.SABOTAGE, (tool,)), 3) for tool in Tool
)

REPAIR_CARDS: tuple[tuple[Card, int], ...] = (
    (action_card(CardType.REPAIR, (Tool.PICKAXE,)), 2),
    (action_card(CardType.REPAIR, (Tool.LANTERN,)), 2),
    (action_card(CardType.REPAIR, (Tool.CART,)), 2),
    (action_card(CardType.REPAIR, (Tool.PICKAXE, Tool.LANTERN)), 1),
    (action_card(CardType.REPAIR, (Tool.PICKAXE, Tool.CART)), 1),
    (action_card(CardType.REPAIR, (Tool.LANTERN, Tool.CART)), 1),
)

MAP_CARDS: tuple[tuple[Card, int], ...] = ((action_card(CardType.MAP), 6),)
ROCKFALL_CARDS: tuple[tuple[Card, int], ...] = ((action_card(CardType.ROCKFALL), 3),)


def build_path_deck() -> list[Card]:
    cards: list[Card] = []
    for card, count in PATH_CARD_SPECS:
        cards.extend([card] * count)
    return cards


def build_action_deck() -> list[Card]:
    cards: list[Card] = []
    for spec in (SABOTAGE_CARDS, REPAIR_CARDS, MAP_CARDS, ROCKFALL_CARDS):
        for card, count in spec:
            cards.extend([card] * count)
    return cards


def build_draw_deck(rng: Random) -> list[Card]:
    deck = build_path_deck() + build_action_deck()
    rng.shuffle(deck)
    return deck


def role_pool_for_player_count(num_players: int) -> list[Role]:
    if num_players not in range(3, 11):
        raise ValueError("Saboteur supports 3 to 10 players")
    composition = {
        3: (3, 1),
        4: (4, 1),
        5: (4, 2),
        6: (5, 2),
        7: (5, 3),
        8: (6, 3),
        9: (7, 3),
        10: (7, 4),
    }
    miners, saboteurs = composition[num_players]
    return [Role.MINER] * miners + [Role.SABOTEUR] * saboteurs


def hand_size_for_player_count(num_players: int) -> int:
    if 3 <= num_players <= 5:
        return 6
    if 6 <= num_players <= 7:
        return 5
    if 8 <= num_players <= 10:
        return 4
    raise ValueError("Saboteur supports 3 to 10 players")
