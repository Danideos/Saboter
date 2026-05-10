"""Single-round, hidden-information-safe Saboteur environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from random import Random
from typing import Any

from saboter.actions import (
    Action,
    Discard,
    MapGoal,
    PlayPath,
    RepairTool,
    Rockfall,
    SabotageTool,
)
from saboter.board import Board, GOAL_COORDS
from saboter.cards import (
    Card,
    CardType,
    GOAL_GOLD_CARD,
    GOAL_STONE_CARDS,
    GoalKind,
    Role,
    Tool,
    build_draw_deck,
    hand_size_for_player_count,
    role_pool_for_player_count,
)


class Outcome(str, Enum):
    MINERS_WIN = "miners_win"
    SABOTEURS_WIN = "saboteurs_win"


@dataclass
class PlayerState:
    role: Role
    hand: list[Card]
    broken_tools: set[Tool]
    known_goals: dict[int, GoalKind]


@dataclass(frozen=True)
class PublicEvent:
    actor: int
    action_type: str
    card: dict[str, object] | None = None
    target_player: int | None = None
    tool: str | None = None
    x: int | None = None
    y: int | None = None
    rotation: int | None = None
    goal_index: int | None = None
    revealed_goal_kind: str | None = None
    removed_card: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class SaboteurEnv:
    """PettingZoo-style AEC surface for a single Saboteur round."""

    metadata = {"name": "saboteur_base_single_round_v0"}

    def __init__(self, num_players: int = 5):
        self.num_players = num_players
        self.rng = Random()
        self.players: list[PlayerState] = []
        self.unused_role: Role | None = None
        self.board: Board | None = None
        self.deck: list[Card] = []
        self.discard_pile: list[dict[str, object] | None] = []
        self.history: list[PublicEvent] = []
        self.agent_selection = 0
        self.turn_number = 0
        self.terminal = False
        self.outcome: Outcome | None = None
        self._rewards: dict[int, float] = {}
        self.reset(num_players=num_players)

    def reset(
        self,
        seed: int | None = None,
        num_players: int | None = None,
        force_roles: list[Role] | None = None,
    ) -> dict[str, Any]:
        if num_players is not None:
            self.num_players = num_players
        if self.num_players not in range(3, 11):
            raise ValueError("Saboteur supports 3 to 10 players")
        if seed is not None:
            self.rng = Random(seed)

        if force_roles is not None:
            if len(force_roles) != self.num_players:
                raise ValueError("force_roles must match num_players")
            dealt_roles = list(force_roles)
            self.unused_role = Role.SABOTEUR if Role.MINER in force_roles else Role.MINER
        else:
            role_pool = role_pool_for_player_count(self.num_players)
            self.rng.shuffle(role_pool)
            dealt_roles = role_pool[: self.num_players]
            self.unused_role = role_pool[self.num_players]

        goal_cards = [GOAL_GOLD_CARD, *GOAL_STONE_CARDS]
        self.rng.shuffle(goal_cards)
        self.board = Board(goal_cards)

        self.deck = build_draw_deck(self.rng)
        hand_size = hand_size_for_player_count(self.num_players)
        self.players = []
        for player_id in range(self.num_players):
            hand = [self.deck.pop(0) for _ in range(hand_size)]
            self.players.append(
                PlayerState(
                    role=dealt_roles[player_id],
                    hand=hand,
                    broken_tools=set(),
                    known_goals={},
                )
            )

        self.discard_pile = []
        self.history = []
        self.agent_selection = 0
        self.turn_number = 0
        self.terminal = False
        self.outcome = None
        self._rewards = {player_id: 0.0 for player_id in range(self.num_players)}
        self._check_saboteur_terminal()
        return self.observe(self.agent_selection)

    def observe(self, player_id: int) -> dict[str, Any]:
        self._validate_player_id(player_id)
        player = self.players[player_id]
        return {
            "player_id": player_id,
            "agent_selection": self.agent_selection,
            "num_players": self.num_players,
            "turn_number": self.turn_number,
            "terminal": self.terminal,
            "own_role": player.role.value,
            "hand": [card.public_dict() for card in player.hand],
            "own_broken_tools": sorted(tool.value for tool in player.broken_tools),
            "players": [self._public_player(player_id, other_id) for other_id in range(self.num_players)],
            "board": self._board().public_tiles(),
            "known_goals": {
                goal_index: goal_kind.value for goal_index, goal_kind in sorted(player.known_goals.items())
            },
            "deck_size": len(self.deck),
            "discard_count": len(self.discard_pile),
            "public_discards": list(self.discard_pile),
            "history": [event.to_dict() for event in self.history],
            "outcome": self.outcome.value if self.outcome is not None else None,
        }

    def legal_actions(self, player_id: int | None = None) -> list[Action]:
        if self.terminal:
            return []
        resolved_player = self.agent_selection if player_id is None else player_id
        return self._legal_actions_for(resolved_player, include_discards=True)

    def action_mask(self, player_id: int | None = None) -> list[int]:
        return [1] * len(self.legal_actions(player_id))

    def step(self, action: Action | None, *, validate: bool = True) -> None:
        if self.terminal:
            raise RuntimeError("Cannot step a terminal SaboteurEnv")

        player_id = self.agent_selection
        player = self.players[player_id]
        if action is None:
            if validate and player.hand:
                raise ValueError("Only players with no cards may step with None")
            self._advance_turn()
            self._check_saboteur_terminal(check_gold=False)
            return
        if validate:
            legal_actions = self.legal_actions(player_id)
            if action not in legal_actions:
                raise ValueError(f"Illegal action for player {player_id}: {action!r}")

        card = player.hand.pop(action.card_slot)
        if isinstance(action, Discard):
            self.discard_pile.append(None)
            self.history.append(PublicEvent(actor=player_id, action_type="discard"))
        elif isinstance(action, PlayPath):
            revealed = self._board().place_path(card, (action.x, action.y), action.rotation)
            self.history.append(
                PublicEvent(
                    actor=player_id,
                    action_type="play_path",
                    card=card.public_dict(),
                    x=action.x,
                    y=action.y,
                    rotation=action.rotation,
                )
            )
            for goal_index in revealed:
                goal_tile = self._board().tile_at(GOAL_COORDS[goal_index])
                goal_kind = goal_tile.card.goal_kind if goal_tile is not None else None
                goal_x, goal_y = GOAL_COORDS[goal_index]
                self.history.append(
                    PublicEvent(
                        actor=player_id,
                        action_type="reveal_goal",
                        card=goal_tile.card.public_dict() if goal_tile is not None else None,
                        x=goal_x,
                        y=goal_y,
                        rotation=goal_tile.rotation if goal_tile is not None else 0,
                        goal_index=goal_index,
                        revealed_goal_kind=goal_kind.value if goal_kind is not None else None,
                    )
                )
            if self._board().has_reached_gold():
                self._finish(Outcome.MINERS_WIN)
                return
        elif isinstance(action, SabotageTool):
            self.players[action.target_player].broken_tools.add(action.tool)
            self.discard_pile.append(card.public_dict())
            self.history.append(
                PublicEvent(
                    actor=player_id,
                    action_type="sabotage",
                    card=card.public_dict(),
                    target_player=action.target_player,
                    tool=action.tool.value,
                )
            )
        elif isinstance(action, RepairTool):
            self.players[action.target_player].broken_tools.remove(action.tool)
            self.discard_pile.append(card.public_dict())
            self.history.append(
                PublicEvent(
                    actor=player_id,
                    action_type="repair",
                    card=card.public_dict(),
                    target_player=action.target_player,
                    tool=action.tool.value,
                )
            )
        elif isinstance(action, MapGoal):
            goal_tile = self._board().tile_at(GOAL_COORDS[action.goal_index])
            if goal_tile is None or goal_tile.card.goal_kind is None:
                raise RuntimeError("Goal card is missing")
            player.known_goals[action.goal_index] = goal_tile.card.goal_kind
            self.discard_pile.append(card.public_dict())
            self.history.append(
                PublicEvent(
                    actor=player_id,
                    action_type="map_goal",
                    card=card.public_dict(),
                    goal_index=action.goal_index,
                )
            )
        elif isinstance(action, Rockfall):
            removed = self._board().remove_path((action.x, action.y))
            self.discard_pile.append(card.public_dict())
            self.discard_pile.append(removed.public_dict())
            self.history.append(
                PublicEvent(
                    actor=player_id,
                    action_type="rockfall",
                    card=card.public_dict(),
                    x=action.x,
                    y=action.y,
                    removed_card=removed.public_dict(),
                )
            )
        else:
            raise TypeError(f"Unsupported action: {action!r}")

        self._draw_if_possible(player)
        self.turn_number += 1
        self._check_saboteur_terminal(check_gold=False)
        if not self.terminal:
            self._advance_turn()

    def step_known_legal(self, action: Action | None) -> None:
        """Apply an action that was already produced by legal_actions()."""
        self.step(action, validate=False)

    def is_terminal(self) -> bool:
        return self.terminal

    def rewards(self) -> dict[int, float]:
        return dict(self._rewards)

    def _public_player(self, observer_id: int, player_id: int) -> dict[str, object]:
        player = self.players[player_id]
        return {
            "player_id": player_id,
            "is_self": player_id == observer_id,
            "relative_position": (player_id - observer_id) % self.num_players,
            "hand_size": len(player.hand),
            "broken_tools": sorted(tool.value for tool in player.broken_tools),
        }

    def _legal_actions_for(self, player_id: int, include_discards: bool) -> list[Action]:
        self._validate_player_id(player_id)
        player = self.players[player_id]
        actions: list[Action] = []
        for slot, card in enumerate(player.hand):
            if include_discards:
                actions.append(Discard(slot))
            if card.type == CardType.PATH:
                if player.broken_tools:
                    continue
                for x, y in self._candidate_path_coords():
                    for rotation in (0, 180):
                        if self._board().can_place_path(card, (x, y), rotation):
                            actions.append(PlayPath(slot, x, y, rotation))
            elif card.type == CardType.SABOTAGE:
                for target_player in range(self.num_players):
                    for tool in card.tools:
                        if tool not in self.players[target_player].broken_tools:
                            actions.append(SabotageTool(slot, target_player, tool))
            elif card.type == CardType.REPAIR:
                for target_player in range(self.num_players):
                    for tool in sorted(card.tools, key=lambda item: item.value):
                        if tool in self.players[target_player].broken_tools:
                            actions.append(RepairTool(slot, target_player, tool))
            elif card.type == CardType.MAP:
                for goal_index, coord in enumerate(GOAL_COORDS):
                    goal = self._board().tile_at(coord)
                    if goal is not None and not goal.revealed:
                        actions.append(MapGoal(slot, goal_index))
            elif card.type == CardType.ROCKFALL:
                for x, y in self._board().removable_path_coords():
                    actions.append(Rockfall(slot, x, y))
        return actions

    def _candidate_path_coords(self) -> list[tuple[int, int]]:
        candidates: set[tuple[int, int]] = set()
        for coord in self._board().tiles:
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                candidate = (coord[0] + dx, coord[1] + dy)
                if self._board().is_empty(candidate):
                    candidates.add(candidate)
        return sorted(candidates)

    def _draw_if_possible(self, player: PlayerState) -> None:
        if self.deck:
            player.hand.append(self.deck.pop(0))

    def _advance_turn(self) -> None:
        self.agent_selection = (self.agent_selection + 1) % self.num_players

    def _check_saboteur_terminal(self, *, check_gold: bool = True) -> None:
        if self.terminal:
            return
        if check_gold and self._board().has_reached_gold():
            self._finish(Outcome.MINERS_WIN)
            return
        if self.deck:
            return
        for player_id in range(self.num_players):
            if self._legal_actions_for(player_id, include_discards=False):
                return
        self._finish(Outcome.SABOTEURS_WIN)

    def _finish(self, outcome: Outcome) -> None:
        self.terminal = True
        self.outcome = outcome
        rewards: dict[int, float] = {}
        for player_id, player in enumerate(self.players):
            if outcome == Outcome.MINERS_WIN:
                rewards[player_id] = 1.0 if player.role == Role.MINER else -1.0
            else:
                rewards[player_id] = 1.0 if player.role == Role.SABOTEUR else -1.0
        self._rewards = rewards

    def _board(self) -> Board:
        if self.board is None:
            raise RuntimeError("Environment has not been reset")
        return self.board

    def _validate_player_id(self, player_id: int) -> None:
        if not 0 <= player_id < self.num_players:
            raise ValueError(f"Invalid player id: {player_id}")
