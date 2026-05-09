"""Action dataclasses for the Saboteur environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from saboter.cards import Tool


@dataclass(frozen=True)
class PlayPath:
    card_slot: int
    x: int
    y: int
    rotation: int = 0


@dataclass(frozen=True)
class SabotageTool:
    card_slot: int
    target_player: int
    tool: Tool


@dataclass(frozen=True)
class RepairTool:
    card_slot: int
    target_player: int
    tool: Tool


@dataclass(frozen=True)
class MapGoal:
    card_slot: int
    goal_index: int


@dataclass(frozen=True)
class Rockfall:
    card_slot: int
    x: int
    y: int


@dataclass(frozen=True)
class Discard:
    card_slot: int


Action: TypeAlias = PlayPath | SabotageTool | RepairTool | MapGoal | Rockfall | Discard

