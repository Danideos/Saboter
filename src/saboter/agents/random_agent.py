"""Legal-random Saboteur baseline."""

from __future__ import annotations

from random import Random

from saboter.actions import Action
from saboter.env import SaboteurEnv


class LegalRandomAgent:
    def __init__(self, seed: int | None = None):
        self.rng = Random(seed)

    def act(self, env: SaboteurEnv, player_id: int | None = None) -> Action | None:
        resolved_player = env.agent_selection if player_id is None else player_id
        actions = env.legal_actions(resolved_player)
        if not actions:
            return None
        return self.rng.choice(actions)

