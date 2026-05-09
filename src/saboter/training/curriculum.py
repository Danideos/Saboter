"""Small curriculum helpers shared by rollout and evaluation code."""

from __future__ import annotations

from saboter.actions import Action, Discard, MapGoal, PlayPath


def filter_actions_for_training_mode(
    actions: list[Action],
    training_mode: str,
) -> list[Action]:
    """Limit legal actions for narrow curricula without changing env rules."""

    if training_mode != "miners_only":
        return actions

    preferred = [
        action
        for action in actions
        if isinstance(action, (PlayPath, Discard, MapGoal))
    ]
    return preferred or actions
