"""Small curriculum helpers shared by rollout and evaluation code."""

from __future__ import annotations

from saboter.actions import Action, Discard, MapGoal, PlayPath


def filter_actions_for_training_mode(
    actions: list[Action],
    training_mode: str,
    miners_only_actions: str = "path_discard_map",
) -> list[Action]:
    """Limit legal actions for narrow curricula without changing env rules."""

    if training_mode != "miners_only":
        return actions
    if miners_only_actions == "all":
        return actions
    if miners_only_actions != "path_discard_map":
        raise ValueError("miners_only_actions must be 'path_discard_map' or 'all'")

    preferred = [
        action
        for action in actions
        if isinstance(action, (PlayPath, Discard, MapGoal))
    ]
    return preferred or actions
