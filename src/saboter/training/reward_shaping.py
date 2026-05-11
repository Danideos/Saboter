"""Shaping-reward helpers shared by rollout collectors."""

from __future__ import annotations

from saboter.training.heuristic_frontier import GoalDistanceSummary, heuristic_path_reward

from saboter.actions import Action, PlayPath, SabotageTool
from saboter.cards import Role
from saboter.env import SaboteurEnv
from saboter.training.progress_metrics import DecisionProgress, GameProgress


def shaping_reward_for_transition(
    env: SaboteurEnv,
    *,
    reward_mode: str,
    role: str,
    action: Action,
    before_progress: DecisionProgress,
    after_progress: DecisionProgress,
    before_game_progress: GameProgress,
    after_game_progress: GameProgress,
    before_heuristic_goal_distances: GoalDistanceSummary | None = None,
    after_heuristic_goal_distances: GoalDistanceSummary | None = None,
) -> float:
    if reward_mode == "terminal":
        return 0.0
    if reward_mode == "progress":
        if role != Role.MINER.value:
            return 0.0
        delta_reachable = after_progress.reachable_tiles - before_progress.reachable_tiles
        delta_distance = before_progress.min_distance_to_goal - after_progress.min_distance_to_goal
        delta_stone = after_game_progress.public_stone_reaches - before_game_progress.public_stone_reaches
        shaping_reward = 0.0
        shaping_reward += delta_reachable * 0.01
        shaping_reward += max(0.0, delta_distance) * 0.01
        shaping_reward += delta_stone * 0.2
        return shaping_reward
    if reward_mode == "sabotage":
        return _sabotage_reward(env, role, action)
    if reward_mode == "heuristic":
        sabotage_reward = _sabotage_reward(env, role, action)
        if sabotage_reward != 0.0:
            return sabotage_reward
        stone_reveal_reward = _miner_stone_reveal_reward(
            role,
            before_game_progress,
            after_game_progress,
        )
        if not isinstance(action, PlayPath):
            return stone_reveal_reward
        if before_heuristic_goal_distances is None or after_heuristic_goal_distances is None:
            return stone_reveal_reward
        return stone_reveal_reward + heuristic_path_reward(
            before_heuristic_goal_distances,
            after_heuristic_goal_distances,
        )
    raise ValueError(f"Unknown reward_mode: {reward_mode}")


def _sabotage_reward(
    env: SaboteurEnv,
    role: str,
    action: Action,
) -> float:
    if not isinstance(action, SabotageTool):
        return 0.0
    target_role = env.players[action.target_player].role.value
    if role == Role.MINER.value:
        return 0.1 if target_role == Role.SABOTEUR.value else -0.1
    if role == Role.SABOTEUR.value:
        return 0.1 if target_role == Role.MINER.value else -0.1
    return 0.0


def _miner_stone_reveal_reward(
    role: str,
    before_game_progress: GameProgress,
    after_game_progress: GameProgress,
) -> float:
    if role != Role.MINER.value:
        return 0.0
    delta_stone = after_game_progress.public_stone_reaches - before_game_progress.public_stone_reaches
    return max(0.0, delta_stone) * 0.2
