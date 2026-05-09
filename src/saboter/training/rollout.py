"""Rollout collection for normal PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from saboter.actions import Discard, MapGoal, PlayPath, RepairTool, Rockfall, SabotageTool
from saboter.agents.neural_agent import NeuralAgent
from saboter.agents.random_agent import LegalRandomAgent
from saboter.env import SaboteurEnv
from saboter.training.curriculum import filter_actions_for_training_mode
from saboter.training.progress_metrics import (
    decision_progress_from_observation,
    game_progress_from_env,
)
from saboter.training.tensorize import tensorize_actions, tensorize_observation


@dataclass(frozen=True)
class Transition:
    board: torch.Tensor
    nonboard: torch.Tensor
    actions: torch.Tensor
    action_index: int
    old_log_prob: float
    value: float
    entropy: float
    player_id: int
    role: str
    action_type: str
    reachable_tiles: float = 0.0
    frontier_empty_cells: float = 0.0
    min_distance_to_goal: float = 0.0
    private_goal_knowledge_count: float = 0.0
    reward: float = 0.0
    terminal_reward: float = 0.0
    shaping_reward: float = 0.0
    done: bool = False


@dataclass(frozen=True)
class RolloutGame:
    transitions: list[Transition]
    outcome: str
    rewards: dict[int, float]
    steps: int
    revealed_goals: int = 0
    public_stone_reaches: float = 0.0
    gold_reaches: float = 0.0


def collect_game_rollout(
    env: SaboteurEnv,
    agent: NeuralAgent,
    *,
    seed: int,
    storage_device: str | torch.device = "cpu",
    max_steps: int = 500,
    reward_mode: str = "terminal",
    training_mode: str = "normal",
    miners_only_actions: str = "path_discard_map",
) -> RolloutGame:
    if training_mode == "miners_only":
        from saboter.cards import Role
        env.reset(seed=seed, force_roles=[Role.MINER] * env.num_players)
    else:
        env.reset(seed=seed)

    random_agent = LegalRandomAgent(seed=seed + 999) if training_mode == "random_saboteurs" else None

    pending: list[Transition] = []
    steps = 0
    while not env.is_terminal():
        if steps >= max_steps:
            raise RuntimeError(f"Rollout seed {seed} exceeded max_steps={max_steps}")
        player_id = env.agent_selection
        legal_actions = env.legal_actions(player_id)
        legal_actions = filter_actions_for_training_mode(
            legal_actions,
            training_mode,
            miners_only_actions,
        )
        if not legal_actions:
            env.step_known_legal(None)
            steps += 1
            continue

        role = env.players[player_id].role.value
        observation = env.observe(player_id)
        before_progress = decision_progress_from_observation(observation)
        before_game_progress = game_progress_from_env(env)

        if training_mode == "random_saboteurs" and role == "saboteur":
            action = random_agent.act(env, player_id)
            env.step_known_legal(action)
            steps += 1
            continue

        action, info = agent.act_with_info(
            env,
            player_id,
            legal_actions=legal_actions,
            observation=observation,
        )
        board, nonboard = tensorize_observation(info.obs_features, storage_device)
        actions = tensorize_actions(info.action_features, storage_device)
        env.step_known_legal(action)

        after_progress = decision_progress_from_observation(env.observe(player_id))
        after_game_progress = game_progress_from_env(env)

        shaping_reward = 0.0
        if reward_mode == "progress" and role == "miner":
            delta_reachable = after_progress.reachable_tiles - before_progress.reachable_tiles
            delta_distance = before_progress.min_distance_to_goal - after_progress.min_distance_to_goal
            delta_stone = after_game_progress.public_stone_reaches - before_game_progress.public_stone_reaches

            shaping_reward += delta_reachable * 0.01
            shaping_reward += max(0.0, delta_distance) * 0.01
            shaping_reward += delta_stone * 0.2

        pending.append(
            Transition(
                board=board.detach().cpu(),
                nonboard=nonboard.detach().cpu(),
                actions=actions.detach().cpu(),
                action_index=info.action_index,
                old_log_prob=info.log_prob,
                value=info.value,
                entropy=info.entropy,
                player_id=player_id,
                role=role,
                action_type=_action_type_name(action),
                reachable_tiles=before_progress.reachable_tiles,
                frontier_empty_cells=before_progress.frontier_empty_cells,
                min_distance_to_goal=before_progress.min_distance_to_goal,
                private_goal_knowledge_count=before_progress.private_goal_knowledge_count,
                shaping_reward=shaping_reward,
            )
        )
        steps += 1

    rewards = env.rewards()
    transitions: list[Transition] = []
    for index, transition in enumerate(pending):
        terminal_reward = rewards[transition.player_id] if index == len(pending) - 1 else 0.0
        total_reward = terminal_reward + transition.shaping_reward
        transitions.append(
            Transition(
                board=transition.board,
                nonboard=transition.nonboard,
                actions=transition.actions,
                action_index=transition.action_index,
                old_log_prob=transition.old_log_prob,
                value=transition.value,
                entropy=transition.entropy,
                player_id=transition.player_id,
                role=transition.role,
                action_type=transition.action_type,
                reachable_tiles=transition.reachable_tiles,
                frontier_empty_cells=transition.frontier_empty_cells,
                min_distance_to_goal=transition.min_distance_to_goal,
                private_goal_knowledge_count=transition.private_goal_knowledge_count,
                reward=total_reward,
                terminal_reward=terminal_reward,
                shaping_reward=transition.shaping_reward,
                done=index == len(pending) - 1,
            )
        )
    game_progress = game_progress_from_env(env)
    return RolloutGame(
        transitions=transitions,
        outcome=env.outcome.value if env.outcome is not None else "unknown",
        rewards=rewards,
        steps=steps,
        revealed_goals=sum(1 for event in env.history if event.action_type == "reveal_goal"),
        public_stone_reaches=game_progress.public_stone_reaches,
        gold_reaches=game_progress.gold_reaches,
    )


def collect_rollouts(
    env: SaboteurEnv,
    agent: NeuralAgent,
    *,
    games: int,
    seed: int,
    storage_device: str | torch.device = "cpu",
    max_steps: int = 500,
    reward_mode: str = "terminal",
    training_mode: str = "normal",
    miners_only_actions: str = "path_discard_map",
) -> list[RolloutGame]:
    if games <= 0:
        raise ValueError("games must be positive")
    return [
        collect_game_rollout(
            env,
            agent,
            seed=seed + game_index,
            storage_device=storage_device,
            max_steps=max_steps,
            reward_mode=reward_mode,
            training_mode=training_mode,
            miners_only_actions=miners_only_actions,
        )
        for game_index in range(games)
    ]


def _action_type_name(action: object) -> str:
    if isinstance(action, PlayPath):
        return "play_path"
    if isinstance(action, SabotageTool):
        return "sabotage"
    if isinstance(action, RepairTool):
        return "repair"
    if isinstance(action, MapGoal):
        return "map_goal"
    if isinstance(action, Rockfall):
        return "rockfall"
    if isinstance(action, Discard):
        return "discard"
    return type(action).__name__
