"""Rollout collection for graph-action PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from saboter.actions import Discard, MapGoal, PlayPath, RepairTool, Rockfall, SabotageTool
from saboter.agents.graph_neural_agent import GraphNeuralAgent
from saboter.agents.random_agent import LegalRandomAgent
from saboter.env import SaboteurEnv
from saboter.training.curriculum import filter_actions_for_training_mode
from saboter.training.graph_tensorize import GraphTensors
from saboter.training.heuristic_frontier import HeuristicRewardTracker
from saboter.training.progress_metrics import (
    decision_progress_from_observation,
    game_progress_from_env,
)
from saboter.training.reward_shaping import shaping_reward_for_transition


@dataclass(frozen=True)
class GraphTransition:
    graph: GraphTensors
    action_index: int
    old_log_prob: float
    value: float
    entropy: float
    player_id: int
    role: str
    action_type: str
    map_available: bool = False
    reachable_tiles: float = 0.0
    frontier_empty_cells: float = 0.0
    min_distance_to_goal: float = 0.0
    private_goal_knowledge_count: float = 0.0
    reward: float = 0.0
    terminal_reward: float = 0.0
    shaping_reward: float = 0.0
    done: bool = False


@dataclass(frozen=True)
class GraphRolloutGame:
    transitions: list[GraphTransition]
    outcome: str
    rewards: dict[int, float]
    steps: int
    revealed_goals: int = 0
    public_stone_reaches: float = 0.0
    gold_reaches: float = 0.0


def collect_graph_game_rollout(
    env: SaboteurEnv,
    agent: GraphNeuralAgent,
    *,
    seed: int,
    max_steps: int = 500,
    reward_mode: str = "terminal",
    training_mode: str = "normal",
    miners_only_actions: str = "path_discard_map",
) -> GraphRolloutGame:
    if training_mode == "miners_only":
        from saboter.cards import Role
        env.reset(seed=seed, force_roles=[Role.MINER] * env.num_players)
    else:
        env.reset(seed=seed)

    random_agent = LegalRandomAgent(seed=seed + 999) if training_mode == "random_saboteurs" else None
    heuristic_tracker = _build_heuristic_tracker(env, reward_mode)

    pending: list[GraphTransition] = []
    steps = 0
    while not env.is_terminal():
        if steps >= max_steps:
            raise RuntimeError(f"Graph rollout seed {seed} exceeded max_steps={max_steps}")
        player_id = env.agent_selection
        legal_actions = env.legal_actions(player_id)
        legal_actions = filter_actions_for_training_mode(
            legal_actions,
            training_mode,
            miners_only_actions,
        )
        if not legal_actions:
            env.step_known_legal(None)
            if heuristic_tracker is not None:
                heuristic_tracker.apply_action(_board_or_raise(env), None)
            steps += 1
            continue

        role = env.players[player_id].role.value
        observation = env.observe(player_id)
        map_available = _has_map_card(observation)
        before_progress = decision_progress_from_observation(observation)
        before_game_progress = game_progress_from_env(env)
        before_heuristic_goal_distances = (
            heuristic_tracker.current_goal_distances()
            if heuristic_tracker is not None
            else None
        )

        if training_mode == "random_saboteurs" and role == "saboteur":
            action = random_agent.act(env, player_id)
            env.step_known_legal(action)
            if heuristic_tracker is not None:
                heuristic_tracker.apply_action(_board_or_raise(env), action)
            steps += 1
            continue

        action, info = agent.act_with_info(
            env,
            player_id,
            legal_actions=legal_actions,
            observation=observation,
        )
        env.step_known_legal(action)
        if heuristic_tracker is not None:
            heuristic_tracker.apply_action(_board_or_raise(env), action)

        after_progress = decision_progress_from_observation(env.observe(player_id))
        after_game_progress = game_progress_from_env(env)
        after_heuristic_goal_distances = (
            heuristic_tracker.current_goal_distances()
            if heuristic_tracker is not None
            else None
        )

        shaping_reward = shaping_reward_for_transition(
            env,
            reward_mode=reward_mode,
            role=role,
            action=action,
            before_progress=before_progress,
            after_progress=after_progress,
            before_game_progress=before_game_progress,
            after_game_progress=after_game_progress,
            before_heuristic_goal_distances=before_heuristic_goal_distances,
            after_heuristic_goal_distances=after_heuristic_goal_distances,
        )

        pending.append(
            GraphTransition(
                graph=info.graph_tensors.detach_cpu(),
                action_index=info.action_index,
                old_log_prob=info.log_prob,
                value=info.value,
                entropy=info.entropy,
                player_id=player_id,
                role=role,
                action_type=_action_type_name(action),
                map_available=map_available,
                reachable_tiles=before_progress.reachable_tiles,
                frontier_empty_cells=before_progress.frontier_empty_cells,
                min_distance_to_goal=before_progress.min_distance_to_goal,
                private_goal_knowledge_count=before_progress.private_goal_knowledge_count,
                shaping_reward=shaping_reward,
            )
        )
        steps += 1

    rewards = env.rewards()
    transitions: list[GraphTransition] = []
    for index, transition in enumerate(pending):
        terminal_reward = rewards[transition.player_id]
        total_reward = terminal_reward + transition.shaping_reward
        transitions.append(
            GraphTransition(
                graph=transition.graph,
                action_index=transition.action_index,
                old_log_prob=transition.old_log_prob,
                value=transition.value,
                entropy=transition.entropy,
                player_id=transition.player_id,
                role=transition.role,
                action_type=transition.action_type,
                map_available=transition.map_available,
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
    return GraphRolloutGame(
        transitions=transitions,
        outcome=env.outcome.value if env.outcome is not None else "unknown",
        rewards=rewards,
        steps=steps,
        revealed_goals=sum(1 for event in env.history if event.action_type == "reveal_goal"),
        public_stone_reaches=game_progress.public_stone_reaches,
        gold_reaches=game_progress.gold_reaches,
    )


def collect_graph_rollouts(
    env: SaboteurEnv,
    agent: GraphNeuralAgent,
    *,
    games: int,
    seed: int,
    max_steps: int = 500,
    reward_mode: str = "terminal",
    training_mode: str = "normal",
    miners_only_actions: str = "path_discard_map",
) -> list[GraphRolloutGame]:
    if games <= 0:
        raise ValueError("games must be positive")
    return [
        collect_graph_game_rollout(
            env,
            agent,
            seed=seed + game_index,
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


def _has_map_card(observation: dict[str, object]) -> bool:
    hand = observation.get("hand", [])
    if not isinstance(hand, list):
        return False
    return any(isinstance(card, dict) and card.get("type") == "map" for card in hand)


def _build_heuristic_tracker(
    env: SaboteurEnv,
    reward_mode: str,
) -> HeuristicRewardTracker | None:
    if reward_mode != "heuristic":
        return None
    return HeuristicRewardTracker.from_board(_board_or_raise(env))


def _board_or_raise(env: SaboteurEnv):
    if env.board is None:
        raise RuntimeError("Environment board is unavailable during graph rollout collection")
    return env.board
