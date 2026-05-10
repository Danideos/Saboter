#!/usr/bin/env python3
"""Train the first minimal PPO Saboteur policy."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from saboter.action_encoding import encode_actions
from saboter.agents.graph_neural_agent import GraphNeuralAgent
from saboter.agents.neural_agent import NeuralAgent
from saboter.agents.random_agent import LegalRandomAgent
from saboter.cards import Role
from saboter.env import Outcome, SaboteurEnv
from saboter.graph_encoding import encode_graph
from saboter.models.graph_policy import GraphPolicy
from saboter.models.policy import ObservationSizes, SaboteurPolicy
from saboter.observation import encode_observation
from saboter.training.checkpoint import load_checkpoint, save_checkpoint
from saboter.training.curriculum import filter_actions_for_training_mode
from saboter.training.graph_ppo import GraphPPOConfig, graph_ppo_update
from saboter.training.graph_rollout import (
    GraphRolloutGame,
    GraphTransition,
    collect_graph_rollouts,
)
from saboter.training.graph_tensorize import GraphTensors
from saboter.training.ppo import PPOConfig, ppo_update
from saboter.training.progress_metrics import (
    decision_progress_from_observation,
    game_progress_from_env,
)
from saboter.training.rollout import RolloutGame, Transition, collect_rollouts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train SaboteurPolicy with minimal PPO.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--games-per-iter", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--games-per-worker", type=int, default=None)
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument(
        "--worker-transport",
        choices=("torch", "plain"),
        default="torch",
        help="Use 'plain' on HPC/CPU servers to avoid PyTorch shared-memory tensor IPC.",
    )
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument("--model", choices=("flat", "graph"), default="flat")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-dir", type=Path, default=Path("runs/ppo_v0"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-games", type=int, default=16)
    parser.add_argument("--eval-neural-counts", type=int, nargs="+", default=[1])
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--graph-layers", type=int, default=3)
    parser.add_argument("--role-belief-coef", type=float, default=0.05)
    parser.add_argument("--goal-belief-coef", type=float, default=0.05)
    parser.add_argument("--reward-mode", choices=("terminal", "progress"), default="terminal")
    parser.add_argument("--training-mode", choices=("normal", "miners_only", "random_saboteurs"), default="normal")
    parser.add_argument(
        "--miners-only-actions",
        choices=("path_discard_map", "all"),
        default="path_discard_map",
        help="Legal-action curriculum used only with --training-mode miners_only.",
    )
    args = parser.parse_args(argv)

    if args.training_mode == "miners_only":
        args.role_belief_coef = 0.0

    _validate_args(args)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    env = SaboteurEnv(num_players=args.players)
    model = (
        _build_policy(args.players, args.seed, device)
        if args.model == "flat"
        else _build_graph_policy(args.players, args.seed, device, args.hidden_dim, args.graph_layers)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_iteration = 0
    if args.load_checkpoint is not None:
        payload = load_checkpoint(args.load_checkpoint, model=model, optimizer=optimizer, map_location=device)
        start_iteration = int(payload.get("iteration", 0))

    agent = (
        NeuralAgent(model, device=device, deterministic=False)
        if args.model == "flat"
        else GraphNeuralAgent(model, device=device, deterministic=False)
    )
    ppo_config = PPOConfig(
        epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        gamma=args.gamma,
    )
    graph_ppo_config = GraphPPOConfig(
        epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        gamma=args.gamma,
        role_belief_coef=args.role_belief_coef,
        goal_belief_coef=args.goal_belief_coef,
    )
    args.save_dir.mkdir(parents=True, exist_ok=True)
    games_per_iteration = _total_games_per_iteration(args)

    for iteration in range(start_iteration + 1, args.iterations + 1):
        model.eval()
        rollout_seed = args.seed + iteration * 100_000
        if args.model == "flat":
            games = _collect_iteration_rollouts(
                model,
                env,
                agent,
                num_players=args.players,
                total_games=games_per_iteration,
                num_workers=args.num_workers,
                worker_torch_threads=args.worker_torch_threads,
                worker_transport=args.worker_transport,
                seed=rollout_seed,
                max_steps=args.max_steps,
                reward_mode=args.reward_mode,
                training_mode=args.training_mode,
                miners_only_actions=args.miners_only_actions,
            )
            transitions = _flatten_transitions(games)
            ppo_metrics = ppo_update(model, optimizer, transitions, ppo_config, device=device)
        else:
            games = _collect_iteration_graph_rollouts(
                model,
                env,
                agent,
                num_players=args.players,
                total_games=games_per_iteration,
                num_workers=args.num_workers,
                worker_torch_threads=args.worker_torch_threads,
                worker_transport=args.worker_transport,
                seed=rollout_seed,
                max_steps=args.max_steps,
                reward_mode=args.reward_mode,
                training_mode=args.training_mode,
                miners_only_actions=args.miners_only_actions,
            )
            transitions = _flatten_graph_transitions(games)
            ppo_metrics = graph_ppo_update(model, optimizer, transitions, graph_ppo_config, device=device)
        rollout_metrics = _rollout_metrics(games, transitions)

        eval_metrics: dict[str, float] = {}
        if args.eval_games > 0 and iteration % args.eval_every == 0:
            model.eval()
            for neural_count in args.eval_neural_counts:
                eval_metrics.update(
                    evaluate_vs_legal_random(
                        model,
                        num_players=args.players,
                        games=args.eval_games,
                        seed=args.seed + 9_000_000 + iteration * 10_000 + neural_count * 1_000,
                        device=device,
                        max_steps=args.max_steps,
                        model_type=args.model,
                        neural_count=neural_count,
                    )
                )
            if args.training_mode == "miners_only":
                eval_metrics.update(
                    evaluate_miners_only(
                        model,
                        num_players=args.players,
                        games=args.eval_games,
                        seed=args.seed + 9_500_000 + iteration * 10_000,
                        device=device,
                        max_steps=args.max_steps,
                        model_type=args.model,
                        miners_only_actions=args.miners_only_actions,
                    )
                )

        checkpoint_path: Path | None = None
        if iteration % args.checkpoint_every == 0 or iteration == args.iterations:
            checkpoint_path = save_checkpoint(
                args.save_dir / f"checkpoint_{iteration:04d}.pt",
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                config=vars(args),
            )

        print(
            _format_metrics(
                iteration=iteration,
                games=games_per_iteration,
                transitions=len(transitions),
                rollout_metrics=rollout_metrics,
                ppo_metrics=ppo_metrics,
                eval_metrics=eval_metrics,
                checkpoint_path=checkpoint_path,
            ),
            flush=True,
        )
    return 0


def evaluate_vs_legal_random(
    model: torch.nn.Module,
    *,
    num_players: int,
    games: int,
    seed: int,
    device: str | torch.device,
    max_steps: int,
    model_type: str = "flat",
    neural_count: int = 1,
) -> dict[str, float]:
    if games <= 0:
        return {}
    if not 1 <= neural_count <= num_players:
        raise ValueError("neural_count must be in 1..num_players")
    neural_agent = (
        NeuralAgent(model, device=device, deterministic=True)
        if model_type == "flat"
        else GraphNeuralAgent(model, device=device, deterministic=True)
    )
    miner_wins = 0
    neural_rewards: list[float] = []
    lengths: list[int] = []
    for game_index in range(games):
        env = SaboteurEnv(num_players=num_players)
        env.reset(seed=seed + game_index)
        first_neural_seat = game_index % num_players
        neural_seats = {
            (first_neural_seat + offset) % num_players
            for offset in range(neural_count)
        }
        random_agents = [
            LegalRandomAgent(seed=seed * 1000 + game_index * 100 + player_id)
            for player_id in range(num_players)
        ]
        steps = 0
        while not env.is_terminal():
            if steps >= max_steps:
                raise RuntimeError(f"Eval seed {seed + game_index} exceeded max_steps={max_steps}")
            player_id = env.agent_selection
            legal_actions = env.legal_actions(player_id)
            if not legal_actions:
                action = None
            elif player_id in neural_seats:
                action = neural_agent.act(env, player_id)
            else:
                action = random_agents[player_id].act(env, player_id)
            env.step(action)
            steps += 1
        lengths.append(steps)
        if env.outcome == Outcome.MINERS_WIN:
            miner_wins += 1
        rewards = env.rewards()
        neural_rewards.extend(rewards[player_id] for player_id in sorted(neural_seats))
    prefix = "eval_vs_random" if neural_count == 1 else f"eval_{neural_count}_ours_vs_random"
    return {
        f"{prefix}_miners_win_rate": miner_wins / games,
        f"{prefix}_neural_avg_reward": sum(neural_rewards) / len(neural_rewards),
        f"{prefix}_avg_game_length": sum(lengths) / len(lengths),
    }


def evaluate_miners_only(
    model: torch.nn.Module,
    *,
    num_players: int,
    games: int,
    seed: int,
    device: str | torch.device,
    max_steps: int,
    model_type: str = "flat",
    miners_only_actions: str = "path_discard_map",
) -> dict[str, float]:
    if games <= 0:
        return {}

    agent = (
        NeuralAgent(model, device=device, deterministic=True)
        if model_type == "flat"
        else GraphNeuralAgent(model, device=device, deterministic=True)
    )
    miner_wins = 0
    lengths: list[int] = []
    reachable_tiles: list[float] = []
    frontier_empty_cells: list[float] = []
    min_distances: list[float] = []
    public_stone_reaches: list[float] = []
    gold_reaches: list[float] = []

    for game_index in range(games):
        env = SaboteurEnv(num_players=num_players)
        env.reset(seed=seed + game_index, force_roles=[Role.MINER] * num_players)
        steps = 0
        while not env.is_terminal():
            if steps >= max_steps:
                raise RuntimeError(
                    f"Miners-only eval seed {seed + game_index} exceeded max_steps={max_steps}"
                )
            player_id = env.agent_selection
            observation = env.observe(player_id)
            progress = decision_progress_from_observation(observation)
            reachable_tiles.append(progress.reachable_tiles)
            frontier_empty_cells.append(progress.frontier_empty_cells)
            min_distances.append(progress.min_distance_to_goal)

            legal_actions = filter_actions_for_training_mode(
                env.legal_actions(player_id),
                "miners_only",
                miners_only_actions,
            )
            if legal_actions:
                action, _info = agent.act_with_info(
                    env,
                    player_id,
                    legal_actions=legal_actions,
                    observation=observation,
                )
            else:
                action = None
            env.step(action)
            steps += 1

        lengths.append(steps)
        if env.outcome == Outcome.MINERS_WIN:
            miner_wins += 1
        game_progress = game_progress_from_env(env)
        public_stone_reaches.append(game_progress.public_stone_reaches)
        gold_reaches.append(game_progress.gold_reaches)

    return {
        "eval_miners_only_win_rate": miner_wins / games,
        "eval_miners_only_gold_reaches": _mean(gold_reaches),
        "eval_miners_only_public_stone_reaches": _mean(public_stone_reaches),
        "eval_miners_only_avg_reachable_tiles": _mean(reachable_tiles),
        "eval_miners_only_avg_frontier_empty_cells": _mean(frontier_empty_cells),
        "eval_miners_only_avg_min_distance_to_goal": _mean(min_distances),
        "eval_miners_only_avg_game_length": _mean(lengths),
    }


def _collect_iteration_rollouts(
    model: SaboteurPolicy,
    env: SaboteurEnv,
    agent: NeuralAgent,
    *,
    num_players: int,
    total_games: int,
    num_workers: int,
    worker_torch_threads: int,
    worker_transport: str,
    seed: int,
    max_steps: int,
    reward_mode: str,
    training_mode: str,
    miners_only_actions: str,
) -> list[RolloutGame]:
    if num_workers <= 1:
        return collect_rollouts(
            env,
            agent,
            games=total_games,
            seed=seed,
            storage_device="cpu",
            max_steps=max_steps,
            reward_mode=reward_mode,
            training_mode=training_mode,
            miners_only_actions=miners_only_actions,
        )

    worker_counts = _split_games(total_games, num_workers)
    cpu_state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    obs_sizes = asdict(model.obs_sizes)
    action_size = model.action_size
    games: list[RolloutGame] = []
    with ProcessPoolExecutor(max_workers=len(worker_counts)) as pool:
        futures = [
            pool.submit(
                _collect_worker,
                worker_id,
                game_count,
                cpu_state_dict,
                obs_sizes,
                action_size,
                num_players,
                seed + worker_id * 10_000,
                max_steps,
                worker_torch_threads,
                reward_mode,
                training_mode,
                miners_only_actions,
                worker_transport,
            )
            for worker_id, game_count in enumerate(worker_counts)
            if game_count > 0
        ]
        for future in futures:
            result = future.result()
            games.extend(
                _deserialize_rollout_games(result)
                if worker_transport == "plain"
                else result
            )
    return games


def _collect_iteration_graph_rollouts(
    model: GraphPolicy,
    env: SaboteurEnv,
    agent: GraphNeuralAgent,
    *,
    num_players: int,
    total_games: int,
    num_workers: int,
    worker_torch_threads: int,
    worker_transport: str,
    seed: int,
    max_steps: int,
    reward_mode: str,
    training_mode: str,
    miners_only_actions: str,
) -> list[GraphRolloutGame]:
    if num_workers <= 1:
        return collect_graph_rollouts(
            env,
            agent,
            games=total_games,
            seed=seed,
            max_steps=max_steps,
            reward_mode=reward_mode,
            training_mode=training_mode,
            miners_only_actions=miners_only_actions,
        )

    worker_counts = _split_games(total_games, num_workers)
    cpu_state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    metadata = model.checkpoint_metadata()
    games: list[GraphRolloutGame] = []
    with ProcessPoolExecutor(max_workers=len(worker_counts)) as pool:
        futures = [
            pool.submit(
                _collect_graph_worker,
                worker_id,
                game_count,
                cpu_state_dict,
                metadata,
                num_players,
                seed + worker_id * 10_000,
                max_steps,
                worker_torch_threads,
                reward_mode,
                training_mode,
                miners_only_actions,
                worker_transport,
            )
            for worker_id, game_count in enumerate(worker_counts)
            if game_count > 0
        ]
        for future in futures:
            result = future.result()
            games.extend(
                _deserialize_graph_rollout_games(result)
                if worker_transport == "plain"
                else result
            )
    return games


def _collect_graph_worker(
    worker_id: int,
    games: int,
    model_state_dict: dict[str, torch.Tensor],
    metadata: dict[str, object],
    num_players: int,
    seed: int,
    max_steps: int,
    torch_threads: int,
    reward_mode: str,
    training_mode: str,
    miners_only_actions: str,
    worker_transport: str,
) -> list[GraphRolloutGame] | list[dict[str, object]]:
    torch.set_num_threads(torch_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    torch.manual_seed(seed + worker_id)
    env = SaboteurEnv(num_players=num_players)
    model = GraphPolicy(
        node_feature_size=int(metadata["node_feature_size"]),
        num_node_types=int(metadata["num_node_types"]),
        num_edge_types=int(metadata["num_edge_types"]),
        hidden_dim=int(metadata["hidden_dim"]),
        graph_layers=int(metadata["graph_layers"]),
    )
    model.load_state_dict(model_state_dict)
    agent = GraphNeuralAgent(model, device="cpu", deterministic=False)
    rollouts = collect_graph_rollouts(
        env,
        agent,
        games=games,
        seed=seed,
        max_steps=max_steps,
        reward_mode=reward_mode,
        training_mode=training_mode,
        miners_only_actions=miners_only_actions,
    )
    if worker_transport == "plain":
        return _serialize_graph_rollout_games(rollouts)
    return rollouts


def _collect_worker(
    worker_id: int,
    games: int,
    model_state_dict: dict[str, torch.Tensor],
    obs_sizes: dict[str, int],
    action_size: int,
    num_players: int,
    seed: int,
    max_steps: int,
    torch_threads: int,
    reward_mode: str,
    training_mode: str,
    miners_only_actions: str,
    worker_transport: str,
) -> list[RolloutGame] | list[dict[str, object]]:
    torch.set_num_threads(torch_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    torch.manual_seed(seed + worker_id)
    env = SaboteurEnv(num_players=num_players)
    model = SaboteurPolicy(ObservationSizes(**obs_sizes), action_size)
    model.load_state_dict(model_state_dict)
    agent = NeuralAgent(model, device="cpu", deterministic=False)
    rollouts = collect_rollouts(
        env,
        agent,
        games=games,
        seed=seed,
        storage_device="cpu",
        max_steps=max_steps,
        reward_mode=reward_mode,
        training_mode=training_mode,
        miners_only_actions=miners_only_actions,
    )
    if worker_transport == "plain":
        return _serialize_rollout_games(rollouts)
    return rollouts


def _build_policy(num_players: int, seed: int, device: torch.device) -> SaboteurPolicy:
    env = SaboteurEnv(num_players=num_players)
    env.reset(seed=seed)
    player_id = env.agent_selection
    legal_actions = env.legal_actions(player_id)
    if not legal_actions:
        raise RuntimeError("Initial training state has no legal actions")
    obs_features = encode_observation(env, player_id, legal_actions)
    action_features = encode_actions(env, player_id, legal_actions)
    model = SaboteurPolicy.from_features(obs_features, len(action_features[0].vector))
    return model.to(device)


def _build_graph_policy(
    num_players: int,
    seed: int,
    device: torch.device,
    hidden_dim: int,
    graph_layers: int,
) -> GraphPolicy:
    env = SaboteurEnv(num_players=num_players)
    env.reset(seed=seed)
    player_id = env.agent_selection
    legal_actions = env.legal_actions(player_id)
    if not legal_actions:
        raise RuntimeError("Initial graph training state has no legal actions")
    graph = encode_graph(env, player_id, legal_actions)
    model = GraphPolicy.from_features(graph, hidden_dim=hidden_dim, graph_layers=graph_layers)
    return model.to(device)


def _flatten_transitions(games: list[RolloutGame]) -> list[Transition]:
    return [transition for game in games for transition in game.transitions]


def _flatten_graph_transitions(games: list[GraphRolloutGame]) -> list[GraphTransition]:
    return [transition for game in games for transition in game.transitions]


def _serialize_rollout_games(games: list[RolloutGame]) -> list[dict[str, object]]:
    return [
        {
            "transitions": [_serialize_transition(transition) for transition in game.transitions],
            "outcome": game.outcome,
            "rewards": game.rewards,
            "steps": game.steps,
            "revealed_goals": game.revealed_goals,
            "public_stone_reaches": game.public_stone_reaches,
            "gold_reaches": game.gold_reaches,
        }
        for game in games
    ]


def _deserialize_rollout_games(payload: list[dict[str, object]]) -> list[RolloutGame]:
    return [
        RolloutGame(
            transitions=[
                _deserialize_transition(item)
                for item in _as_list(game["transitions"])
            ],
            outcome=str(game["outcome"]),
            rewards={int(key): float(value) for key, value in dict(game["rewards"]).items()},
            steps=int(game["steps"]),
            revealed_goals=int(game["revealed_goals"]),
            public_stone_reaches=float(game["public_stone_reaches"]),
            gold_reaches=float(game["gold_reaches"]),
        )
        for game in payload
    ]


def _serialize_transition(transition: Transition) -> dict[str, object]:
    return {
        "board": transition.board.tolist(),
        "nonboard": transition.nonboard.tolist(),
        "actions": transition.actions.tolist(),
        "action_index": transition.action_index,
        "old_log_prob": transition.old_log_prob,
        "value": transition.value,
        "entropy": transition.entropy,
        "player_id": transition.player_id,
        "role": transition.role,
        "action_type": transition.action_type,
        "reachable_tiles": transition.reachable_tiles,
        "frontier_empty_cells": transition.frontier_empty_cells,
        "min_distance_to_goal": transition.min_distance_to_goal,
        "private_goal_knowledge_count": transition.private_goal_knowledge_count,
        "reward": transition.reward,
        "terminal_reward": transition.terminal_reward,
        "shaping_reward": transition.shaping_reward,
        "done": transition.done,
    }


def _deserialize_transition(data: object) -> Transition:
    item = dict(data)
    return Transition(
        board=torch.tensor(item["board"], dtype=torch.float32),
        nonboard=torch.tensor(item["nonboard"], dtype=torch.float32),
        actions=torch.tensor(item["actions"], dtype=torch.float32),
        action_index=int(item["action_index"]),
        old_log_prob=float(item["old_log_prob"]),
        value=float(item["value"]),
        entropy=float(item["entropy"]),
        player_id=int(item["player_id"]),
        role=str(item["role"]),
        action_type=str(item["action_type"]),
        reachable_tiles=float(item["reachable_tiles"]),
        frontier_empty_cells=float(item["frontier_empty_cells"]),
        min_distance_to_goal=float(item["min_distance_to_goal"]),
        private_goal_knowledge_count=float(item["private_goal_knowledge_count"]),
        reward=float(item["reward"]),
        terminal_reward=float(item["terminal_reward"]),
        shaping_reward=float(item["shaping_reward"]),
        done=bool(item["done"]),
    )


def _serialize_graph_rollout_games(games: list[GraphRolloutGame]) -> list[dict[str, object]]:
    return [
        {
            "transitions": [_serialize_graph_transition(transition) for transition in game.transitions],
            "outcome": game.outcome,
            "rewards": game.rewards,
            "steps": game.steps,
            "revealed_goals": game.revealed_goals,
            "public_stone_reaches": game.public_stone_reaches,
            "gold_reaches": game.gold_reaches,
        }
        for game in games
    ]


def _deserialize_graph_rollout_games(payload: list[dict[str, object]]) -> list[GraphRolloutGame]:
    return [
        GraphRolloutGame(
            transitions=[
                _deserialize_graph_transition(item)
                for item in _as_list(game["transitions"])
            ],
            outcome=str(game["outcome"]),
            rewards={int(key): float(value) for key, value in dict(game["rewards"]).items()},
            steps=int(game["steps"]),
            revealed_goals=int(game["revealed_goals"]),
            public_stone_reaches=float(game["public_stone_reaches"]),
            gold_reaches=float(game["gold_reaches"]),
        )
        for game in payload
    ]


def _serialize_graph_transition(transition: GraphTransition) -> dict[str, object]:
    return {
        "graph": _serialize_graph_tensors(transition.graph),
        "action_index": transition.action_index,
        "old_log_prob": transition.old_log_prob,
        "value": transition.value,
        "entropy": transition.entropy,
        "player_id": transition.player_id,
        "role": transition.role,
        "action_type": transition.action_type,
        "reachable_tiles": transition.reachable_tiles,
        "frontier_empty_cells": transition.frontier_empty_cells,
        "min_distance_to_goal": transition.min_distance_to_goal,
        "private_goal_knowledge_count": transition.private_goal_knowledge_count,
        "reward": transition.reward,
        "terminal_reward": transition.terminal_reward,
        "shaping_reward": transition.shaping_reward,
        "done": transition.done,
    }


def _deserialize_graph_transition(data: object) -> GraphTransition:
    item = dict(data)
    return GraphTransition(
        graph=_deserialize_graph_tensors(item["graph"]),
        action_index=int(item["action_index"]),
        old_log_prob=float(item["old_log_prob"]),
        value=float(item["value"]),
        entropy=float(item["entropy"]),
        player_id=int(item["player_id"]),
        role=str(item["role"]),
        action_type=str(item["action_type"]),
        reachable_tiles=float(item["reachable_tiles"]),
        frontier_empty_cells=float(item["frontier_empty_cells"]),
        min_distance_to_goal=float(item["min_distance_to_goal"]),
        private_goal_knowledge_count=float(item["private_goal_knowledge_count"]),
        reward=float(item["reward"]),
        terminal_reward=float(item["terminal_reward"]),
        shaping_reward=float(item["shaping_reward"]),
        done=bool(item["done"]),
    )


def _serialize_graph_tensors(graph: GraphTensors) -> dict[str, object]:
    return {
        "x": graph.x.tolist(),
        "node_type": graph.node_type.tolist(),
        "edge_index": graph.edge_index.tolist(),
        "edge_type": graph.edge_type.tolist(),
        "action_node_indices": graph.action_node_indices.tolist(),
        "global_node_index": graph.global_node_index.tolist(),
        "player_node_indices": graph.player_node_indices.tolist(),
        "goal_node_indices": graph.goal_node_indices.tolist(),
        "role_labels": None if graph.role_labels is None else graph.role_labels.tolist(),
        "goal_labels": None if graph.goal_labels is None else graph.goal_labels.tolist(),
    }


def _deserialize_graph_tensors(data: object) -> GraphTensors:
    item = dict(data)
    return GraphTensors(
        x=torch.tensor(item["x"], dtype=torch.float32),
        node_type=torch.tensor(item["node_type"], dtype=torch.long),
        edge_index=torch.tensor(item["edge_index"], dtype=torch.long),
        edge_type=torch.tensor(item["edge_type"], dtype=torch.long),
        action_node_indices=torch.tensor(item["action_node_indices"], dtype=torch.long),
        global_node_index=torch.tensor(item["global_node_index"], dtype=torch.long),
        player_node_indices=torch.tensor(item["player_node_indices"], dtype=torch.long),
        goal_node_indices=torch.tensor(item["goal_node_indices"], dtype=torch.long),
        role_labels=(
            None
            if item["role_labels"] is None
            else torch.tensor(item["role_labels"], dtype=torch.float32)
        ),
        goal_labels=(
            None
            if item["goal_labels"] is None
            else torch.tensor(item["goal_labels"], dtype=torch.float32)
        ),
    )


def _as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"Expected list payload, got {type(value).__name__}")
    return value


def _rollout_metrics(
    games: list[RolloutGame] | list[GraphRolloutGame],
    transitions: list[Transition] | list[GraphTransition],
) -> dict[str, float]:
    miner_wins = sum(1 for game in games if game.outcome == Outcome.MINERS_WIN.value)
    total_transitions = max(1, len(transitions))
    metrics = {
        "avg_reward": _mean(transition.reward for transition in transitions),
        "avg_terminal_reward": _mean(transition.terminal_reward for transition in transitions),
        "avg_shaping_reward": _mean(transition.shaping_reward for transition in transitions),
        "miners_win_rate": miner_wins / len(games),
        "avg_game_length": _mean(game.steps for game in games),
        "avg_rollout_entropy": _mean(transition.entropy for transition in transitions),
        "avg_rollout_value": _mean(transition.value for transition in transitions),
        "avg_legal_actions": _mean(_transition_action_count(transition) for transition in transitions),
        "avg_revealed_goals": _mean(game.revealed_goals for game in games),
        "avg_reachable_tiles": _mean(transition.reachable_tiles for transition in transitions),
        "avg_frontier_empty_cells": _mean(transition.frontier_empty_cells for transition in transitions),
        "avg_min_distance_to_goal": _mean(transition.min_distance_to_goal for transition in transitions),
        "avg_public_stone_reaches": _mean(game.public_stone_reaches for game in games),
        "avg_gold_reaches": _mean(game.gold_reaches for game in games),
        "avg_private_goal_knowledge_count": _mean(
            transition.private_goal_knowledge_count for transition in transitions
        ),
    }
    for action_type in ("play_path", "discard", "map_goal", "sabotage", "repair", "rockfall"):
        count = sum(1 for transition in transitions if transition.action_type == action_type)
        metrics[f"{action_type}_rate"] = count / total_transitions
    for role in ("miner", "saboteur"):
        role_transitions = [transition for transition in transitions if transition.role == role]
        role_count = max(1, len(role_transitions))
        metrics[f"play_path_rate_{role}"] = (
            sum(1 for transition in role_transitions if transition.action_type == "play_path")
            / role_count
        )
        metrics[f"discard_rate_{role}"] = (
            sum(1 for transition in role_transitions if transition.action_type == "discard")
            / role_count
        )
    return metrics


def _format_metrics(
    *,
    iteration: int,
    games: int,
    transitions: int,
    rollout_metrics: dict[str, float],
    ppo_metrics: object,
    eval_metrics: dict[str, float],
    checkpoint_path: Path | None,
) -> str:
    parts = [
        f"iter={iteration}",
        f"games={games}",
        f"transitions={transitions}",
        f"avg_reward={rollout_metrics['avg_reward']:.4f}",
        f"avg_terminal_reward={rollout_metrics['avg_terminal_reward']:.4f}",
        f"avg_shaping_reward={rollout_metrics['avg_shaping_reward']:.4f}",
        f"miners_win_rate={rollout_metrics['miners_win_rate']:.4f}",
        f"avg_game_length={rollout_metrics['avg_game_length']:.2f}",
        f"avg_rollout_entropy={rollout_metrics['avg_rollout_entropy']:.4f}",
        f"avg_rollout_value={rollout_metrics['avg_rollout_value']:.4f}",
        f"avg_legal_actions={rollout_metrics['avg_legal_actions']:.2f}",
        f"avg_revealed_goals={rollout_metrics['avg_revealed_goals']:.2f}",
        f"avg_reachable_tiles={rollout_metrics['avg_reachable_tiles']:.2f}",
        f"avg_frontier_empty_cells={rollout_metrics['avg_frontier_empty_cells']:.2f}",
        f"avg_min_distance_to_goal={rollout_metrics['avg_min_distance_to_goal']:.2f}",
        f"avg_public_stone_reaches={rollout_metrics['avg_public_stone_reaches']:.2f}",
        f"avg_gold_reaches={rollout_metrics['avg_gold_reaches']:.2f}",
        f"avg_private_goal_knowledge_count={rollout_metrics['avg_private_goal_knowledge_count']:.2f}",
        f"play_path_rate={rollout_metrics['play_path_rate']:.4f}",
        f"discard_rate={rollout_metrics['discard_rate']:.4f}",
        f"play_path_rate_miner={rollout_metrics['play_path_rate_miner']:.4f}",
        f"play_path_rate_saboteur={rollout_metrics['play_path_rate_saboteur']:.4f}",
        f"discard_rate_miner={rollout_metrics['discard_rate_miner']:.4f}",
        f"discard_rate_saboteur={rollout_metrics['discard_rate_saboteur']:.4f}",
        f"map_goal_rate={rollout_metrics['map_goal_rate']:.4f}",
        f"sabotage_rate={rollout_metrics['sabotage_rate']:.4f}",
        f"repair_rate={rollout_metrics['repair_rate']:.4f}",
        f"rockfall_rate={rollout_metrics['rockfall_rate']:.4f}",
        f"policy_loss={ppo_metrics.policy_loss:.6f}",
        f"value_loss={ppo_metrics.value_loss:.6f}",
        f"entropy={ppo_metrics.entropy:.6f}",
        f"approx_kl={ppo_metrics.approx_kl:.6f}",
        f"clip_fraction={ppo_metrics.clip_fraction:.4f}",
        f"grad_norm={ppo_metrics.grad_norm:.6f}",
    ]
    if hasattr(ppo_metrics, "role_belief_loss"):
        parts.append(f"role_belief_loss={ppo_metrics.role_belief_loss:.6f}")
    if hasattr(ppo_metrics, "goal_belief_loss"):
        parts.append(f"goal_belief_loss={ppo_metrics.goal_belief_loss:.6f}")
    for key, value in sorted(eval_metrics.items()):
        parts.append(f"{key}={value:.4f}")
    if checkpoint_path is not None:
        parts.append(f"checkpoint={checkpoint_path}")
    return " ".join(parts)


def _mean(values: object) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _transition_action_count(transition: Transition | GraphTransition) -> int:
    if hasattr(transition, "actions"):
        return int(transition.actions.shape[0])
    return int(transition.graph.action_node_indices.shape[0])


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "iterations": args.iterations,
        "games-per-iter": args.games_per_iter,
        "num-workers": args.num_workers,
        "worker-torch-threads": args.worker_torch_threads,
        "players": args.players,
        "max-steps": args.max_steps,
        "ppo-epochs": args.ppo_epochs,
        "batch-size": args.batch_size,
        "eval-every": args.eval_every,
        "checkpoint-every": args.checkpoint_every,
        "hidden-dim": args.hidden_dim,
        "graph-layers": args.graph_layers,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"--{name} must be positive")
    nonnegative_ints = {"eval-games": args.eval_games}
    for name, value in nonnegative_ints.items():
        if value < 0:
            raise ValueError(f"--{name} must be non-negative")
    for neural_count in args.eval_neural_counts:
        if not 1 <= neural_count <= args.players:
            raise ValueError("--eval-neural-counts values must be in 1..players")
    if args.gamma < 0.0 or args.gamma > 1.0:
        raise ValueError("--gamma must be in [0, 1]")
    if args.games_per_worker is not None and args.games_per_worker <= 0:
        raise ValueError("--games-per-worker must be positive when provided")


def _total_games_per_iteration(args: argparse.Namespace) -> int:
    if args.games_per_worker is not None:
        return args.games_per_worker * args.num_workers
    return args.games_per_iter


def _split_games(total_games: int, num_workers: int) -> list[int]:
    worker_count = min(total_games, num_workers)
    base = total_games // worker_count
    remainder = total_games % worker_count
    return [base + (1 if index < remainder else 0) for index in range(worker_count)]


if __name__ == "__main__":
    raise SystemExit(main())
