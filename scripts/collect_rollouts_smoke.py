#!/usr/bin/env python3
"""Collect PPO-ready rollout transitions without updating the model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from saboter.action_encoding import encode_actions
from saboter.agents.neural_agent import NeuralAgent
from saboter.env import SaboteurEnv
from saboter.models.policy import SaboteurPolicy
from saboter.observation import encode_observation
from saboter.training.rollout import RolloutGame, collect_rollouts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect neural Saboteur rollouts.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args(argv)

    if args.games <= 0:
        raise ValueError("--games must be positive")
    torch.manual_seed(args.seed)

    env = SaboteurEnv(num_players=args.players)
    env.reset(seed=args.seed)
    player_id = env.agent_selection
    legal_actions = env.legal_actions(player_id)
    if not legal_actions:
        raise RuntimeError("Initial state has no legal actions")
    obs_features = encode_observation(env, player_id, legal_actions)
    action_features = encode_actions(env, player_id, legal_actions)
    model = SaboteurPolicy.from_features(obs_features, len(action_features[0].vector))
    agent = NeuralAgent(model, device=args.device, deterministic=args.deterministic)

    games = collect_rollouts(
        env,
        agent,
        games=args.games,
        seed=args.seed,
        storage_device="cpu",
        max_steps=args.max_steps,
    )
    transitions = [transition for game in games for transition in game.transitions]
    if not transitions:
        raise RuntimeError("No transitions were collected")

    print(f"games_collected={len(games)}")
    print(f"transitions_collected={len(transitions)}")
    print(f"avg_game_length={_mean(game.steps for game in games):.2f}")
    print(f"avg_reward={_mean(transition.reward for transition in transitions):.4f}")
    print(f"mean_value={_mean(transition.value for transition in transitions):.4f}")
    print(f"mean_log_prob={_mean(transition.old_log_prob for transition in transitions):.4f}")
    print(f"mean_entropy={_mean(transition.entropy for transition in transitions):.4f}")
    print(f"outcomes={_outcome_counts(games)}")
    print("OK")
    return 0


def _mean(values: object) -> float:
    items = list(values)
    return sum(items) / len(items)


def _outcome_counts(games: list[RolloutGame]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for game in games:
        counts[game.outcome] = counts.get(game.outcome, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
