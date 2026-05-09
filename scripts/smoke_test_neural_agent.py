#!/usr/bin/env python3
"""Run full games with an untrained neural action-scoring policy."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from saboter.action_encoding import encode_actions
from saboter.actions import Discard, MapGoal, PlayPath, RepairTool, Rockfall, SabotageTool
from saboter.agents.neural_agent import NeuralAgent
from saboter.env import SaboteurEnv
from saboter.models.policy import SaboteurPolicy
from saboter.observation import encode_observation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test an untrained neural Saboteur policy.")
    parser.add_argument("--games", type=int, default=1000)
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

    outcomes: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    lengths: list[int] = []
    legal_action_counts: list[int] = []
    for game_index in range(args.games):
        env.reset(seed=args.seed + game_index)
        steps = 0
        while not env.is_terminal():
            if steps >= args.max_steps:
                raise RuntimeError(
                    f"Game {game_index} with seed {args.seed + game_index} exceeded "
                    f"--max-steps={args.max_steps}"
                )
            legal_actions = env.legal_actions(env.agent_selection)
            legal_action_counts.append(len(legal_actions))
            action = agent.act(env, env.agent_selection) if legal_actions else None
            action_counts[_action_kind(action)] += 1
            env.step(action)
            steps += 1
        lengths.append(steps)
        outcomes[env.outcome.value if env.outcome is not None else "unknown"] += 1

    average_length = sum(lengths) / len(lengths)
    average_legal_actions = sum(legal_action_counts) / len(legal_action_counts)
    print(f"completed_games={args.games}")
    print(f"players={args.players}")
    print(f"average_length={average_length:.2f}")
    print(f"average_legal_actions={average_legal_actions:.2f}")
    print(f"action_counts={dict(sorted(action_counts.items()))}")
    print(f"outcomes={dict(sorted(outcomes.items()))}")
    print("OK")
    return 0


def _action_kind(action: object) -> str:
    if action is None:
        return "skip"
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


if __name__ == "__main__":
    raise SystemExit(main())
