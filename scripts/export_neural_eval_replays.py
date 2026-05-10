#!/usr/bin/env python3
"""Export neural-policy eval games as self-contained HTML replays."""

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

from saboter.actions import (
    Action,
    Discard,
    MapGoal,
    PlayPath,
    RepairTool,
    Rockfall,
    SabotageTool,
)
from saboter.agents.graph_neural_agent import GraphNeuralAgent
from saboter.agents.neural_agent import NeuralAgent
from saboter.agents.random_agent import LegalRandomAgent
from saboter.cards import Role
from saboter.env import SaboteurEnv
from saboter.evaluation import GameResult
from saboter.models.graph_policy import GraphPolicy
from saboter.models.policy import ObservationSizes, SaboteurPolicy
from saboter.training.curriculum import filter_actions_for_training_mode
from saboter.visualization import save_html_replay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export neural Saboteur eval games to HTML.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--html-dir", type=Path, default=Path("replays/neural_eval"))
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("miners_only", "vs_random"), default="miners_only")
    parser.add_argument("--neural-count", type=int, default=1)
    parser.add_argument(
        "--miners-only-actions",
        choices=("path_discard_map", "all"),
        default="path_discard_map",
    )
    parser.add_argument("--sampling", action="store_true")
    args = parser.parse_args(argv)

    if args.games <= 0:
        raise ValueError("--games must be positive")
    if args.mode == "miners_only" and not 1 <= args.neural_count <= args.players:
        raise ValueError("--neural-count must be in 1..players")

    device = torch.device(args.device)
    model, model_type = _load_model(args.checkpoint, device)
    agent = _make_agent(model, model_type, device, deterministic=not args.sampling)

    args.html_dir.mkdir(parents=True, exist_ok=True)
    for game_index in range(args.games):
        seed = args.seed + game_index
        result = play_neural_eval_game(
            agent,
            model_type=model_type,
            mode=args.mode,
            num_players=args.players,
            seed=seed,
            max_steps=args.max_steps,
            neural_count=args.neural_count,
            miners_only_actions=args.miners_only_actions,
        )
        path = args.html_dir / f"{args.mode}_seed{seed}_game{game_index + 1}.html"
        save_html_replay(path, result)
        print(
            f"wrote={path} outcome={result.outcome} steps={result.steps} "
            f"actions={result.action_counts}",
            flush=True,
        )
    return 0


def play_neural_eval_game(
    agent: NeuralAgent | GraphNeuralAgent,
    *,
    model_type: str,
    mode: str,
    num_players: int,
    seed: int,
    max_steps: int,
    neural_count: int = 1,
    miners_only_actions: str = "path_discard_map",
) -> GameResult:
    env = SaboteurEnv(num_players=num_players)
    if mode == "miners_only":
        env.reset(seed=seed, force_roles=[Role.MINER] * num_players)
        neural_seats = set(range(num_players))
        roster = [f"{model_type}-neural-miner" for _player in range(num_players)]
    else:
        env.reset(seed=seed)
        neural_seats = {
            player_id
            for player_id, player in enumerate(env.players)
            if player.role == Role.MINER
        }
        roster = [
            f"{model_type}-neural-miner"
            if player_id in neural_seats
            else "legal-random-saboteur"
            for player_id in range(num_players)
        ]

    random_agents = [
        LegalRandomAgent(seed=seed * 1000 + player_id)
        for player_id in range(num_players)
    ]
    action_counts: Counter[str] = Counter()
    debug_steps: list[dict[str, object]] = [_debug_snapshot(env, actor=None)]
    steps = 0
    while not env.is_terminal():
        if steps >= max_steps:
            raise RuntimeError(f"Eval replay seed {seed} exceeded max_steps={max_steps}")

        player_id = env.agent_selection
        legal_actions = env.legal_actions(player_id)
        observation = env.observe(player_id)
        if mode == "miners_only":
            legal_actions = filter_actions_for_training_mode(
                legal_actions,
                "miners_only",
                miners_only_actions,
            )

        if not legal_actions:
            action = None
            info = None
        elif player_id in neural_seats:
            action, info = agent.act_with_info(
                env,
                player_id,
                legal_actions=legal_actions,
                observation=observation,
            )
        else:
            action = random_agents[player_id].act(env, player_id)
            info = None

        debug_step = _debug_step(
            env,
            actor=player_id,
            legal_actions=legal_actions,
            action=action,
            info=info,
            observation=observation,
            controller=roster[player_id],
        )
        action_counts[_action_kind(action)] += 1
        env.step(action)
        debug_step["discard_pile_after"] = _discard_pile(env)
        debug_steps.append(debug_step)
        steps += 1

    return GameResult(
        seed=seed,
        num_players=num_players,
        agent_names=roster,
        outcome=env.outcome.value if env.outcome is not None else "unknown",
        rewards=env.rewards(),
        roles={player_id: player.role.value for player_id, player in enumerate(env.players)},
        steps=steps,
        action_counts=dict(sorted(action_counts.items())),
        illegal_action_attempts=0,
        history=[event.to_dict() for event in env.history],
        final_board=env.board.public_tiles() if env.board is not None else [],
        deck_size=len(env.deck),
        remaining_hand_sizes={
            player_id: len(player.hand) for player_id, player in enumerate(env.players)
        },
        debug={"steps": debug_steps},
    )


def _load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    payload = torch.load(path, map_location=device)
    metadata = payload.get("model_metadata", {})
    model_type = str(metadata.get("model_type", payload.get("config", {}).get("model", "flat")))
    if model_type == "graph":
        model = GraphPolicy(
            node_feature_size=int(metadata["node_feature_size"]),
            num_node_types=int(metadata["num_node_types"]),
            num_edge_types=int(metadata["num_edge_types"]),
            hidden_dim=int(metadata["hidden_dim"]),
            graph_layers=int(metadata["graph_layers"]),
        )
    else:
        obs_sizes = payload.get("obs_sizes")
        action_size = payload.get("action_size")
        if obs_sizes is None or action_size is None:
            raise ValueError("Flat checkpoint is missing obs_sizes/action_size metadata")
        model = SaboteurPolicy(ObservationSizes(**obs_sizes), int(action_size))
        model_type = "flat"
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_type


def _make_agent(
    model: torch.nn.Module,
    model_type: str,
    device: torch.device,
    *,
    deterministic: bool,
) -> NeuralAgent | GraphNeuralAgent:
    if model_type == "graph":
        return GraphNeuralAgent(model, device=device, deterministic=deterministic)
    return NeuralAgent(model, device=device, deterministic=deterministic)


def _action_kind(action: Action | None) -> str:
    if action is None:
        return "skip"
    if isinstance(action, Discard):
        return "discard"
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
    return type(action).__name__


def _debug_snapshot(env: SaboteurEnv, actor: int | None) -> dict[str, object]:
    return {
        "actor": actor,
        "hands": _hands(env),
        "discard_pile": _discard_pile(env),
        "deck_size": len(env.deck),
    }


def _debug_step(
    env: SaboteurEnv,
    *,
    actor: int,
    legal_actions: list[Action],
    action: Action | None,
    info: object | None,
    observation: dict[str, object],
    controller: str,
) -> dict[str, object]:
    selected_card = _selected_card(observation, action)
    scores = getattr(info, "action_scores", None)
    action_index = getattr(info, "action_index", None)
    return {
        **_debug_snapshot(env, actor=actor),
        "controller": controller,
        "selected_action": _action_debug(action, score=None, selected=True),
        "selected_card": selected_card,
        "legal_actions": [
            _action_debug(
                candidate,
                score=(
                    float(scores[index])
                    if isinstance(scores, list) and index < len(scores)
                    else None
                ),
                selected=index == action_index,
            )
            for index, candidate in enumerate(legal_actions)
        ],
    }


def _hands(env: SaboteurEnv) -> dict[int, list[dict[str, object]]]:
    return {
        player_id: [card.public_dict() for card in player.hand]
        for player_id, player in enumerate(env.players)
    }


def _discard_pile(env: SaboteurEnv) -> list[dict[str, object] | None]:
    return list(env.discard_pile)


def _selected_card(observation: dict[str, object], action: Action | None) -> dict[str, object] | None:
    if action is None:
        return None
    hand = observation.get("hand", [])
    if not isinstance(hand, list) or not 0 <= action.card_slot < len(hand):
        return None
    card = hand[action.card_slot]
    return card if isinstance(card, dict) else None


def _action_debug(
    action: Action | None,
    *,
    score: float | None,
    selected: bool,
) -> dict[str, object]:
    data: dict[str, object] = {
        "label": _action_label(action),
        "type": _action_kind(action),
        "score": score,
        "selected": selected,
    }
    if action is None:
        return data
    data["card_slot"] = action.card_slot
    if isinstance(action, PlayPath):
        data.update({"x": action.x, "y": action.y, "rotation": action.rotation})
    elif isinstance(action, (SabotageTool, RepairTool)):
        data.update({"target_player": action.target_player, "tool": action.tool.value})
    elif isinstance(action, MapGoal):
        data["goal_index"] = action.goal_index
    elif isinstance(action, Rockfall):
        data.update({"x": action.x, "y": action.y})
    return data


def _action_label(action: Action | None) -> str:
    if action is None:
        return "skip"
    if isinstance(action, Discard):
        return f"discard slot {action.card_slot}"
    if isinstance(action, PlayPath):
        return f"play slot {action.card_slot} at ({action.x}, {action.y}) r{action.rotation}"
    if isinstance(action, SabotageTool):
        return f"sabotage P{action.target_player} {action.tool.value} slot {action.card_slot}"
    if isinstance(action, RepairTool):
        return f"repair P{action.target_player} {action.tool.value} slot {action.card_slot}"
    if isinstance(action, MapGoal):
        return f"map goal {action.goal_index} slot {action.card_slot}"
    if isinstance(action, Rockfall):
        return f"rockfall ({action.x}, {action.y}) slot {action.card_slot}"
    return repr(action)


if __name__ == "__main__":
    raise SystemExit(main())
