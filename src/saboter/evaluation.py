"""Tournament runner, metrics, replay export, and CLI for baseline agents."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, Sequence

from saboter.actions import (
    Action,
    Discard,
    MapGoal,
    PlayPath,
    RepairTool,
    Rockfall,
    SabotageTool,
)
from saboter.agents import (
    GreedyMinerAgent,
    GreedySaboteurAgent,
    HeuristicRoleInferenceAgent,
    LegalRandomAgent,
    RoleAwareHeuristicAgent,
)
from saboter.env import Outcome, SaboteurEnv
from saboter.visualization import render_game, save_html_replay


class Agent(Protocol):
    def act(self, env: SaboteurEnv, player_id: int | None = None) -> Action | None:
        ...


AGENT_NAMES = {
    "random",
    "legal-random",
    "greedy-miner",
    "miner",
    "greedy-saboteur",
    "saboteur",
    "role-aware",
    "heuristic",
    "role-inference",
    "belief-heuristic",
}


@dataclass(frozen=True)
class GameResult:
    seed: int
    num_players: int
    agent_names: list[str]
    outcome: str
    rewards: dict[int, float]
    roles: dict[int, str]
    steps: int
    action_counts: dict[str, int]
    illegal_action_attempts: int
    history: list[dict[str, object]]
    final_board: list[dict[str, object]]
    deck_size: int
    remaining_hand_sizes: dict[int, int]
    debug: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class TournamentResult:
    summary: dict[str, object]
    games: list[GameResult]

    def to_dict(self, include_games: bool = False) -> dict[str, object]:
        data: dict[str, object] = {"summary": _jsonable(self.summary)}
        if include_games:
            data["games"] = [game.to_dict() for game in self.games]
        return data


def make_agent(name: str, seed: int | None = None) -> Agent:
    normalized = name.lower().replace("_", "-")
    if normalized in {"random", "legal-random"}:
        return LegalRandomAgent(seed=seed)
    if normalized in {"greedy-miner", "miner"}:
        return GreedyMinerAgent(seed=seed)
    if normalized in {"greedy-saboteur", "saboteur"}:
        return GreedySaboteurAgent(seed=seed)
    if normalized in {"role-aware", "heuristic"}:
        return RoleAwareHeuristicAgent(seed=seed)
    if normalized in {"role-inference", "belief-heuristic"}:
        return HeuristicRoleInferenceAgent(seed=seed)
    valid = ", ".join(sorted(AGENT_NAMES))
    raise ValueError(f"Unknown agent '{name}'. Valid agents: {valid}")


def play_game(
    agent_names: Sequence[str],
    *,
    num_players: int = 5,
    seed: int = 0,
    max_steps: int = 500,
    strict: bool = True,
) -> GameResult:
    if not agent_names:
        raise ValueError("At least one agent name is required")

    roster_names = _expand_roster(agent_names, num_players)
    agents = [make_agent(name, seed=seed * 1000 + index) for index, name in enumerate(roster_names)]
    env = SaboteurEnv(num_players=num_players)
    env.reset(seed=seed)

    action_counts: Counter[str] = Counter()
    illegal_action_attempts = 0
    steps = 0
    while not env.is_terminal() and steps < max_steps:
        player_id = env.agent_selection
        legal_actions = env.legal_actions(player_id)
        action = agents[player_id].act(env, player_id)
        if action is None and legal_actions:
            illegal_action_attempts += 1
            if strict:
                raise ValueError(
                    f"Agent {roster_names[player_id]!r} returned no action "
                    f"despite {len(legal_actions)} legal actions for player {player_id}"
                )
            action = legal_actions[0]
        elif action is not None and action not in legal_actions:
            illegal_action_attempts += 1
            if strict:
                raise ValueError(
                    f"Agent {roster_names[player_id]!r} returned illegal action "
                    f"for player {player_id}: {action!r}"
                )
            action = legal_actions[0] if legal_actions else None

        action_counts[_action_kind(action)] += 1
        env.step(action)
        steps += 1

    if not env.is_terminal():
        raise RuntimeError(f"Game seed {seed} exceeded max_steps={max_steps}")

    return GameResult(
        seed=seed,
        num_players=num_players,
        agent_names=list(roster_names),
        outcome=env.outcome.value if env.outcome is not None else "unknown",
        rewards=env.rewards(),
        roles={player_id: player.role.value for player_id, player in enumerate(env.players)},
        steps=steps,
        action_counts=dict(sorted(action_counts.items())),
        illegal_action_attempts=illegal_action_attempts,
        history=[event.to_dict() for event in env.history],
        final_board=env.board.public_tiles() if env.board is not None else [],
        deck_size=len(env.deck),
        remaining_hand_sizes={
            player_id: len(player.hand) for player_id, player in enumerate(env.players)
        },
    )


def run_tournament(
    agent_names: Sequence[str],
    *,
    games: int = 100,
    num_players: int = 5,
    seed: int = 0,
    max_steps: int = 500,
    strict: bool = True,
) -> TournamentResult:
    if games <= 0:
        raise ValueError("games must be positive")

    results = [
        play_game(
            agent_names,
            num_players=num_players,
            seed=seed + game_index,
            max_steps=max_steps,
            strict=strict,
        )
        for game_index in range(games)
    ]
    return TournamentResult(summary=_summarize(results, agent_names, num_players, seed), games=results)


def save_replays(path: str | Path, result: TournamentResult) -> None:
    payload = result.to_dict(include_games=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Saboteur baseline tournaments.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["role-aware"],
        help="Agent names. One name fills all seats; multiple names are cycled by seat.",
    )
    parser.add_argument("--replay-out", type=Path, default=None)
    parser.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help="Write the first game as a self-contained HTML replay viewer.",
    )
    parser.add_argument(
        "--show-game",
        action="store_true",
        help="Print a readable replay for the first game after the JSON summary.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Limit printed replay events when --show-game is used.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Replace illegal agent actions with the first legal action and count them.",
    )
    args = parser.parse_args(argv)

    result = run_tournament(
        args.agents,
        games=args.games,
        num_players=args.players,
        seed=args.seed,
        max_steps=args.max_steps,
        strict=not args.non_strict,
    )
    if args.replay_out is not None:
        save_replays(args.replay_out, result)
    if args.html_out is not None and result.games:
        save_html_replay(args.html_out, result.games[0])
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    if args.show_game and result.games:
        print()
        print(render_game(result.games[0], max_events=args.max_events))
    return 0


def _expand_roster(agent_names: Sequence[str], num_players: int) -> list[str]:
    return [agent_names[index % len(agent_names)] for index in range(num_players)]


def _summarize(
    games: list[GameResult],
    agent_names: Sequence[str],
    num_players: int,
    seed: int,
) -> dict[str, object]:
    outcome_counts: Counter[str] = Counter(game.outcome for game in games)
    action_counts: Counter[str] = Counter()
    reward_by_agent: defaultdict[str, list[float]] = defaultdict(list)
    reward_by_role: defaultdict[str, list[float]] = defaultdict(list)
    illegal_action_attempts = 0
    for game in games:
        action_counts.update(game.action_counts)
        illegal_action_attempts += game.illegal_action_attempts
        for player_id, reward in game.rewards.items():
            reward_by_agent[game.agent_names[player_id]].append(reward)
            reward_by_role[game.roles[player_id]].append(reward)

    game_count = len(games)
    return {
        "agents": list(agent_names),
        "expanded_roster": _expand_roster(agent_names, num_players),
        "games": game_count,
        "num_players": num_players,
        "seed": seed,
        "miner_wins": outcome_counts[Outcome.MINERS_WIN.value],
        "saboteur_wins": outcome_counts[Outcome.SABOTEURS_WIN.value],
        "miner_win_rate": outcome_counts[Outcome.MINERS_WIN.value] / game_count,
        "saboteur_win_rate": outcome_counts[Outcome.SABOTEURS_WIN.value] / game_count,
        "average_game_length": sum(game.steps for game in games) / game_count,
        "action_counts": dict(sorted(action_counts.items())),
        "average_reward_by_agent": {
            agent_name: sum(values) / len(values)
            for agent_name, values in sorted(reward_by_agent.items())
        },
        "average_reward_by_role": {
            role: sum(values) / len(values) for role, values in sorted(reward_by_role.items())
        },
        "illegal_action_attempts": illegal_action_attempts,
    }


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


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
