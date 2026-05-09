#!/usr/bin/env python3
"""Profile the Saboteur neural training pipeline by stage."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from saboter.action_encoding import encode_action_features
from saboter.agents.neural_agent import NeuralAgent
from saboter.env import SaboteurEnv
from saboter.observation import encode_observation_features
from saboter.training.checkpoint import save_checkpoint
from saboter.training.ppo import PPOConfig, ppo_update
from saboter.training.rollout import collect_rollouts
from saboter.training.tensorize import tensorize_actions, tensorize_observation
from train_ppo import (
    _build_policy,
    _collect_iteration_rollouts,
    _flatten_transitions,
    evaluate_vs_legal_random,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Time decision, rollout, PPO update, eval, and checkpoint stages."
    )
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--decision-steps", type=int, default=200)
    parser.add_argument("--rollout-games", type=int, default=16)
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args(argv)
    _validate_args(args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    torch.manual_seed(args.seed)
    model = _build_policy(args.players, args.seed, device)
    agent = NeuralAgent(model, device=device, deterministic=False)

    print(
        "profile_start "
        f"device={device} players={args.players} seed={args.seed} "
        f"torch_threads={torch.get_num_threads()}",
        flush=True,
    )

    decision = profile_decision_loop(
        model=model,
        device=device,
        num_players=args.players,
        seed=args.seed,
        steps=args.decision_steps,
        max_steps=args.max_steps,
    )
    print(_format_metrics("decision_loop", decision), flush=True)

    rollout_single = profile_single_rollout(
        agent=agent,
        num_players=args.players,
        games=args.rollout_games,
        seed=args.seed + 100_000,
        max_steps=args.max_steps,
    )
    print(_format_metrics("rollout_single", rollout_single), flush=True)

    worker_results = []
    for workers in args.worker_counts:
        result = profile_worker_rollout(
            model=model,
            agent=agent,
            num_players=args.players,
            games=args.rollout_games,
            workers=workers,
            worker_torch_threads=args.worker_torch_threads,
            seed=args.seed + 200_000 + workers * 1000,
            max_steps=args.max_steps,
        )
        worker_results.append(result)
        print(_format_metrics("rollout_workers", result), flush=True)

    baseline_games = _collect_iteration_rollouts(
        model,
        SaboteurEnv(num_players=args.players),
        agent,
        num_players=args.players,
        total_games=args.rollout_games,
        num_workers=max(args.worker_counts),
        worker_torch_threads=args.worker_torch_threads,
        seed=args.seed + 300_000,
        max_steps=args.max_steps,
    )
    transitions = _flatten_transitions(baseline_games)
    action_rows = sum(int(transition.actions.shape[0]) for transition in transitions)
    print(
        "ppo_dataset "
        f"games={len(baseline_games)} transitions={len(transitions)} "
        f"action_rows={action_rows} avg_actions_per_transition={_safe_div(action_rows, len(transitions)):.2f}",
        flush=True,
    )

    ppo_results = []
    for batch_size in args.batch_sizes:
        result = profile_ppo_update(
            template_model=model,
            transitions=transitions,
            device=device,
            lr=args.lr,
            epochs=args.ppo_epochs,
            batch_size=batch_size,
        )
        ppo_results.append(result)
        print(_format_metrics("ppo_update", result), flush=True)

    if args.eval_games > 0:
        result = profile_eval(
            model=model,
            num_players=args.players,
            games=args.eval_games,
            seed=args.seed + 400_000,
            device=device,
            max_steps=args.max_steps,
        )
        print(_format_metrics("eval_vs_random", result), flush=True)

    checkpoint_result = profile_checkpoint(model=model, seed=args.seed)
    print(_format_metrics("checkpoint", checkpoint_result), flush=True)

    print_summary(
        decision=decision,
        rollout_single=rollout_single,
        worker_results=worker_results,
        ppo_results=ppo_results,
    )
    print("OK", flush=True)
    return 0


def profile_decision_loop(
    *,
    model: torch.nn.Module,
    device: torch.device,
    num_players: int,
    seed: int,
    steps: int,
    max_steps: int,
) -> dict[str, float]:
    env = SaboteurEnv(num_players=num_players)
    env.reset(seed=seed)
    model.eval()
    counters = {
        "legal_actions_seconds": 0.0,
        "observe_seconds": 0.0,
        "encode_observation_seconds": 0.0,
        "encode_actions_seconds": 0.0,
        "tensorize_observation_seconds": 0.0,
        "tensorize_actions_seconds": 0.0,
        "model_forward_seconds": 0.0,
        "sample_seconds": 0.0,
        "env_step_seconds": 0.0,
        "total_seconds": 0.0,
    }
    completed_decisions = 0
    resets = 0
    skipped_empty_turns = 0
    legal_action_count = 0
    total_start = time.perf_counter()
    while completed_decisions < steps:
        if env.is_terminal():
            resets += 1
            env.reset(seed=seed + resets)
        if completed_decisions + skipped_empty_turns >= max_steps * max(1, resets + 1):
            resets += 1
            env.reset(seed=seed + resets)

        player_id = env.agent_selection
        start = time.perf_counter()
        legal_actions = env.legal_actions(player_id)
        counters["legal_actions_seconds"] += time.perf_counter() - start

        if not legal_actions:
            start = time.perf_counter()
            env.step_known_legal(None)
            counters["env_step_seconds"] += time.perf_counter() - start
            skipped_empty_turns += 1
            continue

        legal_action_count += len(legal_actions)
        start = time.perf_counter()
        observation = env.observe(player_id)
        counters["observe_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        obs_features = encode_observation_features(observation, legal_actions)
        counters["encode_observation_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        action_features = encode_action_features(observation, legal_actions)
        counters["encode_actions_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        board, nonboard = tensorize_observation(obs_features, device)
        _sync(device)
        counters["tensorize_observation_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        actions = tensorize_actions(action_features, device)
        _sync(device)
        counters["tensorize_actions_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        with torch.no_grad():
            logits, _value = model.score_actions(board, nonboard, actions)
        _sync(device)
        counters["model_forward_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        action_index = _sample_action_index(logits)
        counters["sample_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        env.step_known_legal(action_features[action_index].action)
        counters["env_step_seconds"] += time.perf_counter() - start
        completed_decisions += 1

    counters["total_seconds"] = time.perf_counter() - total_start
    counters["decisions"] = float(completed_decisions)
    counters["resets"] = float(resets)
    counters["skipped_empty_turns"] = float(skipped_empty_turns)
    counters["avg_legal_actions"] = _safe_div(legal_action_count, completed_decisions)
    counters["decisions_per_second"] = _safe_div(completed_decisions, counters["total_seconds"])
    _add_component_percentages(counters)
    return counters


def profile_single_rollout(
    *,
    agent: NeuralAgent,
    num_players: int,
    games: int,
    seed: int,
    max_steps: int,
) -> dict[str, float]:
    env = SaboteurEnv(num_players=num_players)
    start = time.perf_counter()
    rollouts = collect_rollouts(
        env,
        agent,
        games=games,
        seed=seed,
        storage_device="cpu",
        max_steps=max_steps,
    )
    seconds = time.perf_counter() - start
    transitions = _flatten_transitions(rollouts)
    return _rollout_result(
        games=games,
        rollouts=rollouts,
        transitions=len(transitions),
        seconds=seconds,
        workers=1,
    )


def profile_worker_rollout(
    *,
    model: torch.nn.Module,
    agent: NeuralAgent,
    num_players: int,
    games: int,
    workers: int,
    worker_torch_threads: int,
    seed: int,
    max_steps: int,
) -> dict[str, float]:
    env = SaboteurEnv(num_players=num_players)
    start = time.perf_counter()
    rollouts = _collect_iteration_rollouts(
        model,
        env,
        agent,
        num_players=num_players,
        total_games=games,
        num_workers=workers,
        worker_torch_threads=worker_torch_threads,
        seed=seed,
        max_steps=max_steps,
    )
    seconds = time.perf_counter() - start
    transitions = _flatten_transitions(rollouts)
    return _rollout_result(
        games=games,
        rollouts=rollouts,
        transitions=len(transitions),
        seconds=seconds,
        workers=workers,
        worker_torch_threads=worker_torch_threads,
    )


def profile_ppo_update(
    *,
    template_model: torch.nn.Module,
    transitions: list[object],
    device: torch.device,
    lr: float,
    epochs: int,
    batch_size: int,
) -> dict[str, float]:
    model = type(template_model)(template_model.obs_sizes, template_model.action_size).to(device)
    model.load_state_dict(template_model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    config = PPOConfig(epochs=epochs, batch_size=batch_size)
    _sync(device)
    start = time.perf_counter()
    metrics = ppo_update(model, optimizer, transitions, config, device=device)
    _sync(device)
    seconds = time.perf_counter() - start
    action_rows = sum(int(transition.actions.shape[0]) for transition in transitions)
    return {
        "batch_size": float(batch_size),
        "epochs": float(epochs),
        "transitions": float(len(transitions)),
        "action_rows": float(action_rows),
        "updates": float(metrics.updates),
        "seconds": seconds,
        "transitions_per_second": _safe_div(len(transitions) * epochs, seconds),
        "action_rows_per_second": _safe_div(action_rows * epochs, seconds),
        "policy_loss": metrics.policy_loss,
        "value_loss": metrics.value_loss,
        "entropy": metrics.entropy,
        "approx_kl": metrics.approx_kl,
        "clip_fraction": metrics.clip_fraction,
    }


def profile_eval(
    *,
    model: torch.nn.Module,
    num_players: int,
    games: int,
    seed: int,
    device: torch.device,
    max_steps: int,
) -> dict[str, float]:
    start = time.perf_counter()
    metrics = evaluate_vs_legal_random(
        model,
        num_players=num_players,
        games=games,
        seed=seed,
        device=device,
        max_steps=max_steps,
    )
    seconds = time.perf_counter() - start
    return {
        "games": float(games),
        "seconds": seconds,
        "games_per_second": _safe_div(games, seconds),
        **metrics,
    }


def profile_checkpoint(*, model: torch.nn.Module, seed: int) -> dict[str, float]:
    temp_dir = Path(tempfile.mkdtemp(prefix="saboter_profile_"))
    try:
        checkpoint_path = temp_dir / "checkpoint.pt"
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        start = time.perf_counter()
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            iteration=0,
            config={"seed": seed},
        )
        seconds = time.perf_counter() - start
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        return {"seconds": seconds, "checkpoint_mb": size_mb}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def print_summary(
    *,
    decision: dict[str, float],
    rollout_single: dict[str, float],
    worker_results: list[dict[str, float]],
    ppo_results: list[dict[str, float]],
) -> None:
    best_worker = min(worker_results, key=lambda item: item["seconds"]) if worker_results else None
    best_ppo = min(ppo_results, key=lambda item: item["seconds"]) if ppo_results else None
    print("summary", flush=True)
    print(
        "  decision_loop "
        f"top_component={_largest_component(decision)} "
        f"decisions_per_second={decision['decisions_per_second']:.2f}",
        flush=True,
    )
    print(
        "  rollout_single "
        f"seconds={rollout_single['seconds']:.3f} "
        f"transitions_per_second={rollout_single['transitions_per_second']:.2f}",
        flush=True,
    )
    if best_worker is not None:
        speedup = _safe_div(rollout_single["seconds"], best_worker["seconds"])
        print(
            "  rollout_workers_best "
            f"workers={int(best_worker['workers'])} seconds={best_worker['seconds']:.3f} "
            f"speedup_vs_single={speedup:.2f}x",
            flush=True,
        )
    if best_ppo is not None:
        rollout_vs_ppo = _safe_div(rollout_single["seconds"], best_ppo["seconds"])
        print(
            "  ppo_best "
            f"batch_size={int(best_ppo['batch_size'])} seconds={best_ppo['seconds']:.3f} "
            f"rollout_single_seconds_per_ppo_second={rollout_vs_ppo:.2f}",
            flush=True,
        )


def _rollout_result(
    *,
    games: int,
    rollouts: list[object],
    transitions: int,
    seconds: float,
    workers: int,
    worker_torch_threads: int | None = None,
) -> dict[str, float]:
    result = {
        "games": float(games),
        "workers": float(workers),
        "transitions": float(transitions),
        "seconds": seconds,
        "games_per_second": _safe_div(games, seconds),
        "transitions_per_second": _safe_div(transitions, seconds),
        "avg_game_length": _safe_div(sum(game.steps for game in rollouts), len(rollouts)),
    }
    if worker_torch_threads is not None:
        result["worker_torch_threads"] = float(worker_torch_threads)
    return result


def _format_metrics(section: str, metrics: dict[str, float]) -> str:
    return " ".join(
        [f"section={section}"]
        + [
            f"{key}={_format_value(value)}"
            for key, value in metrics.items()
        ]
    )


def _format_value(value: float) -> str:
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 1000:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sample_action_index(logits: torch.Tensor) -> int:
    probs = torch.softmax(logits, dim=0)
    return int(torch.multinomial(probs, 1).item())


def _add_component_percentages(metrics: dict[str, float]) -> None:
    total = metrics["total_seconds"]
    for key in list(metrics):
        if key.endswith("_seconds") and key != "total_seconds":
            metrics[f"{key.removesuffix('_seconds')}_pct"] = _safe_div(metrics[key] * 100.0, total)


def _largest_component(metrics: dict[str, float]) -> str:
    components = {
        key: value
        for key, value in metrics.items()
        if key.endswith("_seconds") and key != "total_seconds"
    }
    if not components:
        return "unknown"
    key, value = max(components.items(), key=lambda item: item[1])
    return f"{key.removesuffix('_seconds')}:{value:.3f}s"


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "players": args.players,
        "max-steps": args.max_steps,
        "decision-steps": args.decision_steps,
        "rollout-games": args.rollout_games,
        "worker-torch-threads": args.worker_torch_threads,
        "ppo-epochs": args.ppo_epochs,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.eval_games < 0:
        raise ValueError("--eval-games must be non-negative")
    if any(worker <= 0 for worker in args.worker_counts):
        raise ValueError("--worker-counts must contain only positive integers")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("--batch-sizes must contain only positive integers")


if __name__ == "__main__":
    raise SystemExit(main())
