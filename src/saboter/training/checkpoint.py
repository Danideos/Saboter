"""Checkpoint helpers for neural training scripts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    config: dict[str, Any],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": _plain_config(config),
        "model_metadata": _model_metadata(model),
    }
    if hasattr(model, "obs_sizes"):
        payload["obs_sizes"] = asdict(model.obs_sizes)
    if hasattr(model, "action_size"):
        payload["action_size"] = model.action_size
    torch.save(payload, target)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    expected_action_size = payload.get("action_size")
    if expected_action_size is not None and hasattr(model, "action_size") and expected_action_size != model.action_size:
        raise ValueError(
            f"Checkpoint action_size {expected_action_size} does not match model action_size {model.action_size}"
        )
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def _model_metadata(model: torch.nn.Module) -> dict[str, Any]:
    if hasattr(model, "checkpoint_metadata"):
        metadata = model.checkpoint_metadata()
        return dict(metadata)
    result: dict[str, Any] = {"model_type": getattr(model, "model_type", "flat")}
    if hasattr(model, "obs_sizes"):
        result["obs_sizes"] = asdict(model.obs_sizes)
    if hasattr(model, "action_size"):
        result["action_size"] = model.action_size
    return result


def _plain_config(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in config.items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result
