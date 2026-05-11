"""Checkpoint helpers for neural training scripts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from saboter.graph_encoding import HISTORY_EVENT_FEATURE_NAMES


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
    allow_partial_load: bool = False,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    expected_action_size = payload.get("action_size")
    if expected_action_size is not None and hasattr(model, "action_size") and expected_action_size != model.action_size:
        raise ValueError(
            f"Checkpoint action_size {expected_action_size} does not match model action_size {model.action_size}"
        )
    _validate_model_metadata(payload.get("model_metadata", {}), model, allow_partial_load)
    if allow_partial_load:
        missing_keys, unexpected_keys = model.load_state_dict(payload["model_state_dict"], strict=False)
        payload["partial_load_missing_keys"] = list(missing_keys)
        payload["partial_load_unexpected_keys"] = list(unexpected_keys)
    else:
        model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload and not allow_partial_load:
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


_GRAPH_METADATA_DEFAULTS: dict[str, Any] = {
    "history_event_feature_size": len(HISTORY_EVENT_FEATURE_NAMES),
    "history_encoder": "none",
    "history_max_events": 100,
    "history_layers": 2,
    "history_heads": 4,
    "belief_injection": "none",
    "belief_post_layers": 1,
    "belief_detach": False,
    "role_conditioned_heads": False,
    "goal_loss_type": "ce",
}

_ARCHITECTURE_KEYS = (
    "model_type",
    "node_feature_size",
    "num_node_types",
    "num_edge_types",
    "history_event_feature_size",
    "hidden_dim",
    "graph_layers",
    "history_encoder",
    "history_max_events",
    "history_layers",
    "history_heads",
    "belief_injection",
    "belief_post_layers",
    "belief_detach",
    "role_conditioned_heads",
)


def _validate_model_metadata(
    checkpoint_metadata: object,
    model: torch.nn.Module,
    allow_partial_load: bool,
) -> None:
    if allow_partial_load or not hasattr(model, "checkpoint_metadata"):
        return
    current = _normalized_metadata(_model_metadata(model))
    checkpoint = _normalized_metadata(dict(checkpoint_metadata or {}))
    mismatches = []
    for key in _ARCHITECTURE_KEYS:
        if key not in current:
            continue
        if checkpoint.get(key) != current.get(key):
            mismatches.append(f"{key}: checkpoint={checkpoint.get(key)!r}, model={current.get(key)!r}")
    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(
            "Checkpoint architecture metadata does not match the current model. "
            f"{joined}. Re-run with allow_partial_load=True to load matching weights non-strictly."
        )


def _normalized_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    if result.get("model_type") == "graph":
        for key, value in _GRAPH_METADATA_DEFAULTS.items():
            result.setdefault(key, value)
    for key in ("belief_detach", "role_conditioned_heads"):
        if key in result:
            result[key] = _bool_value(result[key])
    for key in (
        "node_feature_size",
        "num_node_types",
        "num_edge_types",
        "history_event_feature_size",
        "hidden_dim",
        "graph_layers",
        "history_max_events",
        "history_layers",
        "history_heads",
        "belief_post_layers",
    ):
        if key in result:
            result[key] = int(result[key])
    return result


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "y", "on"}
    return bool(value)
