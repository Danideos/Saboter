"""PyTorch-backed neural Saboteur agent."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from saboter.action_encoding import ActionFeatures, encode_action_features
from saboter.actions import Action
from saboter.env import SaboteurEnv
from saboter.observation import ObservationFeatures, encode_observation_features
from saboter.training.tensorize import tensorize_actions, tensorize_observation


@dataclass(frozen=True)
class NeuralActionInfo:
    obs_features: ObservationFeatures
    action_features: list[ActionFeatures]
    action_index: int
    log_prob: float
    value: float
    entropy: float


class NeuralAgent:
    """Selects legal actions by scoring the current legal action batch."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device = "cpu",
        deterministic: bool = False,
        eval_mode: bool = True,
    ):
        self.model = model
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.model.to(self.device)
        if eval_mode:
            self.model.eval()

    def act(self, env: SaboteurEnv, player_id: int | None = None) -> Action:
        action, _info = self.act_with_info(env, player_id)
        return action

    def act_with_info(
        self,
        env: SaboteurEnv,
        player_id: int | None = None,
        legal_actions: list[Action] | None = None,
        observation: dict[str, object] | None = None,
    ) -> tuple[Action, NeuralActionInfo]:
        resolved_player = env.agent_selection if player_id is None else player_id
        actions = env.legal_actions(resolved_player) if legal_actions is None else legal_actions
        if not actions:
            raise RuntimeError(f"No legal actions available for player {resolved_player}")
        resolved_observation = (
            env.observe(resolved_player)
            if observation is None
            else observation
        )

        obs_features = encode_observation_features(resolved_observation, actions)
        action_features = encode_action_features(resolved_observation, actions)
        board, nonboard = tensorize_observation(obs_features, self.device)
        actions = tensorize_actions(action_features, self.device)
        _check_finite("board", board)
        _check_finite("nonboard", nonboard)
        _check_finite("actions", actions)
        if actions.ndim != 2:
            raise RuntimeError(f"Expected action tensor [A, F], got {tuple(actions.shape)}")
        if actions.shape[0] != len(action_features):
            raise RuntimeError(
                "Action tensor/action feature count mismatch: "
                f"{actions.shape[0]} tensor rows vs {len(action_features)} features"
            )

        with torch.no_grad():
            logits, value = self.model.score_actions(board, nonboard, actions)
        expected_logits_shape = (len(action_features),)
        if tuple(logits.shape) != expected_logits_shape:
            raise RuntimeError(
                f"Expected logits shape {expected_logits_shape}, got {tuple(logits.shape)}"
            )
        if tuple(value.shape) != (1,):
            raise RuntimeError(f"Expected value shape {(1,)}, got {tuple(value.shape)}")
        _check_finite("logits", logits)
        _check_finite("value", value)

        log_probs = torch.log_softmax(logits, dim=0)
        probs = log_probs.exp()
        if self.deterministic:
            action_index = int(torch.argmax(logits).item())
        else:
            action_index = int(torch.multinomial(probs, 1).item())
        log_prob = float(log_probs[action_index].item())
        entropy = float((-(probs * log_probs).sum()).item())
        info = NeuralActionInfo(
            obs_features=obs_features,
            action_features=action_features,
            action_index=action_index,
            log_prob=log_prob,
            value=float(value.squeeze(0).item()),
            entropy=entropy,
        )
        return action_features[action_index].action, info


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains non-finite values")
