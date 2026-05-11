"""Graph-policy neural Saboteur agent."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from saboter.actions import Action
from saboter.env import SaboteurEnv
from saboter.graph_encoding import GraphFeatures, encode_graph
from saboter.training.graph_tensorize import GraphTensors, tensorize_graph


@dataclass(frozen=True)
class GraphActionInfo:
    graph_features: GraphFeatures
    graph_tensors: GraphTensors
    action_index: int
    log_prob: float
    value: float
    entropy: float
    action_scores: list[float]
    role_belief_logits: list[float]
    role_belief_probs: list[float]


class GraphNeuralAgent:
    """Select legal actions by scoring action nodes in a graph."""

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
    ) -> tuple[Action, GraphActionInfo]:
        resolved_player = env.agent_selection if player_id is None else player_id
        actions = env.legal_actions(resolved_player) if legal_actions is None else legal_actions
        if not actions:
            raise RuntimeError(f"No legal actions available for player {resolved_player}")
        resolved_observation = env.observe(resolved_player) if observation is None else observation
        graph_features = encode_graph(
            env,
            resolved_player,
            actions,
            resolved_observation,
            include_labels=True,
        )
        graph_tensors = tensorize_graph(
            graph_features,
            self.device,
            history_max_events=int(getattr(self.model, "history_max_events", 100)),
        )
        _check_finite("graph node features", graph_tensors.x)
        with torch.no_grad():
            output = self.model.score_graph(graph_tensors)
        logits = output.action_logits
        value = output.values
        role_logits = output.role_logits
        if tuple(logits.shape) != (len(actions),):
            raise RuntimeError(f"Expected graph logits shape {(len(actions),)}, got {tuple(logits.shape)}")
        if tuple(value.shape) != (1,):
            raise RuntimeError(f"Expected graph value shape {(1,)}, got {tuple(value.shape)}")
        if tuple(role_logits.shape) != (len(graph_features.player_node_indices),):
            raise RuntimeError(
                "Expected role belief logits shape "
                f"{(len(graph_features.player_node_indices),)}, got {tuple(role_logits.shape)}"
            )
        _check_finite("graph logits", logits)
        _check_finite("graph value", value)
        _check_finite("graph role logits", role_logits)

        log_probs = torch.log_softmax(logits, dim=0)
        probs = log_probs.exp()
        role_probs = torch.sigmoid(role_logits)
        if self.deterministic:
            action_index = int(torch.argmax(logits).item())
        else:
            action_index = int(torch.multinomial(probs, 1).item())
        info = GraphActionInfo(
            graph_features=graph_features,
            graph_tensors=graph_tensors.detach_cpu(),
            action_index=action_index,
            log_prob=float(log_probs[action_index].item()),
            value=float(value.squeeze(0).item()),
            entropy=float((-(probs * log_probs).sum()).item()),
            action_scores=[float(score) for score in logits.detach().cpu().tolist()],
            role_belief_logits=[float(score) for score in role_logits.detach().cpu().tolist()],
            role_belief_probs=[float(score) for score in role_probs.detach().cpu().tolist()],
        )
        return graph_features.actions[action_index], info


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains non-finite values")
