import pytest

torch = pytest.importorskip("torch")

from scripts.train_ppo import main as train_ppo_main
from saboter.agents.graph_neural_agent import GraphNeuralAgent
from saboter.env import SaboteurEnv
from saboter.graph_encoding import (
    EDGE_TYPE_NAMES,
    GRAPH_F,
    NODE_TYPE_IDS,
    NODE_TYPE_NAMES,
    encode_graph,
)
from saboter.models.graph_policy import GraphPolicy
from saboter.training.graph_ppo import GraphPPOConfig, graph_ppo_update
from saboter.training.graph_rollout import collect_graph_game_rollout
from saboter.training.graph_tensorize import collate_graph_tensors, tensorize_graph


def _graph_model_for_env(env: SaboteurEnv) -> GraphPolicy:
    graph = encode_graph(env, env.agent_selection, env.legal_actions())
    return GraphPolicy.from_features(graph, hidden_dim=16, graph_layers=1)


def test_graph_encoder_builds_valid_action_node_graph_without_goal_leakage():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=800)
    player = env.agent_selection
    legal_actions = env.legal_actions(player)

    graph = encode_graph(env, player, legal_actions)

    assert graph.actions == legal_actions
    assert len(graph.action_node_indices) == len(legal_actions)
    assert len(graph.player_node_indices) == env.num_players
    assert len(graph.goal_node_indices) == 3
    assert graph.global_node_index >= 0
    assert graph.role_labels is not None and len(graph.role_labels) == env.num_players
    assert graph.goal_labels is not None and len(graph.goal_labels) == 3
    assert all(0 <= node_type < len(NODE_TYPE_NAMES) for node_type in graph.node_type_ids)
    assert all(0 <= edge_type < len(EDGE_TYPE_NAMES) for edge_type in graph.edge_type_ids)
    node_count = len(graph.node_features)
    assert all(0 <= source < node_count and 0 <= target < node_count for source, target in graph.edge_index)

    cell_type = NODE_TYPE_IDS["cell"]
    hidden_goal_rows = [
        row
        for row, node_type in zip(graph.node_features, graph.node_type_ids)
        if node_type == cell_type and row[GRAPH_F["hidden_goal"]] == 1.0
    ]
    assert hidden_goal_rows
    for row in hidden_goal_rows:
        assert row[GRAPH_F["known_gold"]] == 0.0
        assert row[GRAPH_F["known_stone"]] == 0.0


def test_graph_policy_scores_single_and_batched_graphs():
    env_a = SaboteurEnv(num_players=3)
    env_a.reset(seed=801)
    graph_a = encode_graph(env_a, env_a.agent_selection, env_a.legal_actions())
    env_b = SaboteurEnv(num_players=3)
    env_b.reset(seed=802)
    graph_b = encode_graph(env_b, env_b.agent_selection, env_b.legal_actions())
    tensor_a = tensorize_graph(graph_a)
    tensor_b = tensorize_graph(graph_b)
    model = GraphPolicy.from_features(graph_a, hidden_dim=16, graph_layers=1)

    output_a = model.score_graph(tensor_a)
    batch = collate_graph_tensors([tensor_a, tensor_b], [0, 0], "cpu")
    output_batch = model.score_graph_batches(batch)

    assert tuple(output_a.action_logits.shape) == (len(graph_a.actions),)
    assert tuple(output_a.values.shape) == (1,)
    assert tuple(output_a.role_logits.shape) == (env_a.num_players,)
    assert tuple(output_a.goal_logits.shape) == (3,)
    assert tuple(output_batch.action_logits.shape) == (len(graph_a.actions) + len(graph_b.actions),)
    assert tuple(output_batch.values.shape) == (2,)
    assert torch.isfinite(output_batch.action_logits).all()
    assert torch.isfinite(output_batch.values).all()


def test_graph_agent_rollout_and_ppo_update_are_finite():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=803)
    torch.manual_seed(803)
    model = _graph_model_for_env(env)
    agent = GraphNeuralAgent(model, deterministic=False)

    action = agent.act(env, env.agent_selection)
    assert action in env.legal_actions(env.agent_selection)

    game = collect_graph_game_rollout(env, agent, seed=803, max_steps=200)
    assert game.transitions
    for transition in game.transitions:
        assert transition.graph.x.device.type == "cpu"
        assert transition.graph.role_labels is not None
        assert transition.graph.goal_labels is not None
        assert 0 <= transition.action_index < transition.graph.action_node_indices.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = graph_ppo_update(
        model,
        optimizer,
        game.transitions[:8],
        GraphPPOConfig(epochs=1, batch_size=4),
        device="cpu",
    )
    assert metrics.transitions == min(8, len(game.transitions))
    assert metrics.updates > 0
    assert metrics.role_belief_loss >= 0.0
    assert metrics.goal_belief_loss >= 0.0


def test_train_ppo_graph_script_runs_one_iteration_and_saves_checkpoint(tmp_path, capsys):
    exit_code = train_ppo_main(
        [
            "--model",
            "graph",
            "--iterations",
            "1",
            "--games-per-iter",
            "1",
            "--players",
            "3",
            "--device",
            "cpu",
            "--save-dir",
            str(tmp_path),
            "--seed",
            "804",
            "--ppo-epochs",
            "1",
            "--batch-size",
            "16",
            "--eval-games",
            "0",
            "--hidden-dim",
            "16",
            "--graph-layers",
            "1",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "iter=1" in output
    assert "role_belief_loss=" in output
    assert "goal_belief_loss=" in output
    assert "avg_reachable_tiles=" in output
    assert "checkpoint=" in output
    assert (tmp_path / "checkpoint_0001.pt").exists()
