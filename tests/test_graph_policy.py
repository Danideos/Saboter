import pytest

torch = pytest.importorskip("torch")

from scripts.export_neural_eval_replays import play_neural_eval_game
from scripts.train_ppo import (
    _deserialize_graph_rollout_games,
    _serialize_graph_rollout_games,
    main as train_ppo_main,
)
from saboter.actions import MapGoal
from saboter.agents.graph_neural_agent import GraphNeuralAgent
from saboter.board import Board
from saboter.cards import (
    CardType,
    GOAL_GOLD_CARD,
    GOAL_STONE_NE_CARD,
    GOAL_STONE_NW_CARD,
    GoalKind,
    Tool,
    action_card,
    path_card_by_id,
)
from saboter.env import PublicEvent, SaboteurEnv
from saboter.graph_encoding import (
    EDGE_TYPE_NAMES,
    GRAPH_F,
    HISTORY_EVENT_F,
    HISTORY_EVENT_FEATURE_NAMES,
    GRAPH_MAX_HISTORY,
    NODE_TYPE_IDS,
    NODE_TYPE_NAMES,
    encode_graph,
)
from saboter.models.history_transformer import HistoryTransformerEncoder
from saboter.models.graph_policy import GraphPolicy
from saboter.training.graph_ppo import GraphPPOConfig, graph_ppo_update
from saboter.training.graph_rollout import collect_graph_game_rollout
from saboter.training.checkpoint import load_checkpoint, save_checkpoint
from saboter.training.graph_tensorize import collate_graph_tensors, tensorize_graph


def _graph_model_for_env(env: SaboteurEnv) -> GraphPolicy:
    graph = encode_graph(env, env.agent_selection, env.legal_actions())
    return GraphPolicy.from_features(graph, hidden_dim=16, graph_layers=1)


def _env_with_public_history(seed: int = 821) -> SaboteurEnv:
    env = SaboteurEnv(num_players=3)
    env.reset(seed=seed)
    env.history = [
        PublicEvent(actor=0, action_type="discard"),
        PublicEvent(
            actor=1,
            action_type="sabotage",
            card=action_card(CardType.SABOTAGE, (Tool.CART,)).public_dict(),
            target_player=2,
            tool="cart",
        ),
        PublicEvent(
            actor=2,
            action_type="map_goal",
            card=action_card(CardType.MAP).public_dict(),
            goal_index=1,
        ),
    ]
    return env


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


def test_graph_encoder_accepts_reveal_history_goal_card():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=814)
    env.history = [
        PublicEvent(
            actor=1,
            action_type="reveal_goal",
            card=GOAL_STONE_NW_CARD.public_dict(),
            rotation=180,
            goal_index=0,
            revealed_goal_kind=GoalKind.STONE.value,
        )
    ]

    graph = encode_graph(env, env.agent_selection, env.legal_actions())

    history_type = NODE_TYPE_IDS["history"]
    history_rows = [
        row
        for row, node_type in zip(graph.node_features, graph.node_type_ids)
        if node_type == history_type
    ]
    assert history_rows
    assert history_rows[-1][GRAPH_F["revealed_stone"]] == 1.0


def test_graph_encoder_keeps_more_than_twenty_history_events():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=816)
    env.history = [
        PublicEvent(actor=1, action_type="discard")
        for _ in range(30)
    ]

    graph = encode_graph(env, env.agent_selection, env.legal_actions())

    history_type = NODE_TYPE_IDS["history"]
    history_rows = [
        row
        for row, node_type in zip(graph.node_features, graph.node_type_ids)
        if node_type == history_type
    ]
    assert GRAPH_MAX_HISTORY == 60
    assert len(history_rows) == 30


def test_structured_history_tensorization_is_public_and_padded():
    env = _env_with_public_history()
    graph = encode_graph(env, env.agent_selection, env.legal_actions())
    tensor = tensorize_graph(graph, history_max_events=8)

    assert tuple(tensor.history_features.shape) == (8, len(HISTORY_EVENT_FEATURE_NAMES))
    assert tuple(tensor.history_valid_mask.shape) == (8,)
    assert tensor.history_valid_mask.tolist()[:3] == [True, True, True]
    assert tensor.history_actor.tolist()[:3] == [0, 1, 2]
    assert tensor.history_target_player.tolist()[:3] == [-1, 2, -1]
    assert tensor.history_goal.tolist()[:3] == [-1, -1, 1]
    assert tensor.history_features[2, HISTORY_EVENT_F["goal_1"]] == 1.0
    assert tensor.history_features[:, HISTORY_EVENT_F["revealed_gold"]].sum() == 0.0
    assert tensor.history_features[:, HISTORY_EVENT_F["revealed_stone"]].sum() == 0.0
    assert tuple(tensor.player_node_indices.shape) == (10,)
    assert tensor.player_node_indices[: env.num_players].ge(0).all()
    assert tensor.player_node_indices[env.num_players :].eq(-1).all()


def test_history_transformer_forward_shapes():
    encoder = HistoryTransformerEncoder(
        event_feature_size=len(HISTORY_EVENT_FEATURE_NAMES),
        hidden_dim=16,
        max_events=5,
        max_players=10,
        num_goals=3,
        layers=1,
        heads=4,
    )
    features = torch.randn(2, 5, len(HISTORY_EVENT_FEATURE_NAMES))
    valid = torch.tensor(
        [
            [True, True, False, False, False],
            [False, False, False, False, False],
        ]
    )
    actor = torch.tensor([[0, 1, -1, -1, -1], [-1, -1, -1, -1, -1]])
    target = torch.full((2, 5), -1, dtype=torch.long)
    goal = torch.tensor([[-1, 2, -1, -1, -1], [-1, -1, -1, -1, -1]])

    global_history, player_history, goal_history = encoder(features, valid, actor, target, goal)

    assert tuple(global_history.shape) == (2, 16)
    assert tuple(player_history.shape) == (2, 10, 16)
    assert tuple(goal_history.shape) == (2, 3, 16)
    assert torch.isfinite(global_history).all()
    assert torch.isfinite(player_history).all()
    assert torch.isfinite(goal_history).all()
    assert torch.allclose(global_history[1], torch.zeros(16), atol=1e-6)


def test_graph_encoder_exposes_private_mapped_goal_shape_only_to_actor():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=815)
    env.board = Board([GOAL_STONE_NW_CARD, GOAL_GOLD_CARD, GOAL_STONE_NE_CARD])
    env.players[0].hand = [action_card(CardType.MAP)]
    env.deck = [path_card_by_id("path_ew")]

    env.step(MapGoal(0, 0))

    actor_graph = encode_graph(env, 0, env.legal_actions(0))
    other_graph = encode_graph(env, 1, env.legal_actions(1))
    goal_type = NODE_TYPE_IDS["goal"]

    def goal_row(graph):
        for row, node_type in zip(graph.node_features, graph.node_type_ids):
            if node_type == goal_type and row[GRAPH_F["goal_0"]] == 1.0:
                return row
        raise AssertionError("goal node not found")

    actor_row = goal_row(actor_graph)
    other_row = goal_row(other_graph)

    assert actor_row[GRAPH_F["private_known_stone"]] == 1.0
    assert actor_row[GRAPH_F["private_card_goal_stone_nw"]] == 1.0
    assert actor_row[GRAPH_F["connects_N_W"]] == 0.0
    assert actor_row[GRAPH_F["private_connects_N_W"]] == 1.0
    assert actor_row[GRAPH_F["private_connects_E_S"]] == 1.0
    assert other_row[GRAPH_F["private_known_stone"]] == 0.0
    assert other_row[GRAPH_F["connects_N_W"]] == 0.0
    assert other_row[GRAPH_F["private_connects_N_W"]] == 0.0
    assert other_row[GRAPH_F["private_connects_E_S"]] == 0.0


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
    assert tuple(output_batch.role_logits.shape) == (2, 10)
    assert tuple(output_batch.goal_logits.shape) == (2, 3)
    assert torch.isfinite(output_batch.action_logits).all()
    assert torch.isfinite(output_batch.values).all()
    assert batch.role_label_mask is not None
    assert batch.role_label_mask.tolist() == [
        [False, True, True, False, False, False, False, False, False, False],
        [False, True, True, False, False, False, False, False, False, False],
    ]


@pytest.mark.parametrize("belief_injection", ["none", "add", "second_pass"])
def test_graph_policy_forward_with_transformer_and_belief_modes(belief_injection: str):
    env_a = _env_with_public_history(822)
    env_b = _env_with_public_history(823)
    graph_a = encode_graph(env_a, env_a.agent_selection, env_a.legal_actions())
    graph_b = encode_graph(env_b, env_b.agent_selection, env_b.legal_actions())
    tensor_a = tensorize_graph(graph_a, history_max_events=12)
    tensor_b = tensorize_graph(graph_b, history_max_events=12)
    batch = collate_graph_tensors([tensor_a, tensor_b], [0, 0], "cpu")
    model = GraphPolicy.from_features(
        graph_a,
        hidden_dim=16,
        graph_layers=1,
        history_encoder="transformer",
        history_max_events=12,
        history_layers=1,
        history_heads=4,
        belief_injection=belief_injection,
        belief_post_layers=1,
        role_conditioned_heads=True,
    )

    output = model.score_graph_batches(batch)

    assert tuple(output.action_logits.shape) == (len(graph_a.actions) + len(graph_b.actions),)
    assert tuple(output.values.shape) == (2,)
    assert tuple(output.role_logits.shape) == (2, 10)
    assert tuple(output.goal_logits.shape) == (2, 3)
    assert torch.isfinite(output.action_logits).all()
    assert torch.isfinite(output.values).all()
    assert torch.isfinite(output.role_logits).all()
    assert torch.isfinite(output.goal_logits).all()


def test_history_transformer_receives_gradients_from_graph_policy():
    env = _env_with_public_history(824)
    graph = encode_graph(env, env.agent_selection, env.legal_actions())
    tensor = tensorize_graph(graph, history_max_events=12)
    model = GraphPolicy.from_features(
        graph,
        hidden_dim=16,
        graph_layers=1,
        history_encoder="transformer",
        history_max_events=12,
        history_layers=1,
        history_heads=4,
        belief_injection="second_pass",
    )

    output = model.score_graph(tensor)
    loss = output.action_logits.sum() + output.values.sum() + output.role_logits.sum() + output.goal_logits.sum()
    loss.backward()

    assert model.history_encoder is not None
    assert any(param.grad is not None for param in model.history_encoder.parameters())


def test_belief_detach_controls_action_gradient_into_belief_head():
    env = _env_with_public_history(825)
    graph = encode_graph(env, env.agent_selection, env.legal_actions())

    def belief_head_has_action_grad(detach: bool) -> bool:
        model = GraphPolicy.from_features(
            graph,
            hidden_dim=16,
            graph_layers=1,
            belief_injection="add",
            belief_detach=detach,
        )
        tensor = tensorize_graph(graph, history_max_events=model.history_max_events)
        output = model.score_graph(tensor)
        output.action_logits.sum().backward()
        return any(
            param.grad is not None and bool(torch.any(param.grad != 0.0))
            for param in model.role_belief_head.parameters()
        )

    assert belief_head_has_action_grad(False)
    assert not belief_head_has_action_grad(True)


def test_transformer_checkpoint_save_load_preserves_config(tmp_path):
    env = _env_with_public_history(826)
    graph = encode_graph(env, env.agent_selection, env.legal_actions())
    model = GraphPolicy.from_features(
        graph,
        hidden_dim=16,
        graph_layers=1,
        history_encoder="transformer",
        history_max_events=12,
        history_layers=1,
        history_heads=4,
        belief_injection="second_pass",
        belief_post_layers=1,
        role_conditioned_heads=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        tmp_path / "graph_transformer.pt",
        model=model,
        optimizer=optimizer,
        iteration=3,
        config={"model": "graph"},
    )
    loaded = GraphPolicy.from_features(
        graph,
        hidden_dim=16,
        graph_layers=1,
        history_encoder="transformer",
        history_max_events=12,
        history_layers=1,
        history_heads=4,
        belief_injection="second_pass",
        belief_post_layers=1,
        role_conditioned_heads=True,
    )
    loaded_optimizer = torch.optim.Adam(loaded.parameters(), lr=1e-3)

    payload = load_checkpoint(path, model=loaded, optimizer=loaded_optimizer)

    assert payload["model_metadata"]["history_encoder"] == "transformer"
    assert loaded.checkpoint_metadata()["belief_injection"] == "second_pass"


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
    serialized = _serialize_graph_rollout_games([game])
    assert not _contains_tensor(serialized)
    round_tripped = _deserialize_graph_rollout_games(serialized)
    assert len(round_tripped) == 1
    assert len(round_tripped[0].transitions) == len(game.transitions)
    assert round_tripped[0].transitions[0].graph.x.device.type == "cpu"
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
    assert metrics.role_belief_loss_others >= 0.0
    assert 0.0 <= metrics.role_belief_accuracy_others <= 1.0
    assert metrics.role_belief_brier_others >= 0.0
    assert metrics.goal_belief_loss >= 0.0
    assert 0.0 <= metrics.goal_belief_acc <= 1.0
    assert 0.0 <= metrics.goal_gold_prob_on_true_goal <= 1.0


def test_graph_rollout_supports_heuristic_reward_mode():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=819)
    torch.manual_seed(819)
    model = _graph_model_for_env(env)
    agent = GraphNeuralAgent(model, deterministic=False)

    game = collect_graph_game_rollout(
        env,
        agent,
        seed=819,
        max_steps=200,
        reward_mode="heuristic",
    )

    assert game.transitions
    assert all(
        torch.isfinite(torch.tensor(transition.shaping_reward))
        for transition in game.transitions
    )


def test_graph_agent_reports_role_belief_probabilities():
    env = SaboteurEnv(num_players=4)
    env.reset(seed=817)
    model = _graph_model_for_env(env)
    agent = GraphNeuralAgent(model, deterministic=True)

    _action, info = agent.act_with_info(env, env.agent_selection)

    assert len(info.role_belief_logits) == env.num_players
    assert len(info.role_belief_probs) == env.num_players
    assert all(torch.isfinite(torch.tensor(value)) for value in info.role_belief_logits)
    assert all(0.0 <= value <= 1.0 for value in info.role_belief_probs)


def test_export_graph_replay_includes_role_beliefs():
    env = SaboteurEnv(num_players=4)
    env.reset(seed=818)
    model = _graph_model_for_env(env)
    agent = GraphNeuralAgent(model, deterministic=True)

    result = play_neural_eval_game(
        agent,
        model_type="graph",
        mode="miners_only",
        num_players=4,
        seed=818,
        max_steps=200,
    )

    role_belief_steps = [
        step.get("role_beliefs", [])
        for step in result.debug.get("steps", [])
        if isinstance(step, dict) and isinstance(step.get("role_beliefs"), list) and step.get("role_beliefs")
    ]
    assert role_belief_steps
    first = role_belief_steps[0]
    assert len(first) == result.num_players
    assert {belief["player_id"] for belief in first} == set(range(result.num_players))
    assert any(belief["is_self"] for belief in first)
    assert all(0.0 <= belief["saboteur_prob"] <= 1.0 for belief in first)


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_tensor(item) for item in value)
    return False


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
    assert "reach=" in output
    assert "map_if_have=" in output
    assert "checkpoint=" in output
    assert (tmp_path / "checkpoint_0001.pt").exists()
    metrics_log = (tmp_path / "metrics.log").read_text(encoding="utf-8")
    assert "role_belief_loss=" in metrics_log
    assert "goal_belief_loss=" in metrics_log
    assert "avg_reachable_tiles=" in metrics_log
    assert "map_play_when_available_rate=" in metrics_log
    assert (tmp_path / "metrics.jsonl").exists()
