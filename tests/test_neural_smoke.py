import pytest

torch = pytest.importorskip("torch")

from scripts.smoke_test_neural_agent import main as neural_smoke_main
from scripts.collect_rollouts_smoke import main as rollout_smoke_main
from scripts.train_ppo import (
    _split_games,
    _prune_checkpoints,
    evaluate_miners_only,
    evaluate_vs_legal_random,
    main as train_ppo_main,
)
from saboter.action_encoding import ACTION_FEATURE_NAMES, encode_actions
from saboter.actions import Discard, MapGoal, PlayPath, Rockfall, SabotageTool
from saboter.agents.neural_agent import NeuralAgent
from saboter.agents.random_agent import LegalRandomAgent
from saboter.cards import Tool
from saboter.env import SaboteurEnv
from saboter.models.policy import SaboteurPolicy
from saboter.observation import encode_observation
from saboter.training.curriculum import filter_actions_for_training_mode
from saboter.training.rollout import collect_game_rollout
from saboter.training.returns import discounted_returns
from saboter.training.tensorize import tensorize_actions, tensorize_observation


def _initial_features(env: SaboteurEnv):
    player_id = env.agent_selection
    legal_actions = env.legal_actions(player_id)
    obs_features = encode_observation(env, player_id, legal_actions)
    action_features = encode_actions(env, player_id, legal_actions)
    return obs_features, action_features


def _policy_for_env(env: SaboteurEnv) -> SaboteurPolicy:
    obs_features, action_features = _initial_features(env)
    return SaboteurPolicy.from_features(obs_features, len(action_features[0].vector))


def test_tensorize_shapes_match_encoder_shapes():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=700)
    obs_features, action_features = _initial_features(env)

    board, nonboard = tensorize_observation(obs_features)
    actions = tensorize_actions(action_features)

    assert tuple(board.shape) == (1, *obs_features.board_shape)
    expected_nonboard = (
        obs_features.hand_shape[0] * obs_features.hand_shape[1]
        + obs_features.players_shape[0] * obs_features.players_shape[1]
        + obs_features.global_shape[0]
        + obs_features.history_shape[0] * obs_features.history_shape[1]
    )
    assert tuple(nonboard.shape) == (1, expected_nonboard)
    assert tuple(actions.shape) == (len(action_features), len(ACTION_FEATURE_NAMES))


def test_tensorize_actions_rejects_empty_batches():
    with pytest.raises(ValueError, match="empty action feature"):
        tensorize_actions([])


def test_policy_scores_legal_action_batch_and_value_for_initial_state():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=701)
    obs_features, action_features = _initial_features(env)
    model = SaboteurPolicy.from_features(obs_features, len(action_features[0].vector))
    board, nonboard = tensorize_observation(obs_features)
    actions = tensorize_actions(action_features)

    logits, value = model.score_actions(board, nonboard, actions)

    assert tuple(logits.shape) == (len(action_features),)
    assert tuple(value.shape) == (1,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_policy_scores_midgame_state():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=702)
    model = _policy_for_env(env)
    agent = LegalRandomAgent(seed=702)
    for _step in range(5):
        if env.is_terminal():
            break
        env.step(agent.act(env))

    player_id = env.agent_selection
    legal_actions = env.legal_actions(player_id)
    if not legal_actions:
        env.step(None)
        player_id = env.agent_selection
        legal_actions = env.legal_actions(player_id)
    obs_features = encode_observation(env, player_id, legal_actions)
    action_features = encode_actions(env, player_id, legal_actions)
    board, nonboard = tensorize_observation(obs_features)
    actions = tensorize_actions(action_features)

    logits, value = model.score_actions(board, nonboard, actions)

    assert tuple(logits.shape) == (len(action_features),)
    assert tuple(value.shape) == (1,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_policy_batched_action_scoring_matches_single_state_scoring():
    env_a = SaboteurEnv(num_players=5)
    env_a.reset(seed=710)
    obs_a, actions_a = _initial_features(env_a)
    env_b = SaboteurEnv(num_players=5)
    env_b.reset(seed=711)
    random_agent = LegalRandomAgent(seed=711)
    for _step in range(3):
        env_b.step(random_agent.act(env_b))
    obs_b, actions_b = _initial_features(env_b)
    model = SaboteurPolicy.from_features(obs_a, len(actions_a[0].vector))
    board_a, nonboard_a = tensorize_observation(obs_a)
    board_b, nonboard_b = tensorize_observation(obs_b)
    action_tensor_a = tensorize_actions(actions_a)
    action_tensor_b = tensorize_actions(actions_b)

    logits_a, value_a = model.score_actions(board_a, nonboard_a, action_tensor_a)
    logits_b, value_b = model.score_actions(board_b, nonboard_b, action_tensor_b)
    batched_logits, batched_values = model.score_action_batches(
        torch.cat([board_a, board_b], dim=0),
        torch.cat([nonboard_a, nonboard_b], dim=0),
        torch.cat([action_tensor_a, action_tensor_b], dim=0),
        torch.tensor([0] * len(actions_a) + [1] * len(actions_b), dtype=torch.long),
    )

    split_a = len(actions_a)
    assert torch.allclose(batched_logits[:split_a], logits_a, atol=1e-6)
    assert torch.allclose(batched_logits[split_a:], logits_b, atol=1e-6)
    assert torch.allclose(batched_values, torch.cat([value_a, value_b]), atol=1e-6)


def test_neural_agent_returns_legal_actions_in_deterministic_and_sampling_modes():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=703)
    torch.manual_seed(703)
    model = _policy_for_env(env)

    deterministic_agent = NeuralAgent(model, deterministic=True)
    assert not model.training
    deterministic_action = deterministic_agent.act(env, env.agent_selection)
    assert deterministic_action in env.legal_actions(env.agent_selection)

    sampling_agent = NeuralAgent(model, deterministic=False)
    sampled_action = sampling_agent.act(env, env.agent_selection)
    assert sampled_action in env.legal_actions(env.agent_selection)


def test_neural_agent_act_with_info_returns_ppo_metadata():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=706)
    torch.manual_seed(706)
    model = _policy_for_env(env)
    agent = NeuralAgent(model, deterministic=False)

    action, info = agent.act_with_info(env, env.agent_selection)

    assert action in env.legal_actions(env.agent_selection)
    assert info.action_features[info.action_index].action == action
    assert isinstance(info.log_prob, float)
    assert isinstance(info.value, float)
    assert isinstance(info.entropy, float)
    assert info.entropy >= 0.0


def test_short_neural_agent_game_completes_without_illegal_actions():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=704)
    torch.manual_seed(704)
    model = _policy_for_env(env)
    agent = NeuralAgent(model, deterministic=False)
    steps = 0

    while not env.is_terminal() and steps < 200:
        player_id = env.agent_selection
        legal_actions = env.legal_actions(player_id)
        action = agent.act(env, player_id) if legal_actions else None
        assert action is None or action in legal_actions
        env.step(action)
        steps += 1

    assert env.is_terminal()


def test_collect_game_rollout_assigns_terminal_rewards_and_detached_tensors():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=707)
    torch.manual_seed(707)
    model = _policy_for_env(env)
    agent = NeuralAgent(model, deterministic=False)

    game = collect_game_rollout(env, agent, seed=707, max_steps=200)

    assert game.transitions
    assert game.steps >= len(game.transitions)
    assert set(game.rewards) == {0, 1, 2}
    for transition in game.transitions:
        assert transition.board.device.type == "cpu"
        assert transition.nonboard.device.type == "cpu"
        assert transition.actions.device.type == "cpu"
        assert not transition.board.requires_grad
        assert not transition.actions.requires_grad
        assert transition.reward == transition.terminal_reward + transition.shaping_reward
        assert 0 <= transition.action_index < transition.actions.shape[0]
        assert transition.action_type in {
            "discard",
            "play_path",
            "sabotage",
            "repair",
            "map_goal",
            "rockfall",
        }
    assert game.transitions[-1].done is True
    assert game.transitions[-1].terminal_reward == game.rewards[game.transitions[-1].player_id]
    assert all(transition.terminal_reward == 0.0 for transition in game.transitions[:-1])
    assert all(transition.done is False for transition in game.transitions[:-1])


def test_discounted_returns_propagate_terminal_reward_with_episode_resets():
    returns = discounted_returns(
        rewards=[0.0, 0.0, 1.0, 0.0, 2.0],
        dones=[False, False, True, False, True],
        gamma=0.5,
    )

    assert torch.allclose(
        returns,
        torch.tensor([0.25, 0.5, 1.0, 1.0, 2.0]),
    )


def test_miners_only_rollout_filters_non_path_curriculum_actions():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=713)
    torch.manual_seed(713)
    model = _policy_for_env(env)
    agent = NeuralAgent(model, deterministic=False)

    game = collect_game_rollout(
        env,
        agent,
        seed=713,
        max_steps=200,
        reward_mode="progress",
        training_mode="miners_only",
    )

    assert game.transitions
    assert {transition.role for transition in game.transitions} == {"miner"}
    assert {
        transition.action_type
        for transition in game.transitions
    } <= {"discard", "play_path", "map_goal"}


def test_miners_only_action_filter_can_allow_all_actions():
    actions = [
        PlayPath(0, 1, 0, 0),
        Discard(1),
        MapGoal(2, 0),
        SabotageTool(3, 1, Tool.PICKAXE),
        Rockfall(4, 1, 0),
    ]

    assert filter_actions_for_training_mode(actions, "miners_only", "all") == actions
    assert filter_actions_for_training_mode(
        actions,
        "miners_only",
        "path_discard_map",
    ) == actions[:3]


def test_neural_smoke_script_main_runs_modest_game_count(capsys):
    exit_code = neural_smoke_main(["--games", "2", "--players", "3", "--seed", "705"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "completed_games=2" in output
    assert "average_legal_actions=" in output
    assert "action_counts=" in output


def test_rollout_smoke_script_main_runs_modest_game_count(capsys):
    exit_code = rollout_smoke_main(["--games", "2", "--players", "3", "--seed", "708"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "games_collected=2" in output
    assert "transitions_collected=" in output
    assert "mean_entropy=" in output
    assert "OK" in output


def test_train_ppo_script_runs_one_iteration_and_saves_checkpoint(tmp_path, capsys):
    exit_code = train_ppo_main(
        [
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
            "709",
            "--ppo-epochs",
            "1",
            "--batch-size",
            "16",
            "--eval-games",
            "1",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "iter=1" in output
    assert "policy_loss=" in output
    assert "play_path_rate=" in output
    assert "avg_rollout_entropy=" in output
    assert "eval_vs_random_miners_win_rate=" in output
    assert "checkpoint=" in output
    assert (tmp_path / "checkpoint_0001.pt").exists()


def test_evaluate_vs_random_supports_multiple_neural_seats():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=712)
    model = _policy_for_env(env)

    metrics = evaluate_vs_legal_random(
        model,
        num_players=3,
        games=1,
        seed=712,
        device="cpu",
        max_steps=200,
        neural_count=2,
    )

    assert "eval_2_ours_vs_random_miners_win_rate" in metrics
    assert "eval_2_ours_vs_random_neural_avg_reward" in metrics


def test_evaluate_miners_only_reports_progress_metrics():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=714)
    model = _policy_for_env(env)

    metrics = evaluate_miners_only(
        model,
        num_players=3,
        games=1,
        seed=714,
        device="cpu",
        max_steps=200,
    )

    assert "eval_miners_only_win_rate" in metrics
    assert "eval_miners_only_gold_reaches" in metrics
    assert "eval_miners_only_public_stone_reaches" in metrics
    assert "eval_miners_only_avg_reachable_tiles" in metrics
    assert "eval_miners_only_avg_min_distance_to_goal" in metrics
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_split_games_distributes_work_across_workers():
    assert _split_games(8, 3) == [3, 3, 2]
    assert _split_games(2, 8) == [1, 1]


def test_prune_checkpoints_keeps_latest_n(tmp_path):
    for iteration in range(1, 8):
        (tmp_path / f"checkpoint_{iteration:04d}.pt").write_text("x", encoding="utf-8")

    _prune_checkpoints(tmp_path, 5)

    assert [path.name for path in sorted(tmp_path.glob("checkpoint_*.pt"))] == [
        "checkpoint_0003.pt",
        "checkpoint_0004.pt",
        "checkpoint_0005.pt",
        "checkpoint_0006.pt",
        "checkpoint_0007.pt",
    ]
