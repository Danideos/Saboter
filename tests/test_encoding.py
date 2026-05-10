from saboter.action_encoding import (
    ACTION_FEATURE_NAMES,
    encode_action_features,
    encode_action_vector,
    encode_actions,
    encode_legal_action_batch,
)
from saboter.actions import Discard, MapGoal, PlayPath, RepairTool, Rockfall, SabotageTool
from saboter.agents import LegalRandomAgent
from saboter.board import GOAL_COORDS, Board
from saboter.cards import (
    CardType,
    GOAL_GOLD_CARD,
    GOAL_STONE_NE_CARD,
    GOAL_STONE_NW_CARD,
    GoalKind,
    Role,
    Tool,
    action_card,
    path_card_by_id,
)
from saboter.env import PublicEvent
from saboter.observation import (
    BOARD_CHANNEL_NAMES,
    BOARD_HEIGHT,
    BOARD_MIN_X,
    BOARD_MIN_Y,
    BOARD_WIDTH,
    GLOBAL_FEATURE_NAMES,
    HAND_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
    MAX_HAND_SIZE,
    MAX_HISTORY,
    MAX_PLAYERS,
    PLAYER_FEATURE_NAMES,
    encode_board_tensor,
    encode_global_features,
    encode_hand,
    encode_observation,
    encode_observation_features,
    encode_players,
)
from saboter.env import SaboteurEnv


def test_observation_encoder_shapes_are_stable():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=600)

    features = encode_observation(env, 0)

    assert features.board_shape == (len(BOARD_CHANNEL_NAMES), BOARD_HEIGHT, BOARD_WIDTH)
    assert features.hand_shape[0] == MAX_HAND_SIZE
    assert features.players_shape[0] == MAX_PLAYERS
    assert features.history_shape[0] == MAX_HISTORY
    assert len(features.board) == features.board_shape[0]
    assert len(features.board[0]) == features.board_shape[1]
    assert len(features.board[0][0]) == features.board_shape[2]
    assert len(features.hand) == features.hand_shape[0]
    assert len(features.players) == features.players_shape[0]
    assert len(features.global_features) == features.global_shape[0]
    assert len(features.history) == features.history_shape[0]
    assert len(features.flat_vector()) > 0


def test_board_tensor_marks_rotated_edges_and_legal_candidates():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=601)
    env.players[0].hand = [path_card_by_id("path_es")]

    features = encode_observation(env, 0)
    actions = env.legal_actions(0)

    assert PlayPath(0, 1, 0, 180) in actions
    candidate_channel = BOARD_CHANNEL_NAMES.index("legal_candidate_position")
    row = 0 - BOARD_MIN_Y
    col = 1 - BOARD_MIN_X
    assert features.board[candidate_channel][row][col] == 1.0

    env.step(PlayPath(0, 1, 0, 180))
    after = encode_observation(env, 0)
    assert after.board[BOARD_CHANNEL_NAMES.index("has_N")][row][col] == 1.0
    assert after.board[BOARD_CHANNEL_NAMES.index("has_W")][row][col] == 1.0
    assert after.board[BOARD_CHANNEL_NAMES.index("has_E")][row][col] == 0.0


def test_observation_encoder_does_not_change_for_hidden_other_state():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=602)

    before = encode_observation(env, 0).flat_vector()
    env.unused_role = Role.SABOTEUR if env.unused_role == Role.MINER else Role.MINER
    env.deck = list(reversed(env.deck))
    env.players[1].role = Role.SABOTEUR if env.players[1].role == Role.MINER else Role.MINER
    env.players[1].hand = list(reversed(env.players[1].hand))
    after_hidden_changes = encode_observation(env, 0).flat_vector()

    assert before == after_hidden_changes

    env.players[0].role = Role.SABOTEUR if env.players[0].role == Role.MINER else Role.MINER
    after_own_role_change = encode_observation(env, 0).flat_vector()
    assert after_own_role_change != before


def test_private_map_knowledge_changes_only_that_players_encoding():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=603)
    player_zero_before = encode_observation(env, 0).flat_vector()
    player_one_before = encode_observation(env, 1).flat_vector()

    env.players[0].known_goals[1] = GoalKind.GOLD

    assert encode_observation(env, 0).flat_vector() != player_zero_before
    assert encode_observation(env, 1).flat_vector() == player_one_before


def test_hidden_goals_do_not_leak_public_kind_even_if_observation_is_malformed():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=606)
    observation = env.observe(0)
    hidden_goal = next(
        tile for tile in observation["board"] if tile["kind"] == "goal" and not tile["revealed"]
    )
    hidden_goal["goal_kind"] = GoalKind.GOLD.value
    hidden_goal["card"] = {
        "id": "goal_gold",
        "type": "goal",
        "edges": ["N", "E", "S", "W"],
        "groups": [["N", "E", "S", "W"]],
        "goal_kind": GoalKind.GOLD.value,
    }

    board = encode_board_tensor(observation, [])
    row = hidden_goal["y"] - BOARD_MIN_Y
    col = hidden_goal["x"] - BOARD_MIN_X

    assert board[BOARD_CHANNEL_NAMES.index("hidden_goal")][row][col] == 1.0
    assert board[BOARD_CHANNEL_NAMES.index("known_gold")][row][col] == 0.0
    assert board[BOARD_CHANNEL_NAMES.index("known_stone")][row][col] == 0.0

    globals_ = encode_global_features(observation)
    assert globals_[GLOBAL_FEATURE_NAMES.index(f"known_goal_{hidden_goal['goal_index']}_unknown")] == 1.0
    assert globals_[GLOBAL_FEATURE_NAMES.index(f"known_goal_{hidden_goal['goal_index']}_gold")] == 0.0


def test_start_and_revealed_goal_geometry_are_encoded():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=613)
    env.board = Board([GOAL_GOLD_CARD, GOAL_STONE_NE_CARD, GOAL_STONE_NW_CARD])
    env.board.reveal_goal(0)
    env.board.reveal_goal(1)

    board = encode_board_tensor(env.observe(0), [])
    start_row = 0 - BOARD_MIN_Y
    start_col = 0 - BOARD_MIN_X
    gold_x, gold_y = GOAL_COORDS[0]
    gold_row = gold_y - BOARD_MIN_Y
    gold_col = gold_x - BOARD_MIN_X
    stone_x, stone_y = GOAL_COORDS[1]
    stone_row = stone_y - BOARD_MIN_Y
    stone_col = stone_x - BOARD_MIN_X

    for direction in ("N", "E", "S", "W"):
        assert board[BOARD_CHANNEL_NAMES.index(f"has_{direction}")][start_row][start_col] == 1.0
        assert board[BOARD_CHANNEL_NAMES.index(f"has_{direction}")][gold_row][gold_col] == 1.0
    assert board[BOARD_CHANNEL_NAMES.index("connects_N_E")][start_row][start_col] == 1.0
    assert board[BOARD_CHANNEL_NAMES.index("connects_S_W")][gold_row][gold_col] == 1.0
    assert board[BOARD_CHANNEL_NAMES.index("has_N")][stone_row][stone_col] == 1.0
    assert board[BOARD_CHANNEL_NAMES.index("has_E")][stone_row][stone_col] == 1.0
    assert board[BOARD_CHANNEL_NAMES.index("has_S")][stone_row][stone_col] == 0.0
    assert board[BOARD_CHANNEL_NAMES.index("has_W")][stone_row][stone_col] == 0.0
    assert board[BOARD_CHANNEL_NAMES.index("connects_N_E")][stone_row][stone_col] == 1.0


def test_board_encoder_treats_missing_rotation_as_zero():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=614)
    observation = env.observe(0)
    start_tile = next(tile for tile in observation["board"] if tile["kind"] == "start")
    start_tile["rotation"] = None

    board = encode_board_tensor(observation, [])
    row = start_tile["y"] - BOARD_MIN_Y
    col = start_tile["x"] - BOARD_MIN_X

    assert board[BOARD_CHANNEL_NAMES.index("has_N")][row][col] == 1.0


def test_board_tensor_marks_reachable_frontier_and_start_distance():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=621)
    env.board.place_path(path_card_by_id("path_ew"), (1, 0), 0)

    board = encode_board_tensor(env.observe(0), [])
    start_row = 0 - BOARD_MIN_Y
    start_col = 0 - BOARD_MIN_X
    path_row = 0 - BOARD_MIN_Y
    path_col = 1 - BOARD_MIN_X
    frontier_row = 0 - BOARD_MIN_Y
    frontier_col = 2 - BOARD_MIN_X

    assert board[BOARD_CHANNEL_NAMES.index("reachable_open_E")][start_row][start_col] == 0.0
    assert board[BOARD_CHANNEL_NAMES.index("reachable_open_E")][path_row][path_col] == 1.0
    assert board[BOARD_CHANNEL_NAMES.index("reachable_open_W")][path_row][path_col] == 0.0
    assert board[BOARD_CHANNEL_NAMES.index("frontier_empty")][frontier_row][frontier_col] == 1.0
    assert (
        board[BOARD_CHANNEL_NAMES.index("distance_from_start_norm")][path_row][path_col]
        > board[BOARD_CHANNEL_NAMES.index("distance_from_start_norm")][start_row][start_col]
    )
    assert (
        board[BOARD_CHANNEL_NAMES.index("distance_from_start_norm")][frontier_row][frontier_col]
        > board[BOARD_CHANNEL_NAMES.index("distance_from_start_norm")][path_row][path_col]
    )


def test_reachable_frontier_uses_connected_group_not_whole_reachable_tile():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=622)
    env.board.place_path(path_card_by_id("dead_ew_split"), (1, 0), 0)

    board = encode_board_tensor(env.observe(0), [])
    path_row = 0 - BOARD_MIN_Y
    path_col = 1 - BOARD_MIN_X
    frontier_row = 0 - BOARD_MIN_Y
    frontier_col = 2 - BOARD_MIN_X

    assert board[BOARD_CHANNEL_NAMES.index("reachable_from_start")][path_row][path_col] == 1.0
    assert board[BOARD_CHANNEL_NAMES.index("has_E")][path_row][path_col] == 1.0
    assert board[BOARD_CHANNEL_NAMES.index("reachable_open_E")][path_row][path_col] == 0.0
    assert board[BOARD_CHANNEL_NAMES.index("frontier_empty")][frontier_row][frontier_col] == 0.0


def test_hand_encoding_includes_internal_connection_pairs():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=607)
    env.players[0].hand = [path_card_by_id("path_ew"), path_card_by_id("dead_ns_split")]

    hand = encode_hand(env.observe(0))

    assert hand[0][HAND_FEATURE_NAMES.index("connects_E_W")] == 1.0
    assert hand[0][HAND_FEATURE_NAMES.index("connects_N_S")] == 0.0
    assert hand[1][HAND_FEATURE_NAMES.index("edge_N")] == 1.0
    assert hand[1][HAND_FEATURE_NAMES.index("edge_S")] == 1.0
    assert hand[1][HAND_FEATURE_NAMES.index("connects_N_S")] == 0.0
    assert hand[1][HAND_FEATURE_NAMES.index("is_split")] == 1.0


def test_player_encoding_uses_relative_rows_not_absolute_player_ids():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=608)
    env.players[3].broken_tools.add(Tool.CART)

    players = encode_players(env.observe(2))

    assert players[0][PLAYER_FEATURE_NAMES.index("is_self")] == 1.0
    assert players[1][PLAYER_FEATURE_NAMES.index("broken_cart")] == 1.0
    assert players[2][PLAYER_FEATURE_NAMES.index("broken_cart")] == 0.0


def test_history_encoding_marks_present_actor_and_age_with_player_count_relative_positions():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=609)
    env.history.append(PublicEvent(actor=4, action_type="discard"))
    env.history.append(
        PublicEvent(actor=3, action_type="sabotage", target_player=2, tool=Tool.CART.value)
    )

    history = encode_observation(env, 2).history
    previous = history[-2]
    recent = history[-1]

    assert previous[HISTORY_FEATURE_NAMES.index("present")] == 1.0
    assert recent[HISTORY_FEATURE_NAMES.index("present")] == 1.0
    assert recent[HISTORY_FEATURE_NAMES.index("actor_present")] == 1.0
    assert previous[HISTORY_FEATURE_NAMES.index("age_norm")] > recent[HISTORY_FEATURE_NAMES.index("age_norm")]
    assert recent[HISTORY_FEATURE_NAMES.index("actor_relative_norm")] == 0.25
    assert recent[HISTORY_FEATURE_NAMES.index("target_relative_norm")] == 0.0
    assert history[-3][HISTORY_FEATURE_NAMES.index("present")] == 0.0


def test_history_encoder_keeps_opponent_discard_card_type_hidden():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=615)
    env.players[0].hand = [action_card(CardType.MAP)]
    env.deck = [path_card_by_id("path_ew")]

    env.step(Discard(0))

    observation = env.observe(1)
    assert observation["history"][-1] == {"actor": 0, "action_type": "discard"}
    row = encode_observation(env, 1).history[-1]
    assert row[HISTORY_FEATURE_NAMES.index("action_discard")] == 1.0
    for card_type in (CardType.PATH, CardType.SABOTAGE, CardType.REPAIR, CardType.MAP, CardType.ROCKFALL):
        assert row[HISTORY_FEATURE_NAMES.index(f"card_type_{card_type.value}")] == 0.0


def test_history_encoder_keeps_opponent_map_result_hidden():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=616)
    env.players[0].hand = [action_card(CardType.MAP)]
    env.deck = [path_card_by_id("path_ew")]

    env.step(MapGoal(0, 0))

    observation = env.observe(1)
    assert "revealed_goal_kind" not in observation["history"][-1]
    row = encode_observation(env, 1).history[-1]
    assert row[HISTORY_FEATURE_NAMES.index("action_map_goal")] == 1.0
    assert row[HISTORY_FEATURE_NAMES.index("card_type_map")] == 1.0
    assert row[HISTORY_FEATURE_NAMES.index("goal_0")] == 1.0
    assert row[HISTORY_FEATURE_NAMES.index("revealed_gold")] == 0.0
    assert row[HISTORY_FEATURE_NAMES.index("revealed_stone")] == 0.0


def test_every_public_history_action_type_can_be_encoded():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=617)
    env.history = [
        PublicEvent(actor=0, action_type="discard"),
        PublicEvent(
            actor=1,
            action_type="play_path",
            card=path_card_by_id("path_ew").public_dict(),
            x=1,
            y=0,
            rotation=0,
        ),
        PublicEvent(
            actor=2,
            action_type="sabotage",
            card=action_card(CardType.SABOTAGE, (Tool.PICKAXE,)).public_dict(),
            target_player=3,
            tool=Tool.PICKAXE.value,
        ),
        PublicEvent(
            actor=3,
            action_type="repair",
            card=action_card(CardType.REPAIR, (Tool.PICKAXE, Tool.CART)).public_dict(),
            target_player=2,
            tool=Tool.CART.value,
        ),
        PublicEvent(
            actor=4,
            action_type="map_goal",
            card=action_card(CardType.MAP).public_dict(),
            goal_index=1,
        ),
        PublicEvent(
            actor=0,
            action_type="rockfall",
            card=action_card(CardType.ROCKFALL).public_dict(),
            x=1,
            y=0,
            removed_card=path_card_by_id("path_ew").public_dict(),
        ),
        PublicEvent(actor=1, action_type="reveal_goal", goal_index=2, revealed_goal_kind=GoalKind.STONE.value),
    ]

    history = encode_observation(env, 0).history

    for action_type in ("discard", "play_path", "sabotage", "repair", "map_goal", "rockfall", "reveal_goal"):
        assert any(row[HISTORY_FEATURE_NAMES.index(f"action_{action_type}")] == 1.0 for row in history)


def test_global_encoder_counts_off_board_tiles_and_legal_actions():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=610)
    observation = env.observe(0)
    observation["board"].append({"x": 99, "y": 99, "kind": "path", "revealed": True, "card": None})

    features = encode_global_features(observation, [PlayPath(0, 99, 99, 0)])

    assert features[GLOBAL_FEATURE_NAMES.index("off_board_tile_count_norm")] > 0.0
    assert features[GLOBAL_FEATURE_NAMES.index("off_board_legal_action_count_norm")] > 0.0


def test_action_encoder_is_deterministic_and_marks_action_details():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=604)
    env.players[0].hand = [action_card(CardType.SABOTAGE, (Tool.PICKAXE,))]

    encoded_a = encode_actions(env, 0)
    encoded_b = encode_actions(env, 0)

    assert [item.vector for item in encoded_a] == [item.vector for item in encoded_b]
    sabotage_features = next(
        item for item in encoded_a if isinstance(item.action, SabotageTool) and item.action.tool == Tool.PICKAXE
    )
    vector = sabotage_features.vector
    assert len(vector) == len(ACTION_FEATURE_NAMES)
    assert vector[ACTION_FEATURE_NAMES.index("action_sabotage_tool")] == 1.0
    assert vector[ACTION_FEATURE_NAMES.index("selected_tool_pickaxe")] == 1.0
    assert vector[ACTION_FEATURE_NAMES.index("target_present")] == 1.0


def test_action_encoder_includes_path_card_connection_pairs():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=611)
    env.players[0].hand = [path_card_by_id("path_ew")]

    encoded = encode_actions(env, 0)
    play = next(item for item in encoded if item.action == PlayPath(0, 1, 0, 0))

    assert play.vector[ACTION_FEATURE_NAMES.index("card_connects_E_W")] == 1.0
    assert play.vector[ACTION_FEATURE_NAMES.index("card_connects_N_S")] == 0.0


def test_action_encoder_rotates_card_geometry_and_connection_pairs():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=618)
    env.players[0].hand = [path_card_by_id("path_ne")]

    vector = encode_action_vector(env.observe(0), PlayPath(0, 1, 0, 180))

    assert vector[ACTION_FEATURE_NAMES.index("card_edge_S")] == 1.0
    assert vector[ACTION_FEATURE_NAMES.index("card_edge_W")] == 1.0
    assert vector[ACTION_FEATURE_NAMES.index("card_connects_S_W")] == 1.0
    assert vector[ACTION_FEATURE_NAMES.index("card_connects_N_E")] == 0.0


def test_action_target_encoding_uses_relative_one_hot_slots():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=619)
    env.players[2].hand = [action_card(CardType.SABOTAGE, (Tool.PICKAXE,))]

    action = SabotageTool(0, 3, Tool.PICKAXE)
    encoded = encode_actions(env, 2, [action])[0]

    assert "target_player_3" not in ACTION_FEATURE_NAMES
    assert encoded.vector[ACTION_FEATURE_NAMES.index("target_relative_1")] == 1.0
    assert encoded.vector[ACTION_FEATURE_NAMES.index("target_relative_3")] == 0.0
    assert encoded.vector[ACTION_FEATURE_NAMES.index("target_relative_norm")] == 0.25


def test_encode_actions_observes_once_for_the_batch(monkeypatch):
    env = SaboteurEnv(num_players=3)
    env.reset(seed=620)
    call_count = 0
    original_observe = env.observe

    def counting_observe(player_id: int):
        nonlocal call_count
        call_count += 1
        return original_observe(player_id)

    monkeypatch.setattr(env, "observe", counting_observe)

    encode_actions(env, 0)

    assert call_count == 1


def test_lower_level_encoders_reuse_prebuilt_observation():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=621)
    observation = env.observe(0)
    actions = env.legal_actions(0)

    direct_obs = encode_observation_features(observation, actions)
    env_obs = encode_observation(env, 0, actions)
    direct_actions = encode_action_features(observation, actions)
    env_actions = encode_actions(env, 0, actions)

    assert direct_obs.flat_vector() == env_obs.flat_vector()
    assert [features.vector for features in direct_actions] == [
        features.vector for features in env_actions
    ]


def test_encoder_rejects_non_base_rotation():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=612)
    env.players[0].hand = [path_card_by_id("path_ew")]

    try:
        encode_board_tensor(env.observe(0), [PlayPath(0, 1, 0, 90)])
    except ValueError as exc:
        assert "0 or 180" in str(exc)
    else:
        raise AssertionError("90 degree rotation should fail for base Saboteur encoders")


def test_legal_action_batch_encodes_all_current_legal_actions():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=605)

    actions = env.legal_actions(0)
    batch = encode_legal_action_batch(env, 0)

    assert len(batch) == len(actions)
    assert all(len(row) == len(ACTION_FEATURE_NAMES) for row in batch)


def test_random_game_encoder_soak_does_not_crash_or_change_shapes():
    for seed in range(5):
        env = SaboteurEnv(num_players=5)
        env.reset(seed=seed)
        agent = LegalRandomAgent(seed=seed)
        expected_shapes = None
        steps = 0
        while not env.is_terminal() and steps < 200:
            player_id = env.agent_selection
            legal_actions = env.legal_actions(player_id)
            obs = encode_observation(env, player_id, legal_actions)
            action_features = encode_actions(env, player_id, legal_actions)
            shape_tuple = (
                obs.board_shape,
                obs.hand_shape,
                obs.players_shape,
                obs.global_shape,
                obs.history_shape,
                action_features[0].shape if action_features else (len(ACTION_FEATURE_NAMES),),
            )
            if expected_shapes is None:
                expected_shapes = shape_tuple
            assert shape_tuple == expected_shapes
            assert len(action_features) == len(legal_actions)
            env.step(agent.act(env, player_id))
            steps += 1

        assert env.is_terminal()
