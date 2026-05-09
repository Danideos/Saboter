import pytest

from saboter.actions import Discard, MapGoal, PlayPath, RepairTool, Rockfall, SabotageTool
from saboter.agents import LegalRandomAgent
from saboter.board import Board
from saboter.cards import (
    CardType,
    GOAL_GOLD_CARD,
    GOAL_STONE_CARD,
    Role,
    Tool,
    action_card,
    path_card_by_id,
)
from saboter.env import Outcome, SaboteurEnv


def set_current_hand(env: SaboteurEnv, *cards):
    env.players[env.agent_selection].hand = list(cards)


def test_broken_tool_blocks_path_play_but_not_actions_or_discard():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=10)
    path_ew = path_card_by_id("path_ew")
    sabotage_pickaxe = action_card(CardType.SABOTAGE, (Tool.PICKAXE,))
    set_current_hand(env, path_ew, sabotage_pickaxe)
    env.players[0].broken_tools.add(Tool.CART)

    actions = env.legal_actions()

    assert any(isinstance(action, Discard) for action in actions)
    assert any(isinstance(action, SabotageTool) for action in actions)
    assert not any(isinstance(action, PlayPath) for action in actions)


def test_sabotage_prevents_duplicate_broken_tool_on_target():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=11)
    sabotage_pickaxe = action_card(CardType.SABOTAGE, (Tool.PICKAXE,))
    set_current_hand(env, sabotage_pickaxe)
    env.players[1].broken_tools.add(Tool.PICKAXE)

    actions = env.legal_actions()

    assert SabotageTool(0, 1, Tool.PICKAXE) not in actions
    assert SabotageTool(0, 2, Tool.PICKAXE) in actions


def test_repair_card_with_two_tools_repairs_one_matching_tool():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=12)
    repair_card = action_card(CardType.REPAIR, (Tool.LANTERN, Tool.CART))
    set_current_hand(env, repair_card)
    env.deck = [path_card_by_id("path_ew")]
    env.players[1].broken_tools.update({Tool.LANTERN, Tool.CART})

    actions = env.legal_actions()
    assert RepairTool(0, 1, Tool.LANTERN) in actions
    assert RepairTool(0, 1, Tool.CART) in actions

    env.step(RepairTool(0, 1, Tool.LANTERN))

    assert Tool.LANTERN not in env.players[1].broken_tools
    assert Tool.CART in env.players[1].broken_tools


def test_step_known_legal_matches_validated_step_for_legal_action():
    validated = SaboteurEnv(num_players=3)
    fast = SaboteurEnv(num_players=3)
    for env in (validated, fast):
        env.reset(seed=120)
        set_current_hand(env, action_card(CardType.MAP))
        env.deck = [path_card_by_id("path_ew")]

    action = MapGoal(0, 0)
    assert action in validated.legal_actions()

    validated.step(action)
    fast.step_known_legal(action)

    assert validated.agent_selection == fast.agent_selection
    assert validated.turn_number == fast.turn_number
    assert validated.discard_pile == fast.discard_pile
    assert [event.to_dict() for event in validated.history] == [
        event.to_dict() for event in fast.history
    ]
    assert [validated.observe(player_id) for player_id in range(3)] == [
        fast.observe(player_id) for player_id in range(3)
    ]


def test_step_known_legal_does_not_regenerate_legal_actions(monkeypatch):
    env = SaboteurEnv(num_players=3)
    env.reset(seed=121)
    set_current_hand(env, action_card(CardType.MAP))
    env.deck = [path_card_by_id("path_ew")]
    action = MapGoal(0, 0)
    assert action in env.legal_actions()

    def fail_legal_actions(player_id=None):
        raise AssertionError("step_known_legal should not call legal_actions")

    monkeypatch.setattr(env, "legal_actions", fail_legal_actions)

    env.step_known_legal(action)

    assert env.turn_number == 1
    assert env.history[-1].action_type == "map_goal"


def test_map_goal_updates_only_acting_players_private_knowledge():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=13)
    set_current_hand(env, action_card(CardType.MAP))
    env.deck = [path_card_by_id("path_ew")]

    env.step(MapGoal(0, 0))

    obs_actor = env.observe(0)
    obs_other = env.observe(1)
    assert obs_actor["known_goals"]
    assert obs_other["known_goals"] == {}
    assert all(
        tile["goal_kind"] is None
        for tile in obs_actor["board"]
        if tile["kind"] == "goal" and not tile["revealed"]
    )
    assert "revealed_goal_kind" not in obs_actor["history"][-1]
    assert "revealed_goal_kind" not in obs_other["history"][-1]


def test_observation_does_not_leak_other_roles_hands_unused_role_or_deck_order():
    env = SaboteurEnv(num_players=5)
    env.reset(seed=14)

    obs = env.observe(0)

    assert "unused_role" not in obs
    assert "deck" not in obs
    assert obs["deck_size"] == len(env.deck)
    assert obs["own_role"] == env.players[0].role.value
    assert len(obs["hand"]) == len(env.players[0].hand)
    for player in obs["players"]:
        assert "role" not in player
        assert "hand" not in player
        assert "hand_size" in player


def test_discard_hides_card_identity_in_public_history_and_discard_pile():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=15)
    hidden_card = action_card(CardType.MAP)
    set_current_hand(env, hidden_card)
    env.deck = [path_card_by_id("path_ew")]

    env.step(Discard(0))
    obs = env.observe(1)

    assert obs["public_discards"][-1] is None
    assert obs["history"][-1] == {"actor": 0, "action_type": "discard"}


def test_goal_reveal_to_gold_ends_round_with_miner_rewards():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=16)
    env.board = Board([GOAL_STONE_CARD, GOAL_GOLD_CARD, GOAL_STONE_CARD])
    env.players[0].role = Role.MINER
    env.players[1].role = Role.SABOTEUR
    env.players[2].role = Role.MINER
    path_ew = path_card_by_id("path_ew")
    for x in range(1, 7):
        env.board.place_path(path_ew, (x, 0), 0)
    set_current_hand(env, path_ew)

    env.step(PlayPath(0, 7, 0, 0))

    assert env.is_terminal()
    assert env.outcome == Outcome.MINERS_WIN
    assert env.rewards() == {0: 1.0, 1: -1.0, 2: 1.0}
    goal_tiles = {
        tile["goal_index"]: tile for tile in env.observe(0)["board"] if tile["kind"] == "goal"
    }
    assert goal_tiles[1]["revealed"]
    assert goal_tiles[1]["goal_kind"] == "gold"
    assert goal_tiles[0]["goal_kind"] is None


def test_saboteur_win_rewards_and_no_saboteur_edge_case():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=17)
    env.players[0].role = Role.SABOTEUR
    env.players[1].role = Role.MINER
    env.players[2].role = Role.MINER
    env.deck = []
    for player in env.players:
        player.hand = []

    env.step(None)

    assert env.outcome == Outcome.SABOTEURS_WIN
    assert env.rewards() == {0: 1.0, 1: -1.0, 2: -1.0}

    no_sab = SaboteurEnv(num_players=3)
    no_sab.reset(seed=18)
    for player in no_sab.players:
        player.role = Role.MINER
        player.hand = []
    no_sab.deck = []

    no_sab.step(None)

    assert no_sab.outcome == Outcome.SABOTEURS_WIN
    assert no_sab.rewards() == {0: -1.0, 1: -1.0, 2: -1.0}


def test_rockfall_removes_path_card_and_recomputes_reachability():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=19)
    path_ew = path_card_by_id("path_ew")
    env.board = Board([GOAL_STONE_CARD, GOAL_GOLD_CARD, GOAL_STONE_CARD])
    env.board.place_path(path_ew, (1, 0), 0)
    env.board.place_path(path_ew, (2, 0), 0)
    set_current_hand(env, action_card(CardType.ROCKFALL))
    env.deck = [path_card_by_id("path_ns")]

    env.step(Rockfall(0, 1, 0))

    assert env.board.tile_at((1, 0)) is None
    assert (2, 0) not in env.board.reachable_path_coords()
    assert env.observe(1)["history"][-1]["removed_card"]["id"] == "path_ew"


def test_random_agent_soak_games_complete_without_illegal_actions(pytestconfig):
    game_count = 100 if pytestconfig.getoption("--runslow") else 20
    for seed in range(game_count):
        env = SaboteurEnv(num_players=5)
        env.reset(seed=seed)
        agent = LegalRandomAgent(seed=seed)
        steps = 0
        while not env.is_terminal() and steps < 500:
            env.step(agent.act(env))
            steps += 1

        assert env.is_terminal(), f"seed {seed} did not terminate"
        assert env.outcome in {Outcome.MINERS_WIN, Outcome.SABOTEURS_WIN}
        assert set(env.rewards()) == set(range(env.num_players))
