from saboter.actions import Discard, MapGoal, PlayPath, RepairTool, SabotageTool
from saboter.agents import GreedyMinerAgent, GreedySaboteurAgent, HeuristicRoleInferenceAgent
from saboter.cards import CardType, Tool, action_card, path_card_by_id
from saboter.env import PublicEvent, SaboteurEnv


def set_current_hand(env: SaboteurEnv, *cards):
    env.players[env.agent_selection].hand = list(cards)


def test_greedy_miner_extends_reachable_path_when_holding_path_card():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=100)
    set_current_hand(env, path_card_by_id("path_ew"))

    action = GreedyMinerAgent(seed=1).act(env)

    assert action == PlayPath(0, 1, 0, 0)
    assert action in env.legal_actions()


def test_greedy_miner_prefers_center_lane_path_over_side_path():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=104)
    set_current_hand(env, path_card_by_id("path_ns"), path_card_by_id("path_ew"))

    action = GreedyMinerAgent(seed=1).act(env)

    assert action == PlayPath(1, 1, 0, 0)
    assert action in env.legal_actions()


def test_greedy_miner_discards_dead_path_when_useful_path_exists():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=105)
    set_current_hand(env, path_card_by_id("dead_nw_split"), path_card_by_id("path_ew"))

    action = GreedyMinerAgent(seed=1).act(env)

    assert action == PlayPath(1, 1, 0, 0)
    assert not isinstance(action, Discard)


def test_greedy_miner_cycles_sabotage_card_when_no_better_card_exists():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=106)
    set_current_hand(env, action_card(CardType.SABOTAGE, (Tool.PICKAXE,)))

    action = GreedyMinerAgent(seed=1).act(env)

    assert action == Discard(0)
    assert action in env.legal_actions()


def test_greedy_miner_uses_map_for_unknown_goal_information():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=107)
    set_current_hand(env, action_card(CardType.MAP), action_card(CardType.SABOTAGE, (Tool.PICKAXE,)))

    action = GreedyMinerAgent(seed=1).act(env)

    assert isinstance(action, MapGoal)
    assert action in env.legal_actions()


def test_greedy_miner_repairs_self_before_other_actions():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=101)
    set_current_hand(
        env,
        path_card_by_id("path_ew"),
        action_card(CardType.REPAIR, (Tool.CART,)),
    )
    env.players[0].broken_tools.add(Tool.CART)

    action = GreedyMinerAgent(seed=1).act(env)

    assert action == RepairTool(1, 0, Tool.CART)
    assert action in env.legal_actions()


def test_greedy_saboteur_prefers_sabotage_over_path_progress():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=102)
    set_current_hand(
        env,
        path_card_by_id("path_ew"),
        action_card(CardType.SABOTAGE, (Tool.PICKAXE,)),
    )

    action = GreedySaboteurAgent(seed=1).act(env)

    assert isinstance(action, SabotageTool)
    assert action.target_player != env.agent_selection
    assert action in env.legal_actions()


def test_role_inference_scores_public_sabotage_as_suspicious():
    env = SaboteurEnv(num_players=3)
    env.reset(seed=103)
    env.history.append(
        PublicEvent(actor=1, action_type="sabotage", target_player=0, tool=Tool.CART.value)
    )
    env.history.append(PublicEvent(actor=2, action_type="repair", target_player=0, tool=Tool.CART.value))

    obs = env.observe(0)
    scores = HeuristicRoleInferenceAgent(seed=1).suspicion_scores(obs)

    assert scores[0] == 0.0
    assert scores[1] > scores[2]
