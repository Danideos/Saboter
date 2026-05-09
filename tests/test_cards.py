from collections import Counter

from saboter.cards import (
    CardType,
    Role,
    build_action_deck,
    build_path_deck,
    hand_size_for_player_count,
    role_pool_for_player_count,
)
from saboter.env import SaboteurEnv


def test_base_deck_counts():
    path_deck = build_path_deck()
    action_deck = build_action_deck()

    assert len(path_deck) == 40
    assert len(action_deck) == 27

    action_counts = Counter(card.type for card in action_deck)
    assert action_counts == {
        CardType.SABOTAGE: 9,
        CardType.REPAIR: 9,
        CardType.MAP: 6,
        CardType.ROCKFALL: 3,
    }


def test_role_pool_counts_include_one_unused_role_card():
    expected = {
        3: (3, 1),
        4: (4, 1),
        5: (4, 2),
        6: (5, 2),
        7: (5, 3),
        8: (6, 3),
        9: (7, 3),
        10: (7, 4),
    }
    for num_players, (miners, saboteurs) in expected.items():
        pool = role_pool_for_player_count(num_players)
        assert len(pool) == num_players + 1
        assert pool.count(Role.MINER) == miners
        assert pool.count(Role.SABOTEUR) == saboteurs


def test_hand_sizes_by_player_count():
    for num_players in range(3, 6):
        assert hand_size_for_player_count(num_players) == 6
    for num_players in range(6, 8):
        assert hand_size_for_player_count(num_players) == 5
    for num_players in range(8, 11):
        assert hand_size_for_player_count(num_players) == 4


def test_reset_is_deterministic_for_same_seed():
    env_a = SaboteurEnv(num_players=5)
    env_b = SaboteurEnv(num_players=5)
    env_a.reset(seed=123)
    env_b.reset(seed=123)

    assert [player.role for player in env_a.players] == [player.role for player in env_b.players]
    assert env_a.unused_role == env_b.unused_role
    assert [[card.id for card in player.hand] for player in env_a.players] == [
        [card.id for card in player.hand] for player in env_b.players
    ]
    assert [card.id for card in env_a.deck] == [card.id for card in env_b.deck]


def test_reset_deals_expected_hand_sizes():
    for num_players, expected_hand_size in ((3, 6), (6, 5), (8, 4)):
        env = SaboteurEnv(num_players=num_players)
        env.reset(seed=1)
        assert [len(player.hand) for player in env.players] == [expected_hand_size] * num_players

