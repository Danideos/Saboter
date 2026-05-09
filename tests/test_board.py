import pytest

from saboter.board import Board
from saboter.evaluation import play_game
from saboter.cards import GOAL_GOLD_CARD, GOAL_STONE_CARD, path_card_by_id


def make_board() -> Board:
    return Board([GOAL_STONE_CARD, GOAL_GOLD_CARD, GOAL_STONE_CARD])


def test_path_placement_requires_matching_edges_and_start_reachability():
    board = make_board()
    path_ew = path_card_by_id("path_ew")
    path_ns = path_card_by_id("path_ns")

    assert board.can_place_path(path_ew, (1, 0), 0)
    assert not board.can_place_path(path_ns, (1, 0), 0)
    assert not board.can_place_path(path_ew, (3, 0), 0)


def test_path_cards_support_180_degree_rotation_only():
    board = make_board()
    path_es = path_card_by_id("path_es")

    assert not board.can_place_path(path_es, (1, 0), 0)
    assert board.can_place_path(path_es, (1, 0), 180)
    assert not board.can_place_path(path_es, (1, 0), 90)


def test_reachability_tracks_disconnected_after_rockfall():
    board = make_board()
    path_ew = path_card_by_id("path_ew")
    board.place_path(path_ew, (1, 0), 0)
    board.place_path(path_ew, (2, 0), 0)

    assert (2, 0) in board.reachable_path_coords()
    removed = board.remove_path((1, 0))

    assert removed == path_ew
    assert (2, 0) not in board.reachable_path_coords()


def test_rockfall_cannot_remove_start_or_goal_cards():
    board = make_board()

    with pytest.raises(ValueError):
        board.remove_path((0, 0))
    with pytest.raises(ValueError):
        board.remove_path((8, 0))


def test_demo_replay_has_no_mismatched_adjacent_non_goal_edges():
    result = play_game(["role-aware"], num_players=5, seed=33)
    tiles = {
        (tile["x"], tile["y"]): tile
        for tile in result.final_board
        if tile["kind"] != "goal"
    }
    deltas = {
        "N": (0, -1, "S"),
        "E": (1, 0, "W"),
        "S": (0, 1, "N"),
        "W": (-1, 0, "E"),
    }

    for (x, y), tile in tiles.items():
        edges = _rotated_edges(tile)
        for direction, (dx, dy, opposite) in deltas.items():
            neighbor = tiles.get((x + dx, y + dy))
            if neighbor is None:
                continue
            assert (direction in edges) == (opposite in _rotated_edges(neighbor))


def _rotated_edges(tile):
    edges = set(tile["card"]["edges"])
    if tile.get("rotation", 0) % 360 == 180:
        opposite = {"N": "S", "E": "W", "S": "N", "W": "E"}
        return {opposite[edge] for edge in edges}
    return edges
