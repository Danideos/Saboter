from saboter.evaluation import play_game
from saboter.visualization import (
    build_public_board_snapshots,
    render_board,
    render_event,
    render_game,
    render_html_game,
    save_html_replay,
)


def test_render_event_formats_public_actions():
    assert render_event({"actor": 1, "action_type": "discard"}) == "P1 discarded face-down"
    assert (
        render_event(
            {
                "actor": 2,
                "action_type": "sabotage",
                "target_player": 0,
                "tool": "cart",
            }
        )
        == "P2 broke P0's cart"
    )


def test_render_board_shows_core_symbols():
    board = [
        {
            "x": 0,
            "y": 0,
            "kind": "start",
            "revealed": True,
            "card": {"edges": ["N", "E", "S", "W"]},
        },
        {"x": 1, "y": 0, "kind": "goal", "revealed": False, "card": None},
        {
            "x": 2,
            "y": 0,
            "kind": "goal",
            "revealed": True,
            "goal_kind": "gold",
            "card": {"edges": ["N", "E", "S", "W"]},
        },
    ]

    rendered = render_board(board)

    assert "S" in rendered
    assert "?" in rendered
    assert "$" in rendered


def test_render_board_applies_card_rotation():
    rendered = render_board(
        [
            {
                "x": 0,
                "y": 0,
                "kind": "path",
                "rotation": 180,
                "revealed": True,
                "card": {"edges": ["S", "W"], "groups": [["S", "W"]]},
            }
        ]
    )

    lines = rendered.splitlines()
    assert lines[1].strip() == "|"
    assert "+-" in lines[2]
    assert lines[3].strip() == ""


def test_render_board_marks_dead_and_split_path_cards():
    rendered = render_board(
        [
            {
                "x": 0,
                "y": 0,
                "kind": "path",
                "rotation": 0,
                "revealed": True,
                "card": {
                    "id": "dead_ns_split",
                    "edges": ["N", "S"],
                    "groups": [["N"], ["S"]],
                },
            }
        ]
    )

    board_body = "\n".join(rendered.splitlines()[1:])
    assert "x" in board_body
    assert "+" not in board_body


def test_render_game_includes_summary_events_and_final_board():
    result = play_game(["role-aware"], num_players=3, seed=501)

    rendered = render_game(result, max_events=3)

    assert "Seed: 501" in rendered
    assert "Players:" in rendered
    assert "Events:" in rendered
    assert "Final Board:" in rendered
    assert "Legend:" in rendered


def test_public_board_snapshots_track_path_reveal_and_rockfall_events():
    game = {
        "history": [
            {
                "actor": 0,
                "action_type": "play_path",
                "card": {"id": "path_ew", "type": "path", "edges": ["E", "W"]},
                "x": 1,
                "y": 0,
                "rotation": 0,
            },
            {"actor": 1, "action_type": "reveal_goal", "goal_index": 1, "revealed_goal_kind": "gold"},
            {"actor": 2, "action_type": "rockfall", "x": 1, "y": 0},
        ]
    }

    snapshots = build_public_board_snapshots(game)

    assert len(snapshots) == 4
    assert not any(tile["x"] == 1 and tile["y"] == 0 for tile in snapshots[0])
    assert any(tile["x"] == 1 and tile["y"] == 0 for tile in snapshots[1])
    assert any(
        tile.get("goal_index") == 1 and tile.get("goal_kind") == "gold"
        for tile in snapshots[2]
    )
    assert not any(tile["x"] == 1 and tile["y"] == 0 for tile in snapshots[3])


def test_render_html_game_contains_slider_payload_and_board_script(tmp_path):
    result = play_game(["role-aware"], num_players=3, seed=502)

    html = render_html_game(result)

    assert "<!doctype html>" in html
    assert 'id="stepSlider"' in html
    assert 'id="replay-data"' in html
    assert "Saboteur Replay" in html
    assert "snapshots" in html
    assert "renderTunnels" in html

    path = tmp_path / "replay.html"
    save_html_replay(path, result)
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")
