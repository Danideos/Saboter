import json

from saboter.evaluation import main, make_agent, play_game, run_tournament, save_replays
from saboter.env import Outcome


def test_make_agent_accepts_documented_names():
    for name in ("random", "greedy-miner", "greedy-saboteur", "role-aware", "role-inference"):
        assert make_agent(name, seed=1) is not None


def test_play_game_returns_compact_jsonable_replay():
    result = play_game(["random"], num_players=3, seed=200)
    payload = result.to_dict()

    json.dumps(payload)
    assert payload["outcome"] in {Outcome.MINERS_WIN.value, Outcome.SABOTEURS_WIN.value}
    assert payload["num_players"] == 3
    assert payload["steps"] > 0
    assert payload["illegal_action_attempts"] == 0
    assert set(payload["action_counts"])


def test_run_tournament_aggregates_metrics_and_expands_roster():
    result = run_tournament(["random", "role-aware"], games=4, num_players=5, seed=300)
    summary = result.summary

    assert summary["games"] == 4
    assert summary["expanded_roster"] == [
        "random",
        "role-aware",
        "random",
        "role-aware",
        "random",
    ]
    assert summary["miner_wins"] + summary["saboteur_wins"] == 4
    assert summary["illegal_action_attempts"] == 0
    assert "average_game_length" in summary
    assert "average_reward_by_agent" in summary


def test_save_replays_writes_summary_and_games(tmp_path):
    result = run_tournament(["random"], games=2, num_players=3, seed=400)
    path = tmp_path / "replays.json"

    save_replays(path, result)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["games"] == 2
    assert len(payload["games"]) == 2


def test_cli_can_write_html_replay(tmp_path, capsys):
    path = tmp_path / "demo.html"

    exit_code = main(
        [
            "--games",
            "1",
            "--players",
            "3",
            "--agents",
            "role-aware",
            "--seed",
            "403",
            "--html-out",
            str(path),
        ]
    )

    assert exit_code == 0
    assert "miner_win_rate" in capsys.readouterr().out
    assert 'id="stepSlider"' in path.read_text(encoding="utf-8")
