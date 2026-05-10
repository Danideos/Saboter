from saboter.metrics_dashboard import (
    load_metrics_jsonl,
    render_metrics_dashboard,
    save_metrics_dashboard,
)


def test_render_metrics_dashboard_contains_expected_ui_and_payload(tmp_path):
    rows = [
        {
            "iteration": 1,
            "avg_reward": 0.1,
            "miners_win_rate": 0.6,
            "loss": 0.9,
            "checkpoint": "runs/demo/checkpoint_0001.pt",
        },
        {
            "iteration": 2,
            "avg_reward": 0.2,
            "miners_win_rate": 0.65,
            "loss": 0.7,
            "entropy": 1.9,
        },
    ]

    rendered = render_metrics_dashboard(rows, title="Saboter Metrics - Demo Run")

    assert rendered.startswith("<!doctype html>")
    assert 'id="metrics-data"' in rendered
    assert "Trend explorer" in rendered
    assert "Snapshot inspector" in rendered
    assert "All metrics" in rendered
    assert "avg_reward" in rendered
    assert "miners_win_rate" in rendered
    assert "checkpoint_0001" in rendered

    path = tmp_path / "metrics.html"
    save_metrics_dashboard(path, rows, title="Saboter Metrics - Demo Run")
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_load_metrics_jsonl_parses_rows_and_rejects_empty_file(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"iteration": 1, "avg_reward": 0.1}\n{"iteration": 2, "avg_reward": 0.2}\n', encoding="utf-8")

    rows = load_metrics_jsonl(path)

    assert len(rows) == 2
    assert rows[0]["iteration"] == 1
    assert rows[1]["avg_reward"] == 0.2

    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")
    try:
        load_metrics_jsonl(empty_path)
    except ValueError as exc:
        assert "does not contain any metric rows" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty metrics file")
