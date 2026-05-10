"""Self-contained HTML dashboard for training metrics JSONL files."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


MetricRow = Mapping[str, object]

GROUP_ORDER = (
    "Training core",
    "Rewards & outcomes",
    "Action rates",
    "State & beliefs",
    "Evaluation",
    "Counts & schedule",
    "Other",
)

GROUP_PREFERRED_ORDER = {
    "Training core": [
        "loss",
        "policy_loss",
        "value_loss",
        "goal_belief_loss",
        "role_belief_loss",
        "entropy",
        "avg_rollout_entropy",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
    ],
    "Rewards & outcomes": [
        "avg_reward",
        "avg_shaping_reward",
        "avg_terminal_reward",
        "miners_win_rate",
        "avg_game_length",
        "avg_gold_reaches",
        "avg_public_stone_reaches",
        "avg_revealed_goals",
        "avg_rollout_value",
    ],
    "Action rates": [
        "play_path_rate",
        "play_path_rate_miner",
        "play_path_rate_saboteur",
        "discard_rate",
        "discard_rate_miner",
        "discard_rate_saboteur",
        "repair_rate",
        "rockfall_rate",
        "sabotage_rate",
        "map_goal_rate",
        "map_play_when_available_rate",
    ],
    "State & beliefs": [
        "avg_min_distance_to_goal",
        "avg_reachable_tiles",
        "avg_frontier_empty_cells",
        "avg_legal_actions",
        "avg_private_goal_knowledge_count",
    ],
    "Counts & schedule": [
        "iteration",
        "updates",
        "games",
        "transitions",
        "map_available_count",
        "map_play_when_available_count",
    ],
}

DEFAULT_SELECTED_METRICS = [
    "avg_reward",
    "miners_win_rate",
    "loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
]

PRESET_DEFINITIONS = [
    (
        "core",
        "Core",
        [
            "avg_reward",
            "miners_win_rate",
            "loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
        ],
    ),
    (
        "rewards",
        "Rewards",
        [
            "avg_reward",
            "avg_shaping_reward",
            "avg_terminal_reward",
            "miners_win_rate",
            "avg_game_length",
            "avg_gold_reaches",
            "avg_public_stone_reaches",
            "avg_revealed_goals",
        ],
    ),
    (
        "behavior",
        "Behavior",
        [
            "play_path_rate",
            "discard_rate",
            "repair_rate",
            "rockfall_rate",
            "sabotage_rate",
            "map_goal_rate",
            "map_play_when_available_rate",
        ],
    ),
    (
        "beliefs",
        "Beliefs",
        [
            "goal_belief_loss",
            "role_belief_loss",
            "avg_private_goal_knowledge_count",
            "avg_min_distance_to_goal",
            "avg_frontier_empty_cells",
            "avg_reachable_tiles",
            "avg_legal_actions",
        ],
    ),
]


def load_metrics_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Load training metrics from a JSONL file."""

    metrics_path = Path(path)
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{metrics_path}:{line_number} is not a JSON object")
        rows.append({str(key): _clean_value(raw_value) for key, raw_value in value.items()})

    if not rows:
        raise ValueError(f"{metrics_path} does not contain any metric rows")
    return rows


def render_metrics_dashboard(
    rows: Sequence[MetricRow],
    *,
    title: str = "Saboter Metrics Dashboard",
) -> str:
    """Render a self-contained HTML dashboard for numeric metric rows."""

    if not rows:
        raise ValueError("metrics dashboard requires at least one row")

    clean_rows = [{str(key): _clean_value(value) for key, value in row.items()} for row in rows]
    numeric_metrics = _numeric_metrics(clean_rows)
    if not numeric_metrics:
        raise ValueError("metrics dashboard requires at least one numeric metric")

    groups = _grouped_metrics(numeric_metrics)
    eval_metrics = [metric for metric in numeric_metrics if metric.startswith("eval_")]
    presets = [
        {"id": preset_id, "label": label, "metrics": [metric for metric in metrics if metric in numeric_metrics]}
        for preset_id, label, metrics in PRESET_DEFINITIONS
    ]
    if eval_metrics:
        presets.append({"id": "eval", "label": "Eval", "metrics": eval_metrics})
    presets = [preset for preset in presets if preset["metrics"]]

    payload = {
        "title": title,
        "rows": clean_rows,
        "numericMetrics": numeric_metrics,
        "groups": groups,
        "presets": presets,
        "defaultSelected": [
            metric for metric in DEFAULT_SELECTED_METRICS if metric in numeric_metrics
        ]
        or numeric_metrics[: min(4, len(numeric_metrics))],
        "meta": {
            "rowCount": len(clean_rows),
            "metricCount": len(numeric_metrics),
            "xKey": "iteration"
            if all(_is_number(row.get("iteration")) for row in clean_rows)
            else "row",
            "firstIteration": clean_rows[0].get("iteration"),
            "lastIteration": clean_rows[-1].get("iteration"),
            "checkpointCount": sum(1 for row in clean_rows if isinstance(row.get("checkpoint"), str)),
            "runLabel": title,
        },
    }

    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    title_html = html.escape(title, quote=True)

    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__SABOTER_METRICS_TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5efe4;
      --bg-accent: #efe3d2;
      --panel: rgba(255, 252, 247, 0.88);
      --panel-strong: rgba(255, 249, 241, 0.96);
      --panel-alt: rgba(246, 237, 223, 0.82);
      --text: #1f2d2f;
      --muted: #6c7571;
      --line: rgba(88, 92, 84, 0.18);
      --line-strong: rgba(88, 92, 84, 0.32);
      --accent: #0f766e;
      --accent-soft: rgba(15, 118, 110, 0.14);
      --accent-warm: #c26d3d;
      --accent-cool: #245c7f;
      --danger: #9d3d2c;
      --shadow: 0 24px 64px rgba(50, 42, 28, 0.12);
    }

    * { box-sizing: border-box; }

    html, body {
      min-height: 100%;
      margin: 0;
    }

    body {
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 26%),
        radial-gradient(circle at top right, rgba(194, 109, 61, 0.14), transparent 24%),
        linear-gradient(180deg, #f8f3ea 0%, #efe5d7 100%);
      color: var(--text);
      font: 15px/1.45 "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      font-feature-settings: "tnum" 1;
    }

    button,
    input {
      font: inherit;
    }

    .app-shell {
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      min-height: 100vh;
    }

    .sidebar {
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      padding: 24px 20px;
      border-right: 1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255, 251, 246, 0.96), rgba(246, 236, 219, 0.92));
      backdrop-filter: blur(18px);
    }

    .sidebar-inner {
      display: grid;
      gap: 18px;
      align-content: start;
    }

    .kicker {
      margin: 0;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 11px;
      font-weight: 700;
    }

    .sidebar h1,
    .hero h1,
    .panel-head h2,
    .section-title {
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .sidebar h1 {
      margin: 4px 0 6px;
      font-size: 30px;
      line-height: 1.05;
    }

    .subtle {
      margin: 0;
      color: var(--muted);
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .summary-card,
    .control-panel,
    .hint-card,
    .panel,
    .metric-card {
      border: 1px solid var(--line);
      border-radius: 20px;
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }

    .summary-card {
      padding: 12px 14px;
    }

    .summary-label {
      margin: 0 0 2px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .summary-value {
      margin: 0;
      font-size: 22px;
      font-weight: 700;
    }

    .control-panel {
      padding: 16px;
      display: grid;
      gap: 14px;
    }

    .control-title {
      margin: 0;
      font-size: 14px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }

    .field {
      display: grid;
      gap: 6px;
    }

    .field label,
    .scrub label {
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }

    .field input[type="search"],
    .field input[type="text"] {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.72);
      padding: 10px 14px;
      color: var(--text);
    }

    .field input[type="range"],
    .scrub input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }

    .toggle {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--text);
      font-size: 14px;
    }

    .toggle input {
      inline-size: 18px;
      block-size: 18px;
    }

    .preset-row,
    .chip-list,
    .pill-row,
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .preset-button,
    .metric-chip,
    .pill,
    .ghost-button {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.68);
      color: var(--text);
      padding: 7px 12px;
    }

    .preset-button,
    .metric-chip,
    .ghost-button,
    .metric-card {
      cursor: pointer;
      transition: transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease, background 140ms ease;
    }

    .preset-button:hover,
    .metric-chip:hover,
    .ghost-button:hover,
    .metric-card:hover {
      transform: translateY(-1px);
      border-color: var(--line-strong);
      box-shadow: 0 16px 34px rgba(50, 42, 28, 0.12);
    }

    .preset-button.active,
    .metric-chip.active {
      border-color: rgba(15, 118, 110, 0.4);
      background: var(--accent-soft);
      color: var(--accent);
    }

    .metric-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding-right: 10px;
    }

    .swatch {
      inline-size: 10px;
      block-size: 10px;
      border-radius: 999px;
      background: currentColor;
      flex: 0 0 auto;
    }

    .ghost-button {
      justify-self: start;
    }

    .hint-card {
      padding: 14px 16px;
      background: var(--panel-alt);
    }

    .hint-card p {
      margin: 0;
      color: var(--muted);
    }

    .main {
      min-width: 0;
      display: grid;
      gap: 16px;
      padding: 22px 22px 28px;
    }

    .hero,
    .panel {
      padding: 18px 20px;
    }

    .hero {
      display: grid;
      gap: 12px;
      border: 1px solid rgba(15, 118, 110, 0.14);
      border-radius: 28px;
      background:
        linear-gradient(135deg, rgba(255, 248, 238, 0.95), rgba(247, 240, 230, 0.92)),
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 38%);
      box-shadow: var(--shadow);
    }

    .hero h1 {
      margin: 4px 0 4px;
      font-size: 38px;
      line-height: 1;
    }

    .hero p {
      margin: 0;
    }

    .panel {
      display: grid;
      gap: 16px;
    }

    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }

    .panel-head h2,
    .section-title {
      margin: 0 0 4px;
      font-size: 26px;
      line-height: 1.05;
    }

    .panel-head p {
      margin: 0;
      color: var(--muted);
    }

    .chart-wrap {
      position: relative;
      min-height: 390px;
      border-radius: 22px;
      overflow: hidden;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.5), rgba(249, 242, 232, 0.72));
      border: 1px solid var(--line);
    }

    #detailChart {
      display: block;
      inline-size: 100%;
      block-size: 390px;
    }

    .chart-tooltip {
      position: absolute;
      min-inline-size: 220px;
      max-inline-size: min(340px, calc(100% - 24px));
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid rgba(15, 118, 110, 0.22);
      background: rgba(255, 251, 245, 0.97);
      box-shadow: 0 18px 40px rgba(50, 42, 28, 0.16);
      pointer-events: none;
      transform: translate(-50%, calc(-100% - 14px));
      backdrop-filter: blur(10px);
      z-index: 3;
    }

    .chart-tooltip.hidden {
      display: none;
    }

    .chart-tooltip-title {
      margin: 0 0 8px;
      font-weight: 700;
    }

    .chart-tooltip-list {
      display: grid;
      gap: 6px;
    }

    .chart-tooltip-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
    }

    .chart-tooltip-row strong {
      color: var(--text);
      font-weight: 600;
    }

    .scrub {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
    }

    .scrub-label {
      color: var(--muted);
      text-align: right;
      white-space: nowrap;
    }

    .snapshot-groups {
      display: grid;
      gap: 18px;
    }

    .snapshot-group {
      display: grid;
      gap: 10px;
    }

    .snapshot-title {
      margin: 0;
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }

    .snapshot-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
    }

    .snapshot-item {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.64);
      padding: 10px 12px;
    }

    .snapshot-item.selected {
      border-color: rgba(15, 118, 110, 0.34);
      background: var(--accent-soft);
    }

    .snapshot-name {
      margin: 0 0 3px;
      color: var(--muted);
      font-size: 13px;
    }

    .snapshot-value {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }

    .metric-sections {
      display: grid;
      gap: 22px;
    }

    .metric-section {
      display: grid;
      gap: 12px;
    }

    .metric-section-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
    }

    .metric-section-header p {
      margin: 0;
      color: var(--muted);
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }

    .metric-card {
      display: grid;
      gap: 12px;
      padding: 14px;
      text-align: left;
      color: inherit;
    }

    .metric-card.selected {
      border-color: rgba(15, 118, 110, 0.38);
      background:
        linear-gradient(180deg, rgba(255, 254, 251, 0.94), rgba(235, 247, 245, 0.92));
    }

    .metric-card-top,
    .metric-card-bottom,
    .legend-item {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
    }

    .metric-card-name {
      margin: 0;
      font-size: 16px;
      font-weight: 700;
      line-height: 1.15;
    }

    .metric-card-key {
      display: inline-block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }

    .metric-card-value {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      white-space: nowrap;
    }

    .metric-card-bottom {
      color: var(--muted);
      font-size: 13px;
    }

    .legend-item {
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid var(--line);
      color: var(--muted);
    }

    .legend-item strong {
      color: var(--text);
      font-weight: 600;
    }

    .empty-state {
      display: grid;
      place-items: center;
      min-height: 180px;
      border: 1px dashed var(--line-strong);
      border-radius: 20px;
      color: var(--muted);
      text-align: center;
      padding: 24px;
      background: rgba(255, 255, 255, 0.38);
    }

    .axis-label,
    .axis-tick,
    .grid-label,
    .chart-caption,
    .checkpoint-label {
      fill: var(--muted);
      font: 12px "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
    }

    .axis-line,
    .grid-line {
      stroke: rgba(88, 92, 84, 0.14);
      stroke-width: 1;
    }

    .grid-line.zero {
      stroke: rgba(15, 118, 110, 0.22);
      stroke-dasharray: 4 6;
    }

    .series-line {
      fill: none;
      stroke-width: 3;
      stroke-linejoin: round;
      stroke-linecap: round;
    }

    .series-area {
      opacity: 0.08;
    }

    .checkpoint-line {
      stroke: rgba(194, 109, 61, 0.32);
      stroke-width: 1;
      stroke-dasharray: 4 8;
    }

    .hover-line {
      stroke: rgba(31, 45, 47, 0.42);
      stroke-width: 1.5;
      stroke-dasharray: 5 6;
    }

    .hover-dot {
      stroke: #fffdf9;
      stroke-width: 2;
    }

    .sparkline {
      inline-size: 100%;
      block-size: 64px;
      display: block;
    }

    .sparkline .track {
      fill: none;
      stroke: rgba(88, 92, 84, 0.12);
      stroke-width: 1;
    }

    .sparkline .line {
      fill: none;
      stroke-width: 2.5;
      stroke-linecap: round;
      stroke-linejoin: round;
    }

    @media (max-width: 1100px) {
      .app-shell {
        grid-template-columns: 1fr;
      }

      .sidebar {
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
    }

    @media (max-width: 720px) {
      .main {
        padding: 16px;
      }

      .hero h1,
      .sidebar h1 {
        font-size: 28px;
      }

      .panel-head,
      .scrub {
        grid-template-columns: 1fr;
        display: grid;
      }

      .scrub-label {
        text-align: left;
      }
    }
  </style>
</head>
<body>
  <script id="metrics-data" type="application/json">__SABOTER_METRICS_PAYLOAD__</script>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="sidebar-inner">
        <div>
          <p class="kicker">Saboter dashboard</p>
          <h1 id="sidebarTitle"></h1>
          <p id="sidebarSubtitle" class="subtle"></p>
        </div>
        <div id="summaryStats" class="summary-grid"></div>
        <section class="control-panel">
          <p class="control-title">Controls</p>
          <div class="field">
            <label for="metricSearch">Search metrics</label>
            <input id="metricSearch" type="search" placeholder="reward, entropy, eval...">
          </div>
          <div class="field">
            <label for="smoothingRange">Smoothing window <span id="smoothingValue"></span></label>
            <input id="smoothingRange" type="range" min="1" max="15" step="2" value="1">
          </div>
          <label class="toggle" for="normalizeToggle">
            <input id="normalizeToggle" type="checkbox">
            <span>Normalize when comparing metrics</span>
          </label>
          <div class="field">
            <label>Quick presets</label>
            <div id="presetButtons" class="preset-row"></div>
          </div>
          <button id="clearSelection" class="ghost-button" type="button">Clear selection</button>
        </section>
        <section>
          <p class="control-title">Selected metrics</p>
          <div id="selectedMetrics" class="chip-list"></div>
        </section>
        <section class="hint-card">
          <p>Click a metric card to add or remove it from the main chart. Shift-click a card to solo it.</p>
        </section>
      </div>
    </aside>
    <main class="main">
      <section class="hero">
        <div>
          <p class="kicker">Training timeline</p>
          <h1 id="heroTitle"></h1>
          <p id="heroDescription" class="subtle"></p>
        </div>
        <div id="heroBadges" class="pill-row"></div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Trend explorer</h2>
            <p id="chartCaption"></p>
          </div>
          <div id="chartLegend" class="legend"></div>
        </div>
        <div class="chart-wrap" id="chartWrap">
          <svg id="detailChart" viewBox="0 0 1000 390" preserveAspectRatio="none"></svg>
          <div id="chartTooltip" class="chart-tooltip hidden"></div>
        </div>
        <div class="scrub">
          <label for="rowSlider">Snapshot</label>
          <input id="rowSlider" type="range" min="0" step="1">
          <div id="rowLabel" class="scrub-label"></div>
        </div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Snapshot inspector</h2>
            <p id="snapshotCaption"></p>
          </div>
          <div id="snapshotMeta" class="pill-row"></div>
        </div>
        <div id="snapshotGroups" class="snapshot-groups"></div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>All metrics</h2>
            <p id="metricsCaption"></p>
          </div>
        </div>
        <div id="metricSections" class="metric-sections"></div>
      </section>
    </main>
  </div>
  <script>
    const payload = JSON.parse(document.getElementById("metrics-data").textContent);
    const rows = payload.rows;
    const metrics = payload.numericMetrics;
    const groups = payload.groups;
    const presets = payload.presets;
    const xValues = rows.map((row, index) => isFiniteNumber(row.iteration) ? row.iteration : index + 1);
    const colorPalette = [
      "#0f766e",
      "#c26d3d",
      "#245c7f",
      "#8a3b5b",
      "#5b8f2c",
      "#7e4b98",
      "#a6472c",
      "#0a5b57",
      "#496b9a",
      "#9f8a1b",
      "#8c5142",
      "#34724f"
    ];

    const state = {
      selectedMetrics: new Set(payload.defaultSelected),
      search: "",
      smoothingWindow: 1,
      normalize: payload.defaultSelected.length > 1,
      highlightedIndex: Math.max(0, rows.length - 1),
    };

    const refs = {
      sidebarTitle: document.getElementById("sidebarTitle"),
      sidebarSubtitle: document.getElementById("sidebarSubtitle"),
      summaryStats: document.getElementById("summaryStats"),
      metricSearch: document.getElementById("metricSearch"),
      smoothingRange: document.getElementById("smoothingRange"),
      smoothingValue: document.getElementById("smoothingValue"),
      normalizeToggle: document.getElementById("normalizeToggle"),
      presetButtons: document.getElementById("presetButtons"),
      clearSelection: document.getElementById("clearSelection"),
      selectedMetrics: document.getElementById("selectedMetrics"),
      heroTitle: document.getElementById("heroTitle"),
      heroDescription: document.getElementById("heroDescription"),
      heroBadges: document.getElementById("heroBadges"),
      chartCaption: document.getElementById("chartCaption"),
      chartLegend: document.getElementById("chartLegend"),
      detailChart: document.getElementById("detailChart"),
      chartWrap: document.getElementById("chartWrap"),
      chartTooltip: document.getElementById("chartTooltip"),
      rowSlider: document.getElementById("rowSlider"),
      rowLabel: document.getElementById("rowLabel"),
      snapshotCaption: document.getElementById("snapshotCaption"),
      snapshotMeta: document.getElementById("snapshotMeta"),
      snapshotGroups: document.getElementById("snapshotGroups"),
      metricsCaption: document.getElementById("metricsCaption"),
      metricSections: document.getElementById("metricSections"),
    };

    const statsByMetric = Object.fromEntries(metrics.map((metric) => [metric, computeMetricStats(metric)]));
    let chartModel = null;

    setup();

    function setup() {
      refs.sidebarTitle.textContent = payload.title;
      refs.heroTitle.textContent = payload.title;
      refs.sidebarSubtitle.textContent = `${payload.meta.rowCount} snapshots · ${payload.meta.metricCount} numeric metrics`;
      refs.heroDescription.textContent = describeTimeline();
      refs.rowSlider.max = String(Math.max(0, rows.length - 1));
      refs.rowSlider.value = String(state.highlightedIndex);
      refs.normalizeToggle.checked = state.normalize;

      refs.metricSearch.addEventListener("input", () => {
        state.search = refs.metricSearch.value.trim().toLowerCase();
        renderMetricSections();
        renderSnapshotInspector();
      });
      refs.smoothingRange.addEventListener("input", () => {
        state.smoothingWindow = Number(refs.smoothingRange.value);
        refs.smoothingValue.textContent = formatWindow(state.smoothingWindow);
        renderTrendExplorer();
      });
      refs.normalizeToggle.addEventListener("change", () => {
        state.normalize = refs.normalizeToggle.checked;
        renderTrendExplorer();
      });
      refs.rowSlider.addEventListener("input", () => {
        updateHighlight(Number(refs.rowSlider.value));
      });
      refs.clearSelection.addEventListener("click", () => {
        state.selectedMetrics = new Set();
        renderPresets();
        renderSelectedMetrics();
        renderTrendExplorer();
        renderMetricSections();
        renderSnapshotInspector();
      });

      window.addEventListener("resize", () => {
        if (chartModel) {
          updateChartHighlight();
        }
      });

      renderSummary();
      renderPresets();
      renderSelectedMetrics();
      renderTrendExplorer();
      renderSnapshotInspector();
      renderMetricSections();
    }

    function renderSummary() {
      refs.summaryStats.innerHTML = [
        summaryCard("Rows", formatCount(payload.meta.rowCount)),
        summaryCard("Metrics", formatCount(payload.meta.metricCount)),
        summaryCard(
          payload.meta.xKey === "iteration" ? "Iterations" : "Range",
          describeIterationSpan()
        ),
        summaryCard("Checkpoints", formatCount(payload.meta.checkpointCount)),
      ].join("");

      refs.heroBadges.innerHTML = [
        pill(`${payload.meta.rowCount} snapshots`),
        pill(`${payload.meta.metricCount} numeric metrics`),
        pill(payload.meta.xKey === "iteration" ? `iterations ${describeIterationSpan()}` : "indexed by row"),
        pill(`${payload.meta.checkpointCount} checkpoint markers`),
      ].join("");
    }

    function renderPresets() {
      refs.presetButtons.innerHTML = presets
        .map((preset) => {
          const active = sameMetricSet([...state.selectedMetrics], preset.metrics) ? " active" : "";
          return `<button class="preset-button${active}" data-preset="${escapeHtml(preset.id)}" type="button">${escapeHtml(preset.label)}</button>`;
        })
        .join("");

      refs.presetButtons.querySelectorAll("[data-preset]").forEach((button) => {
        button.addEventListener("click", () => {
          const preset = presets.find((entry) => entry.id === button.dataset.preset);
          if (!preset) {
            return;
          }
          state.selectedMetrics = new Set(preset.metrics);
          state.normalize = preset.metrics.length > 1;
          refs.normalizeToggle.checked = state.normalize;
          renderPresets();
          renderSelectedMetrics();
          renderTrendExplorer();
          renderMetricSections();
          renderSnapshotInspector();
        });
      });
    }

    function renderSelectedMetrics() {
      const selected = [...state.selectedMetrics].filter((metric) => metrics.includes(metric));
      if (!selected.length) {
        refs.selectedMetrics.innerHTML = `<div class="subtle">No metrics selected yet.</div>`;
      } else {
        refs.selectedMetrics.innerHTML = selected
          .map((metric) => {
            const color = colorForMetric(metric);
            return `
              <button class="metric-chip active" type="button" data-remove-metric="${escapeHtml(metric)}" style="color: ${escapeHtml(color)};">
                <span class="swatch"></span>
                <span>${escapeHtml(humanizeMetric(metric))}</span>
              </button>
            `;
          })
          .join("");

        refs.selectedMetrics.querySelectorAll("[data-remove-metric]").forEach((button) => {
          button.addEventListener("click", () => {
            const metric = button.dataset.removeMetric;
            if (!metric) {
              return;
            }
            state.selectedMetrics.delete(metric);
            renderPresets();
            renderSelectedMetrics();
            renderTrendExplorer();
            renderMetricSections();
            renderSnapshotInspector();
          });
        });
      }

    }

    function renderTrendExplorer() {
      const selected = [...state.selectedMetrics].filter((metric) => metrics.includes(metric));
      refs.smoothingValue.textContent = formatWindow(state.smoothingWindow);

      if (!selected.length) {
        refs.chartLegend.innerHTML = "";
        refs.chartCaption.textContent = "Select one or more metrics to compare their trajectories.";
        refs.detailChart.innerHTML = `
          <foreignObject x="0" y="0" width="1000" height="390">
            <div xmlns="http://www.w3.org/1999/xhtml" class="empty-state">
              Choose a metric card below to start plotting its training history.
            </div>
          </foreignObject>
        `;
        chartModel = null;
        refs.chartTooltip.classList.add("hidden");
        updateRowLabel();
        return;
      }

      const useNormalizedScale = state.normalize && selected.length > 1;
      const series = selected.map((metric) => buildSeries(metric, useNormalizedScale));
      const plottedValues = series.flatMap((entry) => entry.points.map((point) => point.plotValue));
      const yExtent = niceExtent(plottedValues);
      const xExtent = niceExtent(xValues);
      const plot = { left: 72, top: 18, right: 976, bottom: 342 };
      const plotWidth = plot.right - plot.left;
      const plotHeight = plot.bottom - plot.top;
      const yTicks = niceTicks(yExtent.min, yExtent.max, 5);
      const xTicks = niceTicks(xExtent.min, xExtent.max, 6);
      const checkpoints = rows
        .map((row, index) => ({ index, checkpoint: row.checkpoint }))
        .filter((entry) => typeof entry.checkpoint === "string");

      const gridMarkup = yTicks
        .map((tick) => {
          const y = scaleY(tick, yExtent.min, yExtent.max, plot.top, plotHeight);
          const zeroClass = Math.abs(tick) < 1e-9 ? " zero" : "";
          return `
            <g>
              <line class="grid-line${zeroClass}" x1="${plot.left}" y1="${y}" x2="${plot.right}" y2="${y}"></line>
              <text class="grid-label" x="${plot.left - 10}" y="${y + 4}" text-anchor="end">${escapeHtml(formatAxisValue(tick, useNormalizedScale))}</text>
            </g>
          `;
        })
        .join("");

      const xAxisMarkup = xTicks
        .map((tick) => {
          const x = scaleX(tick, xExtent.min, xExtent.max, plot.left, plotWidth);
          return `
            <g>
              <line class="axis-line" x1="${x}" y1="${plot.bottom}" x2="${x}" y2="${plot.bottom + 6}"></line>
              <text class="axis-tick" x="${x}" y="${plot.bottom + 22}" text-anchor="middle">${escapeHtml(formatCount(Math.round(tick)))}</text>
            </g>
          `;
        })
        .join("");

      const checkpointMarkup = checkpoints
        .map((entry) => {
          const xValue = xValues[entry.index];
          const x = scaleX(xValue, xExtent.min, xExtent.max, plot.left, plotWidth);
          const label = basename(entry.checkpoint);
          return `
            <g>
              <line class="checkpoint-line" x1="${x}" y1="${plot.top}" x2="${x}" y2="${plot.bottom}"></line>
              <text class="checkpoint-label" x="${x + 6}" y="${plot.top + 14}">${escapeHtml(label.replace(".pt", ""))}</text>
            </g>
          `;
        })
        .join("");

      const areaMarkup = selected.length === 1
        ? `
          <path
            class="series-area"
            d="${escapeHtml(buildAreaPath(series[0].points, plot, xExtent, yExtent))}"
            fill="${escapeHtml(series[0].color)}"></path>
        `
        : "";

      const linesMarkup = series
        .map((entry) => `
          <path class="series-line" d="${escapeHtml(buildLinePath(entry.points, plot, xExtent, yExtent))}" stroke="${escapeHtml(entry.color)}"></path>
        `)
        .join("");

      refs.detailChart.innerHTML = `
        ${gridMarkup}
        ${checkpointMarkup}
        <line class="axis-line" x1="${plot.left}" y1="${plot.bottom}" x2="${plot.right}" y2="${plot.bottom}"></line>
        <line class="axis-line" x1="${plot.left}" y1="${plot.top}" x2="${plot.left}" y2="${plot.bottom}"></line>
        ${xAxisMarkup}
        ${areaMarkup}
        ${linesMarkup}
        <text class="axis-label" x="${plot.left}" y="${plot.top - 2}">${escapeHtml(useNormalizedScale ? "normalized value" : "metric value")}</text>
        <text class="axis-label" x="${plot.right}" y="${plot.bottom + 34}" text-anchor="end">${escapeHtml(payload.meta.xKey)}</text>
        <g id="hoverLayer"></g>
        <rect id="chartHotzone" x="${plot.left}" y="${plot.top}" width="${plotWidth}" height="${plotHeight}" fill="transparent"></rect>
      `;

      refs.chartLegend.innerHTML = series
        .map((entry) => {
          const latest = entry.points.length ? entry.points[entry.points.length - 1].rawValue : null;
          return `
            <div class="legend-item" style="color: ${escapeHtml(entry.color)};">
              <span class="swatch"></span>
              <strong>${escapeHtml(humanizeMetric(entry.metric))}</strong>
              <span>${escapeHtml(formatMetricValue(entry.metric, latest))}</span>
            </div>
          `;
        })
        .join("");

      refs.chartCaption.textContent = chartDescription(selected, useNormalizedScale);

      chartModel = {
        plot,
        xExtent,
        yExtent,
        useNormalizedScale,
        selected,
        series,
      };

      const hotzone = document.getElementById("chartHotzone");
      hotzone.addEventListener("mousemove", onChartHover);
      hotzone.addEventListener("mouseenter", () => refs.chartTooltip.classList.remove("hidden"));
      hotzone.addEventListener("mouseleave", () => refs.chartTooltip.classList.add("hidden"));

      updateChartHighlight();
      updateRowLabel();
      renderSnapshotInspector();
      renderPresets();
    }

    function renderSnapshotInspector() {
      const row = rows[state.highlightedIndex] || {};
      const visibleGroups = groupedVisibleMetrics();
      refs.snapshotCaption.textContent = `Inspecting snapshot ${state.highlightedIndex + 1} of ${rows.length}. Hover the chart or scrub the slider for exact values.`;

      const meta = [
        row.iteration !== undefined ? pill(metaPill("iteration", row.iteration)) : "",
        row.updates !== undefined ? pill(metaPill("updates", row.updates)) : "",
        row.games !== undefined ? pill(metaPill("games", row.games)) : "",
        row.transitions !== undefined ? pill(metaPill("transitions", row.transitions)) : "",
        row.checkpoint ? pill(basename(row.checkpoint)) : "",
      ].join("");
      refs.snapshotMeta.innerHTML = meta;

      if (!visibleGroups.length) {
        refs.snapshotGroups.innerHTML = `<div class="empty-state">No metrics match the current search.</div>`;
        return;
      }

      refs.snapshotGroups.innerHTML = visibleGroups
        .map((group) => {
          const items = group.metrics
            .map((metric) => {
              const selected = state.selectedMetrics.has(metric) ? " selected" : "";
              return `
                <div class="snapshot-item${selected}">
                  <p class="snapshot-name">${escapeHtml(humanizeMetric(metric))}</p>
                  <p class="snapshot-value">${escapeHtml(formatMetricValue(metric, row[metric]))}</p>
                </div>
              `;
            })
            .join("");
          return `
            <section class="snapshot-group">
              <h3 class="snapshot-title">${escapeHtml(group.label)}</h3>
              <div class="snapshot-grid">${items}</div>
            </section>
          `;
        })
        .join("");
    }

    function renderMetricSections() {
      refs.metricsCaption.textContent = describeMetricFilter();
      const visibleGroups = groupedVisibleMetrics();
      if (!visibleGroups.length) {
        refs.metricSections.innerHTML = `<div class="empty-state">No metrics match the current search.</div>`;
        return;
      }

      refs.metricSections.innerHTML = visibleGroups
        .map((group) => {
          const cards = group.metrics.map((metric) => renderMetricCard(metric)).join("");
          return `
            <section class="metric-section">
              <div class="metric-section-header">
                <div>
                  <h3 class="section-title">${escapeHtml(group.label)}</h3>
                  <p>${escapeHtml(group.metrics.length === 1 ? "1 metric" : `${group.metrics.length} metrics`)}</p>
                </div>
              </div>
              <div class="metric-grid">${cards}</div>
            </section>
          `;
        })
        .join("");

      refs.metricSections.querySelectorAll("[data-metric-card]").forEach((button) => {
        button.addEventListener("click", (event) => {
          const metric = button.dataset.metricCard;
          if (!metric) {
            return;
          }
          if (event.shiftKey) {
            state.selectedMetrics = new Set([metric]);
            state.normalize = false;
            refs.normalizeToggle.checked = false;
          } else if (state.selectedMetrics.has(metric)) {
            state.selectedMetrics.delete(metric);
          } else {
            state.selectedMetrics.add(metric);
          }
          renderPresets();
          renderSelectedMetrics();
          renderTrendExplorer();
          renderMetricSections();
          renderSnapshotInspector();
        });
      });
    }

    function renderMetricCard(metric) {
      const stats = statsByMetric[metric];
      const selected = state.selectedMetrics.has(metric) ? " selected" : "";
      const color = colorForMetric(metric);
      const latest = stats.lastValue;
      const delta = stats.delta;
      return `
        <button class="metric-card${selected}" type="button" data-metric-card="${escapeHtml(metric)}">
          <div class="metric-card-top">
            <div>
              <p class="metric-card-name">${escapeHtml(humanizeMetric(metric))}</p>
              <span class="metric-card-key">${escapeHtml(metric)}</span>
            </div>
            <p class="metric-card-value">${escapeHtml(formatMetricValue(metric, latest))}</p>
          </div>
          ${sparklineSvg(metric, color)}
          <div class="metric-card-bottom">
            <span>delta ${escapeHtml(formatSignedMetricValue(metric, delta))}</span>
            <span>${escapeHtml(formatCoverage(stats.validCount, rows.length))}</span>
          </div>
        </button>
      `;
    }

    function updateHighlight(index) {
      state.highlightedIndex = clamp(Math.round(index), 0, Math.max(0, rows.length - 1));
      refs.rowSlider.value = String(state.highlightedIndex);
      updateRowLabel();
      renderSnapshotInspector();
      updateChartHighlight();
    }

    function updateRowLabel() {
      const row = rows[state.highlightedIndex] || {};
      const xLabel = payload.meta.xKey === "iteration" && isFiniteNumber(row.iteration)
        ? `iteration ${formatCount(row.iteration)}`
        : `row ${formatCount(state.highlightedIndex + 1)}`;
      refs.rowLabel.textContent = row.checkpoint
        ? `${xLabel} · ${basename(row.checkpoint)}`
        : xLabel;
    }

    function updateChartHighlight() {
      if (!chartModel) {
        return;
      }

      const hoverLayer = document.getElementById("hoverLayer");
      if (!hoverLayer) {
        return;
      }

      const row = rows[state.highlightedIndex] || {};
      const xValue = xValues[state.highlightedIndex];
      const x = scaleX(
        xValue,
        chartModel.xExtent.min,
        chartModel.xExtent.max,
        chartModel.plot.left,
        chartModel.plot.right - chartModel.plot.left
      );

      const circles = chartModel.series
        .map((entry) => {
          const point = entry.indexMap.get(state.highlightedIndex);
          if (!point) {
            return "";
          }
          const y = scaleY(
            point.plotValue,
            chartModel.yExtent.min,
            chartModel.yExtent.max,
            chartModel.plot.top,
            chartModel.plot.bottom - chartModel.plot.top
          );
          return `<circle class="hover-dot" cx="${x}" cy="${y}" r="5" fill="${escapeHtml(entry.color)}"></circle>`;
        })
        .join("");

      hoverLayer.innerHTML = `
        <line class="hover-line" x1="${x}" y1="${chartModel.plot.top}" x2="${x}" y2="${chartModel.plot.bottom}"></line>
        ${circles}
      `;

      const tooltipRows = chartModel.series
        .map((entry) => {
          const point = entry.indexMap.get(state.highlightedIndex);
          if (!point) {
            return "";
          }
          return `
            <div class="chart-tooltip-row" style="color: ${escapeHtml(entry.color)};">
              <span><span class="swatch"></span> ${escapeHtml(humanizeMetric(entry.metric))}</span>
              <strong>${escapeHtml(formatMetricValue(entry.metric, point.rawValue))}</strong>
            </div>
          `;
        })
        .join("");

      refs.chartTooltip.innerHTML = `
        <p class="chart-tooltip-title">${escapeHtml(tooltipTitle(row, state.highlightedIndex))}</p>
        <div class="chart-tooltip-list">${tooltipRows}</div>
      `;

      positionTooltip(x);
    }

    function onChartHover(event) {
      if (!chartModel) {
        return;
      }
      const rect = event.currentTarget.getBoundingClientRect();
      const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
      const targetX = chartModel.xExtent.min + ratio * (chartModel.xExtent.max - chartModel.xExtent.min);
      let bestIndex = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let index = 0; index < xValues.length; index += 1) {
        const distance = Math.abs(xValues[index] - targetX);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestIndex = index;
        }
      }
      updateHighlight(bestIndex);
      refs.chartTooltip.classList.remove("hidden");
    }

    function positionTooltip(svgX) {
      const wrap = refs.chartWrap.getBoundingClientRect();
      const svg = refs.detailChart.getBoundingClientRect();
      if (!wrap.width || !svg.width) {
        return;
      }
      const px = ((svgX / 1000) * svg.width) + (svg.left - wrap.left);
      const tooltipWidth = refs.chartTooltip.offsetWidth || 240;
      const clampedX = clamp(px, tooltipWidth / 2 + 10, wrap.width - tooltipWidth / 2 - 10);
      refs.chartTooltip.style.left = `${clampedX}px`;
      refs.chartTooltip.style.top = `${Math.max(28, svg.top - wrap.top + 18)}px`;
    }

    function buildSeries(metric, normalize) {
      const rawValues = rows.map((row) => isFiniteNumber(row[metric]) ? Number(row[metric]) : null);
      const smoothed = smoothValues(rawValues, state.smoothingWindow);
      const validSmoothed = smoothed.filter((value) => value !== null);
      const localExtent = normalize ? extent(validSmoothed) : null;
      const indexMap = new Map();
      const points = [];

      for (let index = 0; index < rawValues.length; index += 1) {
        const rawValue = rawValues[index];
        const plotSource = smoothed[index];
        if (rawValue === null || plotSource === null) {
          continue;
        }
        let plotValue = plotSource;
        if (normalize) {
          if (localExtent && Math.abs(localExtent.max - localExtent.min) > 1e-12) {
            plotValue = (plotSource - localExtent.min) / (localExtent.max - localExtent.min);
          } else {
            plotValue = 0.5;
          }
        }
        const point = { index, xValue: xValues[index], rawValue, plotValue };
        points.push(point);
        indexMap.set(index, point);
      }

      return {
        metric,
        color: colorForMetric(metric),
        points,
        indexMap,
      };
    }

    function computeMetricStats(metric) {
      const values = rows
        .map((row) => row[metric])
        .filter((value) => isFiniteNumber(value))
        .map((value) => Number(value));
      const firstValue = values.length ? values[0] : null;
      const lastValue = values.length ? values[values.length - 1] : null;
      const minValue = values.length ? Math.min(...values) : null;
      const maxValue = values.length ? Math.max(...values) : null;
      return {
        firstValue,
        lastValue,
        minValue,
        maxValue,
        delta: firstValue === null || lastValue === null ? null : lastValue - firstValue,
        validCount: values.length,
      };
    }

    function sparklineSvg(metric, color) {
      const values = rows.map((row) => isFiniteNumber(row[metric]) ? Number(row[metric]) : null);
      const points = values
        .map((value, index) => value === null ? null : { index, value })
        .filter(Boolean);
      if (!points.length) {
        return `<svg class="sparkline" viewBox="0 0 220 64" preserveAspectRatio="none"></svg>`;
      }
      const xExtent = { min: 0, max: Math.max(1, rows.length - 1) };
      const yExtent = niceExtent(points.map((point) => point.value));
      const plot = { left: 4, top: 6, right: 216, bottom: 58 };
      const baselineY = scaleY(yExtent.min, yExtent.min, yExtent.max, plot.top, plot.bottom - plot.top);
      return `
        <svg class="sparkline" viewBox="0 0 220 64" preserveAspectRatio="none" aria-hidden="true">
          <line class="track" x1="${plot.left}" y1="${baselineY}" x2="${plot.right}" y2="${baselineY}"></line>
          <path class="line" d="${escapeHtml(buildLinePath(points, plot, xExtent, yExtent))}" stroke="${escapeHtml(color)}"></path>
        </svg>
      `;
    }

    function buildLinePath(points, plot, xExtent, yExtent) {
      if (!points.length) {
        return "";
      }
      let path = "";
      let previousIndex = null;
      for (const point of points) {
        const xValue = point.xValue ?? point.index;
        const x = scaleX(xValue, xExtent.min, xExtent.max, plot.left, plot.right - plot.left);
        const y = scaleY(point.plotValue ?? point.value, yExtent.min, yExtent.max, plot.top, plot.bottom - plot.top);
        if (previousIndex !== null && point.index !== previousIndex + 1) {
          path += ` M ${x.toFixed(2)} ${y.toFixed(2)}`;
        } else if (!path) {
          path = `M ${x.toFixed(2)} ${y.toFixed(2)}`;
        } else {
          path += ` L ${x.toFixed(2)} ${y.toFixed(2)}`;
        }
        previousIndex = point.index;
      }
      return path;
    }

    function buildAreaPath(points, plot, xExtent, yExtent) {
      if (!points.length) {
        return "";
      }
      const linePath = buildLinePath(points, plot, xExtent, yExtent);
      const first = points[0];
      const last = points[points.length - 1];
      const firstX = scaleX(first.xValue ?? first.index, xExtent.min, xExtent.max, plot.left, plot.right - plot.left);
      const lastX = scaleX(last.xValue ?? last.index, xExtent.min, xExtent.max, plot.left, plot.right - plot.left);
      const bottom = plot.bottom;
      return `${linePath} L ${lastX.toFixed(2)} ${bottom.toFixed(2)} L ${firstX.toFixed(2)} ${bottom.toFixed(2)} Z`;
    }

    function scaleX(value, min, max, left, width) {
      if (Math.abs(max - min) < 1e-12) {
        return left + width / 2;
      }
      return left + ((value - min) / (max - min)) * width;
    }

    function scaleY(value, min, max, top, height) {
      if (Math.abs(max - min) < 1e-12) {
        return top + height / 2;
      }
      return top + height - ((value - min) / (max - min)) * height;
    }

    function niceExtent(values) {
      const actual = extent(values);
      if (!actual) {
        return { min: 0, max: 1 };
      }
      if (Math.abs(actual.max - actual.min) < 1e-12) {
        const padding = Math.abs(actual.max) > 1 ? Math.abs(actual.max) * 0.1 : 1;
        return { min: actual.min - padding, max: actual.max + padding };
      }
      const padding = (actual.max - actual.min) * 0.08;
      return { min: actual.min - padding, max: actual.max + padding };
    }

    function niceTicks(min, max, count) {
      if (Math.abs(max - min) < 1e-12) {
        return [min];
      }
      const rawStep = Math.abs(max - min) / Math.max(1, count);
      const magnitude = 10 ** Math.floor(Math.log10(rawStep));
      const residual = rawStep / magnitude;
      let niceStep = magnitude;
      if (residual >= 5) {
        niceStep = 10 * magnitude;
      } else if (residual >= 2) {
        niceStep = 5 * magnitude;
      } else if (residual >= 1) {
        niceStep = 2 * magnitude;
      }

      const ticks = [];
      const start = Math.ceil(min / niceStep) * niceStep;
      const end = Math.floor(max / niceStep) * niceStep;
      for (let value = start; value <= end + niceStep * 0.5; value += niceStep) {
        ticks.push(Number(value.toFixed(12)));
      }
      if (!ticks.length) {
        return [min, max];
      }
      return ticks;
    }

    function smoothValues(values, windowSize) {
      if (windowSize <= 1) {
        return values.slice();
      }
      const radius = Math.floor(windowSize / 2);
      return values.map((value, index) => {
        if (value === null) {
          return null;
        }
        let total = 0;
        let count = 0;
        for (let cursor = Math.max(0, index - radius); cursor <= Math.min(values.length - 1, index + radius); cursor += 1) {
          const candidate = values[cursor];
          if (candidate !== null) {
            total += candidate;
            count += 1;
          }
        }
        return count ? total / count : value;
      });
    }

    function groupedVisibleMetrics() {
      return groups
        .map((group) => ({
          label: group.label,
          metrics: group.metrics.filter((metric) => metricMatchesSearch(metric)),
        }))
        .filter((group) => group.metrics.length);
    }

    function metricMatchesSearch(metric) {
      if (!state.search) {
        return true;
      }
      const label = humanizeMetric(metric).toLowerCase();
      return metric.toLowerCase().includes(state.search) || label.includes(state.search);
    }

    function chartDescription(selected, normalized) {
      const metricText = selected.length === 1
        ? humanizeMetric(selected[0])
        : `${selected.length} selected metrics`;
      const smoothing = state.smoothingWindow > 1 ? ` with ${state.smoothingWindow}-point smoothing` : "";
      const scale = normalized ? "normalized to 0-1 for comparison" : "shown in raw metric units";
      return `${metricText}, ${scale}${smoothing}.`;
    }

    function describeTimeline() {
      if (payload.meta.xKey === "iteration" && isFiniteNumber(payload.meta.firstIteration) && isFiniteNumber(payload.meta.lastIteration)) {
        return `Iterations ${formatCount(payload.meta.firstIteration)} through ${formatCount(payload.meta.lastIteration)}, with checkpoint annotations overlaid on the trend chart.`;
      }
      return "Sequential training snapshots with checkpoint annotations overlaid on the trend chart.";
    }

    function describeIterationSpan() {
      if (!isFiniteNumber(payload.meta.firstIteration) || !isFiniteNumber(payload.meta.lastIteration)) {
        return formatCount(payload.meta.rowCount);
      }
      return `${formatCount(payload.meta.firstIteration)}-${formatCount(payload.meta.lastIteration)}`;
    }

    function describeMetricFilter() {
      if (!state.search) {
        return "Every numeric metric from the JSONL file is shown below.";
      }
      return `Filtered to metrics matching "${refs.metricSearch.value.trim()}".`;
    }

    function tooltipTitle(row, index) {
      const parts = [`snapshot ${index + 1}`];
      if (isFiniteNumber(row.iteration)) {
        parts.push(`iteration ${formatCount(row.iteration)}`);
      }
      if (row.checkpoint) {
        parts.push(basename(row.checkpoint));
      }
      return parts.join(" · ");
    }

    function metaPill(label, value) {
      return `${label}: ${formatMetaValue(value)}`;
    }

    function formatMetaValue(value) {
      if (isFiniteNumber(value)) {
        return formatCount(Number(value));
      }
      return String(value);
    }

    function humanizeMetric(metric) {
      return metric
        .replace(/^eval_(\d+)_/, "eval $1p ")
        .replace(/_/g, " ")
        .replace(/\bavg\b/g, "avg.")
        .replace(/\bkl\b/gi, "KL")
        .replace(/\bppo\b/gi, "PPO")
        .replace(/\bvs\b/gi, "vs.")
        .replace(/\bmo\b/gi, "MO")
        .replace(/\baa\b/gi, "AA")
        .replace(/(^|\s)\S/g, (match) => match.toUpperCase());
    }

    function formatMetricValue(metric, value) {
      if (!isFiniteNumber(value)) {
        return "—";
      }
      const numeric = Number(value);
      if (isPercentMetric(metric)) {
        return `${(numeric * 100).toFixed(Math.abs(numeric) >= 0.2 ? 1 : 2)}%`;
      }
      if (isCountMetric(metric)) {
        if (Math.abs(numeric - Math.round(numeric)) < 1e-9) {
          return formatCount(Math.round(numeric));
        }
        return numeric.toLocaleString(undefined, {
          minimumFractionDigits: 0,
          maximumFractionDigits: 2,
        });
      }
      return formatFloat(numeric);
    }

    function formatSignedMetricValue(metric, value) {
      if (!isFiniteNumber(value)) {
        return "—";
      }
      const numeric = Number(value);
      const sign = numeric > 0 ? "+" : "";
      return `${sign}${formatMetricValue(metric, numeric)}`;
    }

    function formatAxisValue(value, normalized) {
      if (normalized) {
        return Number(value).toFixed(2);
      }
      return formatFloat(Number(value));
    }

    function formatFloat(value) {
      const absolute = Math.abs(value);
      if (absolute >= 1000) {
        return value.toLocaleString(undefined, {
          minimumFractionDigits: 0,
          maximumFractionDigits: 1,
        });
      }
      if (absolute >= 100) {
        return value.toFixed(1);
      }
      if (absolute >= 10) {
        return value.toFixed(2);
      }
      if (absolute >= 1) {
        return value.toFixed(3);
      }
      if (absolute >= 0.01) {
        return value.toFixed(4);
      }
      return value.toExponential(2);
    }

    function formatCount(value) {
      return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
    }

    function formatCoverage(validCount, totalCount) {
      return `${formatCount(validCount)}/${formatCount(totalCount)} points`;
    }

    function formatWindow(windowSize) {
      return windowSize <= 1 ? "off" : `${windowSize} samples`;
    }

    function isPercentMetric(metric) {
      return metric.includes("win_rate") || metric.endsWith("_rate") || metric.endsWith("_fraction");
    }

    function isCountMetric(metric) {
      return metric.endsWith("_count") || ["games", "transitions", "updates", "iteration"].includes(metric);
    }

    function colorForMetric(metric) {
      const index = metrics.indexOf(metric);
      return colorPalette[(index >= 0 ? index : 0) % colorPalette.length];
    }

    function summaryCard(label, value) {
      return `
        <article class="summary-card">
          <p class="summary-label">${escapeHtml(label)}</p>
          <p class="summary-value">${escapeHtml(String(value))}</p>
        </article>
      `;
    }

    function pill(text) {
      return `<span class="pill">${escapeHtml(String(text))}</span>`;
    }

    function basename(path) {
      return String(path).split(/[\\\\/]/).pop() || String(path);
    }

    function extent(values) {
      if (!values.length) {
        return null;
      }
      return {
        min: Math.min(...values),
        max: Math.max(...values),
      };
    }

    function clamp(value, min, max) {
      return Math.min(max, Math.max(min, value));
    }

    function sameMetricSet(left, right) {
      if (left.length !== right.length) {
        return false;
      }
      const leftSorted = [...left].sort();
      const rightSorted = [...right].sort();
      return leftSorted.every((value, index) => value === rightSorted[index]);
    }

    function isFiniteNumber(value) {
      return typeof value === "number" && Number.isFinite(value);
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }
  </script>
</body>
</html>
"""

    return (
        template.replace("__SABOTER_METRICS_TITLE__", title_html)
        .replace("__SABOTER_METRICS_PAYLOAD__", payload_json)
    )


def save_metrics_dashboard(
    path: str | Path,
    rows: Sequence[MetricRow],
    *,
    title: str = "Saboter Metrics Dashboard",
) -> None:
    """Write a rendered metrics dashboard to disk."""

    Path(path).write_text(render_metrics_dashboard(rows, title=title), encoding="utf-8")


def _clean_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _numeric_metrics(rows: Sequence[MetricRow]) -> list[str]:
    numeric = {
        key
        for row in rows
        for key, value in row.items()
        if _is_number(value)
    }
    return sorted(numeric, key=_metric_sort_key)


def _grouped_metrics(metrics: Sequence[str]) -> list[dict[str, object]]:
    grouped: list[dict[str, object]] = []
    for label in GROUP_ORDER:
        group_metrics = [metric for metric in metrics if _metric_group(metric) == label]
        if group_metrics:
            grouped.append({"label": label, "metrics": group_metrics})
    return grouped


def _metric_sort_key(metric: str) -> tuple[int, int, str]:
    label = _metric_group(metric)
    group_rank = GROUP_ORDER.index(label)
    preferred = GROUP_PREFERRED_ORDER.get(label, [])
    preferred_rank = preferred.index(metric) if metric in preferred else len(preferred) + 1
    return (group_rank, preferred_rank, metric)


def _metric_group(metric: str) -> str:
    if metric.startswith("eval_"):
        return "Evaluation"
    if (
        metric.endswith("_loss")
        or metric in {"loss", "approx_kl", "clip_fraction", "entropy", "grad_norm"}
    ):
        return "Training core"
    if metric in {
        "avg_reward",
        "avg_shaping_reward",
        "avg_terminal_reward",
        "miners_win_rate",
        "avg_game_length",
        "avg_gold_reaches",
        "avg_public_stone_reaches",
        "avg_revealed_goals",
        "avg_rollout_entropy",
        "avg_rollout_value",
    }:
        return "Rewards & outcomes"
    if (
        metric.startswith("avg_")
        or "belief" in metric
        or "distance" in metric
        or "reachable" in metric
        or "frontier" in metric
        or "legal_actions" in metric
    ):
        return "State & beliefs"
    if "_rate" in metric or metric.endswith("_fraction"):
        return "Action rates"
    if metric.endswith("_count") or metric in {"games", "transitions", "updates", "iteration"}:
        return "Counts & schedule"
    return "Other"


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
