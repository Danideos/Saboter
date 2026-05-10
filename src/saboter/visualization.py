"""Rendering helpers for Saboteur game replays."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable


GOAL_COORDS: tuple[tuple[int, int], ...] = ((8, -2), (8, 0), (8, 2))
START_CARD = {
    "id": "start",
    "type": "start",
    "edges": ["E", "N", "S", "W"],
    "groups": [["E", "N", "S", "W"]],
}
GOAL_CARD_FALLBACKS = {
    "gold": {
        "id": "goal_gold",
        "type": "goal",
        "edges": ["E", "N", "S", "W"],
        "groups": [["E", "N", "S", "W"]],
        "goal_kind": "gold",
    },
    "stone_ne": {
        "id": "goal_stone_ne",
        "type": "goal",
        "edges": ["E", "N"],
        "groups": [["E", "N"]],
        "goal_kind": "stone",
    },
    "stone_nw": {
        "id": "goal_stone_nw",
        "type": "goal",
        "edges": ["N", "W"],
        "groups": [["N", "W"]],
        "goal_kind": "stone",
    },
}


def render_board(board_tiles: Iterable[dict[str, object]]) -> str:
    """Render public board tiles as an ASCII tunnel map."""

    tiles = {
        (int(tile["x"]), int(tile["y"])): tile
        for tile in board_tiles
        if isinstance(tile, dict) and isinstance(tile.get("x"), int) and isinstance(tile.get("y"), int)
    }
    if not tiles:
        return "<empty board>"

    min_x = min(x for x, _y in tiles)
    max_x = max(x for x, _y in tiles)
    min_y = min(y for _x, y in tiles)
    max_y = max(y for _x, y in tiles)
    lines = [
        "Legend: S=start, ?=hidden goal, $=gold, X=stone, +=path",
    ]
    for y in range(min_y, max_y + 1):
        rendered_rows = ["", "", ""]
        for x in range(min_x, max_x + 1):
            tile_rows = _render_tile(tiles.get((x, y)))
            for row_index, row in enumerate(tile_rows):
                rendered_rows[row_index] += row + " "
        for row_index, row in enumerate(rendered_rows):
            prefix = f"y={y:>3} " if row_index == 1 else "      "
            lines.append(prefix + row.rstrip())
    return "\n".join(lines)


def build_public_board_snapshots(result: object) -> list[list[dict[str, object]]]:
    """Reconstruct public board state after each public replay event.

    Snapshot 0 is the initial public setup. Snapshot N is the board after the
    first N public events have been applied.
    """

    game = _game_dict(result)
    board = _initial_public_board()
    snapshots = [_sorted_tiles(board)]
    history = game.get("history", [])
    if not isinstance(history, list):
        return snapshots
    for event in history:
        if isinstance(event, dict):
            _apply_public_event(board, event)
        snapshots.append(_sorted_tiles(board))
    return snapshots


def render_game(result: object, *, max_events: int | None = None) -> str:
    """Render a compact human-readable replay from a GameResult-like object."""

    game = _game_dict(result)

    roles = _dict_from(game.get("roles"))
    rewards = _dict_from(game.get("rewards"))
    agent_names = game.get("agent_names", [])
    lines = [
        f"Seed: {game.get('seed')} | players: {game.get('num_players')} | "
        f"steps: {game.get('steps')} | outcome: {game.get('outcome')}",
        "Players:",
    ]
    if isinstance(agent_names, list):
        for player_id, agent_name in enumerate(agent_names):
            role = roles.get(str(player_id), roles.get(player_id, "?"))
            reward = rewards.get(str(player_id), rewards.get(player_id, "?"))
            lines.append(f"  P{player_id}: {agent_name} | role={role} | reward={reward}")

    history = game.get("history", [])
    if isinstance(history, list):
        events = history if max_events is None else history[:max_events]
        lines.append("Events:")
        for index, event in enumerate(events, start=1):
            if isinstance(event, dict):
                lines.append(f"  {index:>3}. {render_event(event)}")
        if max_events is not None and len(history) > max_events:
            lines.append(f"  ... {len(history) - max_events} more events omitted")

    lines.append("Final Board:")
    board = game.get("final_board", [])
    if isinstance(board, list):
        lines.append(render_board(board))
    else:
        lines.append("<board unavailable>")
    return "\n".join(lines)


def render_html_game(result: object) -> str:
    """Render a self-contained HTML replay viewer."""

    game = _game_dict(result)
    payload = {
        "game": game,
        "events": [event for event in game.get("history", []) if isinstance(event, dict)],
        "snapshots": build_public_board_snapshots(game),
        "debugSteps": _debug_steps(game),
        "eventLabels": [
            render_event(event) for event in game.get("history", []) if isinstance(event, dict)
        ],
    }
    payload_json = json.dumps(payload, sort_keys=True).replace("</", "<\\/")
    title = html.escape(
        f"Saboteur Replay seed {game.get('seed')} - {game.get('outcome')}",
        quote=True,
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #64707d;
      --line: #cfd6de;
      --path: #7a4f2a;
      --path-bg: #e8d8bb;
      --start: #2d6cdf;
      --gold: #c99516;
      --stone: #6f7782;
      --miner: #1f8a5b;
      --saboteur: #b43b43;
      --focus: #2d6cdf;
    }}
    * {{ box-sizing: border-box; }}
    html {{ height: 100%; overflow: hidden; }}
    body {{
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .shell {{
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      height: 100vh;
      min-height: 0;
    }}
    aside {{
      min-height: 0;
      background: var(--panel);
      border-right: 1px solid var(--line);
      padding: 18px;
      overflow: auto;
    }}
    main {{
      min-width: 0;
      min-height: 0;
      height: 100vh;
      overflow: hidden;
      padding: 18px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) 206px;
      gap: 14px;
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 18px 0 8px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0;
      color: var(--muted);
    }}
    .meta {{
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      background: #fbfcfd;
      white-space: nowrap;
    }}
    .player-list {{
      display: grid;
      gap: 8px;
    }}
    .player {{
      position: relative;
      display: grid;
      grid-template-columns: 34px 1fr auto;
      gap: 8px;
      align-items: center;
      min-height: 42px;
      border: 1px solid var(--line);
      border-left-width: 5px;
      border-radius: 6px;
      padding: 7px 8px;
      background: #fbfcfd;
    }}
    .player.miner {{ border-left-color: var(--miner); }}
    .player.saboteur {{ border-left-color: var(--saboteur); }}
    .player.hidden {{ border-left-color: var(--line); }}
    .player-id {{ font-weight: 700; }}
    .player-agent {{ color: var(--muted); overflow-wrap: anywhere; }}
    .player-role {{ font-weight: 700; }}
    .player.miner .player-role {{ color: var(--miner); }}
    .player.saboteur .player-role {{ color: var(--saboteur); }}
    .player.hidden .player-role {{ color: var(--muted); }}
    .player-belief {{
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
    }}
    .player-observer {{
      color: var(--focus);
      font-weight: 700;
    }}
    .controls {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      display: grid;
      grid-template-columns: minmax(160px, 1fr) 92px;
      gap: 8px 12px;
      align-items: center;
    }}
    #stepTitle {{
      grid-column: 1 / -1;
      min-height: 21px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    #stepSlider {{ grid-column: 1; }}
    input[type="range"] {{ width: 100%; accent-color: var(--focus); }}
    .step-label {{ width: 92px; text-align: right; color: var(--muted); }}
    .viewer {{
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(250px, 360px);
      gap: 14px;
    }}
    .board-panel,
    .event-panel {{
      min-width: 0;
      min-height: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: auto;
    }}
    .board-wrap {{ padding: 16px; min-width: max-content; }}
    .board {{
      display: grid;
      gap: 7px;
      align-items: center;
      justify-items: center;
    }}
    .tile {{
      position: relative;
      width: 58px;
      height: 58px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fafafa;
      display: grid;
      place-items: center;
      font-weight: 800;
      font-size: 18px;
      color: var(--text);
    }}
    .tile.path {{ background: var(--path-bg); }}
    .tile.start {{ background: #dce8ff; color: var(--start); }}
    .tile.goal.hidden {{ background: #eef0f2; color: var(--stone); }}
    .tile.goal.gold {{ background: #fff0bd; color: var(--gold); }}
    .tile.goal.stone {{ background: #e7e9ec; color: var(--stone); }}
    .tile.empty {{
      border-color: transparent;
      background: transparent;
    }}
    .action-badge {{
      position: absolute;
      right: -6px;
      top: -7px;
      z-index: 3;
      min-width: 36px;
      border: 1px solid var(--focus);
      border-radius: 999px;
      padding: 2px 5px;
      background: #ffffff;
      color: var(--focus);
      font-size: 11px;
      font-weight: 800;
      text-align: center;
      box-shadow: 0 2px 8px rgba(23, 32, 42, 0.14);
    }}
    .action-badge.selected {{
      background: var(--focus);
      color: #ffffff;
    }}
    .edge {{
      position: absolute;
      background: var(--path);
      border-radius: 2px;
      display: none;
    }}
    .edge.n,
    .edge.s {{
      width: 7px;
      height: 26px;
      left: 25px;
    }}
    .edge.e,
    .edge.w {{
      width: 26px;
      height: 7px;
      top: 25px;
    }}
    .edge.n {{ top: 0; }}
    .edge.s {{ bottom: 0; }}
    .edge.e {{ right: 0; }}
    .edge.w {{ left: 0; }}
    .has-n .edge.n,
    .has-e .edge.e,
    .has-s .edge.s,
    .has-w .edge.w {{ display: block; }}
    .tunnels {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }}
    .tunnel {{
      stroke: var(--path);
      stroke-width: 7;
      stroke-linecap: round;
      fill: none;
    }}
    .center {{
      z-index: 1;
      width: 28px;
      height: 28px;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.72);
      display: grid;
      place-items: center;
    }}
    .event-panel {{
      --debug-panel-height: 180px;
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(96px, var(--debug-panel-height)) 12px minmax(0, 1fr);
    }}
    .event-current {{
      border-bottom: 1px solid var(--line);
      padding: 12px;
      min-height: 68px;
    }}
    .event-current strong {{ display: block; margin-bottom: 4px; }}
    .event-current span {{ color: var(--muted); }}
    .debug-panel {{
      padding: 10px 12px;
      overflow: auto;
      display: grid;
      gap: 10px;
      align-content: start;
    }}
    .event-resizer {{
      position: relative;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      background: #f3f6f9;
      cursor: ns-resize;
      touch-action: none;
    }}
    .event-resizer::before {{
      content: "";
      position: absolute;
      left: 50%;
      top: 50%;
      width: 44px;
      height: 4px;
      border-radius: 999px;
      background: #b2bdc8;
      transform: translate(-50%, -50%);
    }}
    .event-resizer:focus-visible {{
      outline: 2px solid var(--focus);
      outline-offset: -2px;
    }}
    body.is-resizing {{
      cursor: ns-resize;
      user-select: none;
    }}
    .debug-title {{
      margin: 0 0 4px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
    }}
    .hand-row,
    .score-row {{
      display: grid;
      gap: 4px;
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 6px;
      background: #fbfcfd;
    }}
    .hand-row strong,
    .score-row strong {{ font-size: 12px; }}
    .belief-list {{
      display: grid;
      gap: 6px;
    }}
    .belief-row {{
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 6px;
      background: #fbfcfd;
    }}
    .belief-meter {{
      position: relative;
      height: 8px;
      border-radius: 999px;
      background: #e6ebf1;
      overflow: hidden;
    }}
    .belief-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--miner), var(--saboteur));
    }}
    .belief-value {{
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .cards,
    .discard-pile {{
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }}
    .card-chip {{
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 2px 5px;
      background: #fff;
      font-size: 12px;
      white-space: nowrap;
    }}
    .score-row.selected {{
      border-color: var(--focus);
      background: #eef5ff;
    }}
    .score-value {{
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    .hand-tray {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 12px;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }}
    .hand-toolbar {{
      display: grid;
      grid-template-columns: minmax(110px, 1fr) auto;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
      color: var(--muted);
    }}
    .hand-controls {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex: 0 0 auto;
    }}
    .mode-tabs,
    .rotation-tabs {{
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 999px;
      overflow: hidden;
      background: #fbfcfd;
    }}
    .mode-tabs button,
    .rotation-tabs button {{
      border: 0;
      border-right: 1px solid var(--line);
      padding: 4px 10px;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font: inherit;
    }}
    .mode-tabs button:last-child,
    .rotation-tabs button:last-child {{ border-right: 0; }}
    .mode-tabs button.active,
    .rotation-tabs button.active {{
      background: var(--focus);
      color: #ffffff;
    }}
    .rotation-tabs {{
      width: 82px;
      justify-content: stretch;
    }}
    .rotation-tabs button {{
      flex: 1;
      padding-inline: 0;
    }}
    .rotation-tabs button:disabled {{
      color: #aab3bd;
      cursor: default;
    }}
    .hand-cards {{
      display: flex;
      gap: 10px;
      overflow-x: auto;
      overflow-y: hidden;
      padding-bottom: 4px;
      min-height: 0;
      align-items: flex-start;
    }}
    .hand-card-stack {{
      flex: 0 0 88px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-height: 0;
    }}
    .hand-card {{
      position: relative;
      width: 88px;
      height: 104px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(#fffdf7, #efe4cd);
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding: 7px;
      cursor: pointer;
      box-shadow: 0 2px 8px rgba(23, 32, 42, 0.08);
    }}
    .hand-card.selected {{
      border-color: var(--focus);
      box-shadow: 0 0 0 2px rgba(45, 108, 223, 0.22), 0 4px 12px rgba(23, 32, 42, 0.14);
    }}
    .hand-card-title {{
      font-size: 11px;
      font-weight: 800;
      overflow-wrap: anywhere;
      line-height: 1.05;
    }}
    .hand-card-art {{
      position: relative;
      flex: 1 1 auto;
      min-height: 38px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(122, 79, 42, 0.28);
      border-radius: 5px;
      background: var(--path-bg);
      overflow: hidden;
    }}
    .hand-card-meta {{
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 10px;
      line-height: 1;
      gap: 4px;
    }}
    .hand-card-scores {{
      display: grid;
      gap: 2px;
      flex: 0 0 auto;
    }}
    .hand-card-prob,
    .discard-score {{
      border: 1px solid rgba(45, 108, 223, 0.35);
      border-radius: 999px;
      padding: 1px 5px;
      background: rgba(255, 255, 255, 0.94);
      color: var(--focus);
      font-size: 10px;
      font-weight: 800;
      line-height: 1.35;
      text-align: center;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .card-score {{
      color: var(--focus);
      font-weight: 800;
      font-variant-numeric: tabular-nums;
    }}
    .discard-score {{
      border-color: rgba(100, 112, 125, 0.32);
      color: var(--muted);
    }}
    .event-list {{
      overflow: auto;
      padding: 8px;
      display: grid;
      gap: 4px;
      align-content: start;
    }}
    .event-row {{
      border: 1px solid transparent;
      border-radius: 5px;
      padding: 6px 8px;
      cursor: pointer;
      color: var(--muted);
    }}
    .event-row.active {{
      border-color: var(--focus);
      background: #eef5ff;
      color: var(--text);
    }}
    .event-row:hover {{ background: #f1f4f7; color: var(--text); }}
    @media (max-width: 900px) {{
      .shell {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); max-height: 170px; }}
      main {{ height: calc(100vh - 170px); }}
      .viewer {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: minmax(160px, 1fr) 92px; }}
      .step-label {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1>Saboteur Replay</h1>
      <div class="meta" id="meta"></div>
      <h2>Players</h2>
      <div class="player-list" id="players"></div>
      <h2>Actions</h2>
      <div class="meta" id="actions"></div>
    </aside>
    <main>
      <section class="controls">
        <strong id="stepTitle"></strong>
        <input id="stepSlider" type="range" min="0" value="0">
        <span class="step-label" id="stepLabel"></span>
      </section>
      <section class="viewer">
        <div class="board-panel">
          <div class="board-wrap">
            <div class="board" id="board"></div>
          </div>
        </div>
        <div class="event-panel">
          <div class="event-current">
            <strong id="eventHeadline"></strong>
            <span id="eventDetail"></span>
          </div>
          <div class="debug-panel" id="debugPanel"></div>
          <div
            class="event-resizer"
            id="eventResizer"
            role="separator"
            aria-label="Resize event details"
            aria-orientation="horizontal"
            tabindex="0"
          ></div>
          <div class="event-list" id="eventList"></div>
        </div>
      </section>
      <section class="hand-tray">
        <div class="hand-toolbar">
          <strong id="handTitle">Hand</strong>
          <div class="hand-controls">
            <div class="mode-tabs" id="overlayTabs" aria-label="Overlay mode">
              <button type="button" data-overlay-mode="prob" class="active">Policy %</button>
              <button type="button" data-overlay-mode="chosen">Chosen</button>
            </div>
            <div class="rotation-tabs" id="rotationTabs" aria-label="Path rotation"></div>
          </div>
        </div>
        <div class="hand-cards" id="handCards"></div>
      </section>
    </main>
  </div>
  <script id="replay-data" type="application/json">{payload_json}</script>
  <script>
    const data = JSON.parse(document.getElementById("replay-data").textContent);
    const game = data.game;
    const snapshots = data.snapshots;
    const events = data.events;
    const eventLabels = data.eventLabels;
    const debugSteps = data.debugSteps || [];
    const goalCoords = [[8, -2], [8, 0], [8, 2]];
    const slider = document.getElementById("stepSlider");
    const board = document.getElementById("board");
    const debugPanel = document.getElementById("debugPanel");
    const eventPanel = document.querySelector(".event-panel");
    const eventCurrent = document.querySelector(".event-current");
    const eventList = document.getElementById("eventList");
    const eventResizer = document.getElementById("eventResizer");
    const DEBUG_PANEL_STORAGE_KEY = "saboteur-replay-debug-panel-height";
    const MIN_DEBUG_PANEL_HEIGHT = 96;
    const MIN_EVENT_LIST_HEIGHT = 120;
    let currentStep = -1;
    let selectedCardSlot = null;
    let selectedRotation = 0;
    let overlayMode = "prob";

    slider.max = Math.max(0, snapshots.length - 1);
    slider.addEventListener("input", () => render(Number(slider.value)));

    function setup() {{
      document.getElementById("meta").replaceChildren(
        pill(`seed ${{game.seed}}`),
        pill(game.outcome),
        pill(`${{game.steps}} steps`),
        pill(`${{game.num_players}} players`)
      );
      const actions = document.getElementById("actions");
      Object.entries(game.action_counts || {{}}).forEach(([name, count]) => {{
        actions.appendChild(pill(`${{name}} ${{count}}`));
      }});
      setupEventResizer();
      document.querySelectorAll("[data-overlay-mode]").forEach(button => {{
        button.addEventListener("click", () => {{
          overlayMode = button.dataset.overlayMode || "prob";
          document.querySelectorAll("[data-overlay-mode]").forEach(node => {{
            node.classList.toggle("active", node.dataset.overlayMode === overlayMode);
          }});
          render(currentStep < 0 ? 0 : currentStep);
        }});
      }});
      events.forEach((event, index) => {{
        const row = document.createElement("div");
        row.className = "event-row";
        row.title = JSON.stringify(event);
        row.textContent = `${{index + 1}}. ${{eventLabels[index]}}`;
        row.addEventListener("click", () => {{
          slider.value = String(index + 1);
          render(index + 1);
        }});
        eventList.appendChild(row);
      }});
      render(0);
    }}

    function setupEventResizer() {{
      const storedHeight = loadStoredDebugPanelHeight();
      if (storedHeight !== null) {{
        setDebugPanelHeight(storedHeight);
      }}
      eventResizer.addEventListener("pointerdown", startEventResize);
      eventResizer.addEventListener("keydown", handleEventResizeKeydown);
      window.addEventListener("resize", () => {{
        setDebugPanelHeight(debugPanel.getBoundingClientRect().height);
      }});
    }}

    function startEventResize(event) {{
      event.preventDefault();
      const startY = event.clientY;
      const startHeight = debugPanel.getBoundingClientRect().height;
      const pointerId = event.pointerId;
      eventResizer.setPointerCapture(pointerId);
      document.body.classList.add("is-resizing");

      const onMove = moveEvent => {{
        setDebugPanelHeight(startHeight + (moveEvent.clientY - startY));
      }};
      const onUp = () => {{
        document.body.classList.remove("is-resizing");
        try {{
          eventResizer.releasePointerCapture(pointerId);
        }} catch (_error) {{
        }}
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        window.removeEventListener("pointercancel", onUp);
        persistDebugPanelHeight();
      }};

      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      window.addEventListener("pointercancel", onUp);
    }}

    function handleEventResizeKeydown(event) {{
      if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
      event.preventDefault();
      const currentHeight = debugPanel.getBoundingClientRect().height;
      const delta = event.key === "ArrowUp" ? -24 : 24;
      setDebugPanelHeight(currentHeight + delta);
      persistDebugPanelHeight();
    }}

    function setDebugPanelHeight(height) {{
      eventPanel.style.setProperty("--debug-panel-height", `${{Math.round(clampDebugPanelHeight(height))}}px`);
    }}

    function clampDebugPanelHeight(height) {{
      const panelHeight = eventPanel.getBoundingClientRect().height;
      const currentHeight = eventCurrent.getBoundingClientRect().height;
      const resizerHeight = eventResizer.getBoundingClientRect().height || 12;
      const maxHeight = Math.max(
        MIN_DEBUG_PANEL_HEIGHT,
        panelHeight - currentHeight - resizerHeight - MIN_EVENT_LIST_HEIGHT
      );
      return Math.max(MIN_DEBUG_PANEL_HEIGHT, Math.min(maxHeight, Number(height) || MIN_DEBUG_PANEL_HEIGHT));
    }}

    function persistDebugPanelHeight() {{
      try {{
        window.localStorage.setItem(
          DEBUG_PANEL_STORAGE_KEY,
          String(Math.round(debugPanel.getBoundingClientRect().height))
        );
      }} catch (_error) {{
      }}
    }}

    function loadStoredDebugPanelHeight() {{
      try {{
        const rawValue = window.localStorage.getItem(DEBUG_PANEL_STORAGE_KEY);
        const parsed = Number(rawValue);
        return Number.isFinite(parsed) ? parsed : null;
      }} catch (_error) {{
        return null;
      }}
    }}

    function render(step) {{
      const snapshot = snapshots[step] || [];
      const debug = debugSteps[step] || null;
      if (step !== currentStep) {{
        currentStep = step;
        selectedCardSlot = defaultSelectedCardSlot(debug);
        selectedRotation = defaultSelectedRotation(debug);
      }}
      renderPlayers(step);
      const cellActions = cellActionMap(debug);
      const bounds = getBounds(snapshot, cellActions);
      board.style.gridTemplateColumns = `repeat(${{bounds.width}}, 58px)`;
      board.replaceChildren();
      for (let y = bounds.minY; y <= bounds.maxY; y++) {{
        for (let x = bounds.minX; x <= bounds.maxX; x++) {{
          const key = coordKey(x, y);
          board.appendChild(renderTile(
            snapshot.find(tile => tile.x === x && tile.y === y),
            cellActions.get(key) || []
          ));
        }}
      }}
      document.getElementById("stepTitle").textContent =
        step === 0 ? "Initial setup" : eventLabels[step - 1];
      document.getElementById("stepLabel").textContent = `${{step}} / ${{snapshots.length - 1}}`;
      document.getElementById("eventHeadline").textContent =
        step === 0 ? "Initial setup" : `Event ${{step}}`;
      document.getElementById("eventDetail").textContent =
        step === 0 ? "Start card and hidden goals are public." : eventLabels[step - 1];
      renderDebug(step);
      renderHandTray(step);
      [...eventList.children].forEach((row, index) => {{
        row.classList.toggle("active", index === step - 1);
      }});
    }}

    function renderPlayers(step) {{
      const players = document.getElementById("players");
      const debug = debugSteps[step] || null;
      const targetActions = targetActionMap(debug);
      const roleBeliefs = roleBeliefMap(debug);
      const observerId = debug && Number.isInteger(debug.actor) ? debug.actor : null;
      players.replaceChildren();
      game.agent_names.forEach((agent, id) => {{
        const finalRole = game.roles[String(id)] ?? game.roles[id] ?? "?";
        const finalReward = game.rewards[String(id)] ?? game.rewards[id] ?? "?";
        const role = finalRole;
        const reward = finalReward;
        const belief = roleBeliefs.get(Number(id)) || null;
        const beliefLine =
          Number(id) === observerId
            ? '<div class="player-belief player-observer">observer</div>'
            : belief
              ? `<div class="player-belief">seen as ${{formatPercent(belief.saboteur_prob || 0)}} sab</div>`
              : "";
        const row = document.createElement("div");
        row.className = `player ${{role}}`;
        row.title = `P${{id}} | ${{agent}} | role ${{finalRole}} | reward ${{finalReward}}`;
        row.innerHTML = `
          <div class="player-id">P${{id}}</div>
          <div>
            <div class="player-agent">${{escapeText(agent)}}</div>
            <div class="player-role">${{escapeText(role)}}</div>
            ${{beliefLine}}
          </div>
          <strong>${{reward}}</strong>
        `;
        const actions = targetActions.get(Number(id)) || [];
        if (actions.length) {{
          row.title += ` | selected-card policy: ${{actions.map(formatActionProb).join(", ")}}`;
          const badge = document.createElement("span");
          badge.className = `action-badge${{actions.some(action => action.selected) ? " selected" : ""}}`;
          badge.textContent = bestActionLabel(actions);
          row.appendChild(badge);
        }}
        players.appendChild(row);
      }});
    }}

    function renderDebug(step) {{
      const panel = document.getElementById("debugPanel");
      const debug = debugSteps[step] || null;
      panel.replaceChildren();
      if (!debug) {{
        panel.textContent = "No private debug data in this replay.";
        return;
      }}
      const actorLine = document.createElement("div");
      actorLine.innerHTML = `<div class="debug-title">Decision</div>${{debug.actor === null || debug.actor === undefined ? "Initial state" : `P${{debug.actor}} · ${{escapeText(debug.controller || "")}}`}}`;
      panel.appendChild(actorLine);

      const selected = document.createElement("div");
      selected.innerHTML = `<div class="debug-title">Selected</div>${{escapeText(debug.selected_action?.label || "none")}}${{debug.selected_card ? ` · card: ${{cardName(debug.selected_card)}}` : ""}}`;
      panel.appendChild(selected);

      const beliefs = otherRoleBeliefs(debug);
      if (beliefs.length) {{
        const section = document.createElement("div");
        section.innerHTML = `<div class="debug-title">Role beliefs</div>`;
        const list = document.createElement("div");
        list.className = "belief-list";
        beliefs.forEach(belief => {{
          const row = document.createElement("div");
          row.className = "belief-row";
          row.title = typeof belief.saboteur_logit === "number"
            ? `raw logit ${{Number(belief.saboteur_logit).toFixed(4)}}`
            : "";
          row.innerHTML = `
            <strong>P${{belief.player_id}}</strong>
            <div class="belief-meter"><div class="belief-fill" style="width: ${{beliefWidth(belief.saboteur_prob)}}"></div></div>
            <span class="belief-value">${{formatPercent(belief.saboteur_prob || 0)}} sab</span>
          `;
          list.appendChild(row);
        }});
        section.appendChild(list);
        panel.appendChild(section);
      }}

      const discard = document.createElement("div");
      discard.innerHTML = `<div class="debug-title">Discard pile</div>`;
      const discardCards = document.createElement("div");
      discardCards.className = "discard-pile";
      const pile = debug.discard_pile_after || debug.discard_pile || [];
      if (!pile.length) discardCards.appendChild(chip("empty"));
      pile.forEach(card => discardCards.appendChild(chip(card ? cardName(card) : "face-down")));
      discard.appendChild(discardCards);
      panel.appendChild(discard);

      const scored = scoredActions(debug);
      if (scored.length) {{
        const scores = document.createElement("div");
        scores.innerHTML = `<div class="debug-title">Selected card policy</div>`;
        selectedCardActions(debug)
          .sort((left, right) => Number(right.prob || 0) - Number(left.prob || 0))
          .forEach(action => {{
            const row = document.createElement("div");
            row.className = `score-row${{action.selected ? " selected" : ""}}`;
            row.title = typeof action.score === "number" ? `raw logit ${{Number(action.score).toFixed(4)}}` : "";
            row.innerHTML = `<strong>${{escapeText(compactActionLabel(action))}}</strong><span class="score-value">${{formatPercent(action.prob || 0)}}</span>`;
            scores.appendChild(row);
          }});
        if (scores.children.length > 1) panel.appendChild(scores);
      }}
    }}

    function renderHandTray(step) {{
      const debug = debugSteps[step] || null;
      const handCards = document.getElementById("handCards");
      const rotationTabs = document.getElementById("rotationTabs");
      const handTitle = document.getElementById("handTitle");
      handCards.replaceChildren();
      rotationTabs.replaceChildren();
      if (!debug || debug.actor === null || debug.actor === undefined) {{
        handTitle.textContent = "Hand";
        handCards.appendChild(chip("No decision hand for this step"));
        return;
      }}
      const actor = String(debug.actor);
      const hand = (debug.hands || {{}})[actor] || [];
      handTitle.textContent = `P${{actor}} hand`;
      hand.forEach((card, slot) => {{
        const node = renderHandCard(card, slot, debug);
        handCards.appendChild(node);
      }});
      if (!hand.length) handCards.appendChild(chip("empty"));

      const selectedCard = hand[selectedCardSlot];
      if (selectedCard && selectedCard.type === "path") {{
        [0, 180].forEach(rotation => {{
          const button = document.createElement("button");
          button.type = "button";
          button.className = rotation === selectedRotation ? "active" : "";
          button.textContent = `r${{rotation}}`;
          button.addEventListener("click", () => {{
            selectedRotation = rotation;
            render(currentStep);
          }});
          rotationTabs.appendChild(button);
        }});
      }} else {{
        [0, 180].forEach(rotation => {{
          const button = document.createElement("button");
          button.type = "button";
          button.disabled = true;
          button.textContent = `r${{rotation}}`;
          rotationTabs.appendChild(button);
        }});
      }}
    }}

    function renderHandCard(card, slot, debug) {{
      const node = document.createElement("div");
      node.className = "hand-card-stack";
      const cardActions = scoredActionsForCard(debug, slot);
      const totalProb = cardProbability(debug, slot);
      const discardProb = actionTypeProbability(cardActions, "discard");
      const cardRotation = slot === selectedCardSlot ? selectedRotation : 0;
      const groups = card && Array.isArray(card.groups) ? rotatedGroups(card.groups, cardRotation) : [];
      node.innerHTML = `
        <button type="button" class="hand-card${{slot === selectedCardSlot ? " selected" : ""}}">
          <div class="hand-card-title">${{escapeText(cardName(card))}}</div>
          <div class="hand-card-art">${{renderTunnels(groups)}}<span class="center">${{cardGlyph(card)}}</span></div>
          <div class="hand-card-meta"><span>slot ${{slot}}</span><span></span></div>
        </button>
        <div class="hand-card-scores">
          <div class="hand-card-prob">total ${{formatPercent(totalProb)}}</div>
          ${{discardProb ? `<div class="discard-score">discard ${{formatPercent(discardProb)}}</div>` : ""}}
        </div>
      `;
      const cardButton = node.querySelector(".hand-card");
      cardButton.title = JSON.stringify(card);
      cardButton.addEventListener("click", () => {{
        selectedCardSlot = slot;
        selectedRotation = 0;
        render(currentStep);
      }});
      return node;
    }}

    function renderTile(tile, actions = []) {{
      const div = document.createElement("div");
      if (!tile) {{
        div.className = "tile empty";
        appendActionBadges(div, actions);
        return div;
      }}
      const groups = tileGroups(tile);
      const center = tileCenter(tile);
      div.className = [
        "tile",
        tile.kind,
        tile.kind === "goal" && !tile.revealed ? "hidden" : "",
        tile.goal_kind || "",
      ].filter(Boolean).join(" ");
      div.title = tileTitle(tile);
      div.innerHTML = `
        ${{renderTunnels(groups)}}
        <span class="center">${{center}}</span>
      `;
      appendActionBadges(div, actions);
      return div;
    }}

    function appendActionBadges(tileNode, actions) {{
      if (!actions.length) return;
      const visible = overlayMode === "chosen"
        ? actions.filter(action => action.selected)
        : actions;
      if (!visible.length) return;
      const best = visible.slice().sort((left, right) => Number(right.prob || 0) - Number(left.prob || 0))[0];
      const badge = document.createElement("span");
      badge.className = `action-badge${{visible.some(action => action.selected) ? " selected" : ""}}`;
      badge.textContent = overlayMode === "chosen" ? "played" : formatPercent(best.prob || 0);
      badge.title = visible.map(formatActionProb).join("\\n");
      tileNode.appendChild(badge);
    }}

    function getBounds(snapshot, cellActions = new Map()) {{
      if (!snapshot.length && !cellActions.size) return {{minX: 0, maxX: 0, minY: 0, maxY: 0, width: 1}};
      const xs = snapshot.map(tile => tile.x);
      const ys = snapshot.map(tile => tile.y);
      for (const key of cellActions.keys()) {{
        const [x, y] = key.split(",").map(Number);
        xs.push(x);
        ys.push(y);
      }}
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      return {{minX, maxX, minY, maxY, width: maxX - minX + 1}};
    }}

    function tileCenter(tile) {{
      if (tile.kind === "start") return "S";
      if (tile.kind === "goal" && !tile.revealed) return "?";
      if (tile.goal_kind === "gold") return "$";
      if (tile.goal_kind === "stone") return "X";
      return hasDeadOrSplitGroups(tileGroups(tile)) ? "x" : "+";
    }}

    function tileTitle(tile) {{
      const card = tile.card || {{}};
      const coord = `(${{tile.x}}, ${{tile.y}})`;
      if (tile.kind === "goal") {{
        return tile.revealed
          ? `${{coord}} goal: ${{tile.goal_kind}}`
          : `${{coord}} hidden goal`;
      }}
      const edges = [...rotatedEdges(card.edges || [], tile.rotation || 0)]
        .map(edge => edge.toUpperCase())
        .join("");
      const groups = tileGroups(tile)
        .map(group => `[${{group.map(edge => edge.toUpperCase()).join("")}}]`)
        .join(" ");
      return `${{coord}} ${{card.id || tile.kind}} r${{tile.rotation || 0}} edges:${{edges}} groups:${{groups}}`;
    }}

    function tileGroups(tile) {{
      const card = tile.card || {{}};
      const rotation = tile.rotation || 0;
      if (Array.isArray(card.groups) && card.groups.length) {{
        return rotatedGroups(card.groups, rotation);
      }}
      const edges = [...rotatedEdges(card.edges || [], rotation)];
      return edges.length ? [edges] : [];
    }}

    function rotatedGroups(groups, rotation) {{
      return groups
        .map(group => group
          .map(edge => rotateEdge(String(edge).toLowerCase(), rotation))
          .filter(Boolean)
        )
        .filter(group => group.length);
    }}

    function rotateEdge(edge, rotation) {{
      const normalized = ((rotation % 360) + 360) % 360;
      if (normalized !== 180) return ["n", "e", "s", "w"].includes(edge) ? edge : null;
      const opposite = {{n: "s", e: "w", s: "n", w: "e"}};
      return opposite[edge] || null;
    }}

    function renderTunnels(groups) {{
      if (!groups.length) return "";
      const side = {{n: [29, 0], e: [58, 29], s: [29, 58], w: [0, 29]}};
      const stub = {{n: [29, 18], e: [40, 29], s: [29, 40], w: [18, 29]}};
      const center = [29, 29];
      const lines = [];
      groups.forEach(group => {{
        const target = group.length === 1 ? stub[group[0]] : center;
        group.forEach(edge => {{
          const start = side[edge];
          if (!start || !target) return;
          lines.push(`<line class="tunnel" x1="${{start[0]}}" y1="${{start[1]}}" x2="${{target[0]}}" y2="${{target[1]}}"></line>`);
        }});
      }});
      return `<svg class="tunnels" viewBox="0 0 58 58" aria-hidden="true">${{lines.join("")}}</svg>`;
    }}

    function hasDeadOrSplitGroups(groups) {{
      return groups.length > 1 || groups.some(group => group.length === 1);
    }}

    function rotatedEdges(edges, rotation) {{
      const normalized = ((rotation % 360) + 360) % 360;
      const edgeSet = new Set(edges.map(edge => String(edge).toLowerCase()));
      if (normalized !== 180) return edgeSet;
      const opposite = {{n: "s", e: "w", s: "n", w: "e"}};
      return new Set([...edgeSet].map(edge => opposite[edge]).filter(Boolean));
    }}

    function defaultSelectedCardSlot(debug) {{
      if (!debug) return null;
      if (debug.selected_action && Number.isInteger(debug.selected_action.card_slot)) {{
        return debug.selected_action.card_slot;
      }}
      const actor = debug.actor;
      const hand = actor === null || actor === undefined ? [] : ((debug.hands || {{}})[String(actor)] || []);
      return hand.length ? 0 : null;
    }}

    function defaultSelectedRotation(debug) {{
      if (debug && debug.selected_action && Number.isInteger(debug.selected_action.rotation)) {{
        return debug.selected_action.rotation;
      }}
      return 0;
    }}

    function scoredActions(debug) {{
      if (!debug || !Array.isArray(debug.legal_actions)) return [];
      const scored = debug.legal_actions
        .filter(action => typeof action.score === "number")
        .map(action => ({{...action}}));
      const maxScore = scored.length ? Math.max(...scored.map(action => Number(action.score))) : 0;
      const weights = scored.map(action => Math.exp(Number(action.score) - maxScore));
      const total = weights.reduce((sum, value) => sum + value, 0);
      scored.forEach((action, index) => {{
        action.prob = total > 0 ? weights[index] / total : 0;
      }});
      return scored;
    }}

    function scoredActionsForCard(debug, slot) {{
      return scoredActions(debug).filter(action => action.card_slot === slot);
    }}

    function cardProbability(debug, slot) {{
      return scoredActionsForCard(debug, slot)
        .reduce((sum, action) => sum + Number(action.prob || 0), 0);
    }}

    function actionTypeProbability(actions, type) {{
      return actions
        .filter(action => action.type === type)
        .reduce((sum, action) => sum + Number(action.prob || 0), 0);
    }}

    function selectedCardActions(debug) {{
      return scoredActions(debug).filter(action => {{
        if (action.card_slot !== selectedCardSlot) return false;
        if (action.type === "play_path" && Number.isInteger(action.rotation)) {{
          return action.rotation === selectedRotation;
        }}
        return true;
      }});
    }}

    function cellActionMap(debug) {{
      const result = new Map();
      selectedCardActions(debug).forEach(action => {{
        let x = action.x;
        let y = action.y;
        if (action.type === "map_goal" && Number.isInteger(action.goal_index)) {{
          const coord = goalCoords[action.goal_index];
          if (coord) {{
            x = coord[0];
            y = coord[1];
          }}
        }}
        if (!Number.isInteger(x) || !Number.isInteger(y)) return;
        const key = coordKey(x, y);
        if (!result.has(key)) result.set(key, []);
        result.get(key).push(action);
      }});
      return result;
    }}

    function targetActionMap(debug) {{
      const result = new Map();
      selectedCardActions(debug).forEach(action => {{
        if (overlayMode === "chosen" && !action.selected) return;
        if (!Number.isInteger(action.target_player)) return;
        if (!result.has(action.target_player)) result.set(action.target_player, []);
        result.get(action.target_player).push(action);
      }});
      return result;
    }}

    function roleBeliefEntries(debug) {{
      if (!debug || !Array.isArray(debug.role_beliefs)) return [];
      return debug.role_beliefs.filter(belief =>
        belief &&
        Number.isInteger(belief.player_id) &&
        typeof belief.saboteur_prob === "number"
      );
    }}

    function roleBeliefMap(debug) {{
      const result = new Map();
      roleBeliefEntries(debug).forEach(belief => {{
        result.set(Number(belief.player_id), belief);
      }});
      return result;
    }}

    function otherRoleBeliefs(debug) {{
      return roleBeliefEntries(debug)
        .filter(belief => !belief.is_self)
        .sort((left, right) => Number(right.saboteur_prob || 0) - Number(left.saboteur_prob || 0));
    }}

    function coordKey(x, y) {{
      return `${{x}},${{y}}`;
    }}

    function beliefWidth(value) {{
      const percent = Math.max(0, Math.min(100, Number(value || 0) * 100));
      return `${{percent}}%`;
    }}

    function formatPercent(value) {{
      return `${{Math.round(Number(value || 0) * 100)}}%`;
    }}

    function formatActionProb(action) {{
      return `${{formatPercent(action.prob || 0)}} ${{action.label}}`;
    }}

    function compactActionLabel(action) {{
      if (action.type === "play_path") return `place (${{action.x}}, ${{action.y}}) r${{action.rotation}}`;
      if (action.type === "rockfall") return `rockfall (${{action.x}}, ${{action.y}})`;
      if (action.type === "map_goal") return `map goal ${{action.goal_index}}`;
      if (action.type === "sabotage") return `sabotage P${{action.target_player}} ${{action.tool}}`;
      if (action.type === "repair") return `repair P${{action.target_player}} ${{action.tool}}`;
      if (action.type === "discard") return "discard";
      return action.label || action.type || "action";
    }}

    function bestActionLabel(actions) {{
      const best = actions.slice().sort((left, right) => Number(right.prob || 0) - Number(left.prob || 0))[0];
      return best ? formatPercent(best.prob || 0) : "";
    }}

    function cardGlyph(card) {{
      if (!card) return "?";
      if (card.type === "map") return "map";
      if (card.type === "sabotage") return "break";
      if (card.type === "repair") return "fix";
      if (card.type === "rockfall") return "rock";
      if (card.type === "goal") return card.goal_kind === "gold" ? "$" : "X";
      return card.id && String(card.id).startsWith("dead") ? "x" : "+";
    }}

    function pill(text) {{
      const node = document.createElement("span");
      node.className = "pill";
      node.textContent = text;
      return node;
    }}

    function chip(text) {{
      const node = document.createElement("span");
      node.className = "card-chip";
      node.textContent = text;
      return node;
    }}

    function strong(text) {{
      const node = document.createElement("strong");
      node.textContent = text;
      return node;
    }}

    function cardName(card) {{
      return card && (card.id || card.type) ? String(card.id || card.type) : "unknown";
    }}

    function escapeText(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    setup();
  </script>
</body>
</html>
"""


def save_html_replay(path: str | Path, result: object) -> None:
    Path(path).write_text(render_html_game(result), encoding="utf-8")


def render_event(event: dict[str, object]) -> str:
    actor = _player(event.get("actor"))
    action_type = event.get("action_type")
    if action_type == "play_path":
        card = _card_id(event.get("card"))
        return (
            f"{actor} played {card} at "
            f"({event.get('x')}, {event.get('y')}) r{event.get('rotation')}"
        )
    if action_type == "sabotage":
        return f"{actor} broke {_player(event.get('target_player'))}'s {event.get('tool')}"
    if action_type == "repair":
        return f"{actor} repaired {_player(event.get('target_player'))}'s {event.get('tool')}"
    if action_type == "map_goal":
        return f"{actor} mapped goal {event.get('goal_index')}"
    if action_type == "rockfall":
        removed = _card_id(event.get("removed_card"))
        return f"{actor} used rockfall at ({event.get('x')}, {event.get('y')}); removed {removed}"
    if action_type == "discard":
        return f"{actor} discarded face-down"
    if action_type == "reveal_goal":
        return f"{actor} revealed goal {event.get('goal_index')}: {event.get('revealed_goal_kind')}"
    return f"{actor} {action_type}"


def _render_tile(tile: dict[str, object] | None) -> tuple[str, str, str]:
    if tile is None:
        return ("   ", "   ", "   ")

    center = _tile_center(tile)
    edges = _tile_edges(tile)
    north = "|" if "N" in edges else " "
    east = "-" if "E" in edges else " "
    south = "|" if "S" in edges else " "
    west = "-" if "W" in edges else " "
    return (f" {north} ", f"{west}{center}{east}", f" {south} ")


def _tile_center(tile: dict[str, object]) -> str:
    kind = tile.get("kind")
    if kind == "start":
        return "S"
    if kind == "goal":
        if not tile.get("revealed"):
            return "?"
        goal_kind = tile.get("goal_kind")
        if goal_kind == "gold":
            return "$"
        if goal_kind == "stone":
            return "X"
        return "?"
    groups = _tile_groups(tile)
    if len(groups) > 1 or any(len(group) == 1 for group in groups):
        return "x"
    return "+"


def _tile_edges(tile: dict[str, object]) -> set[str]:
    card = tile.get("card")
    if not isinstance(card, dict):
        return set()
    edges = card.get("edges")
    if not isinstance(edges, list):
        return set()
    rotation = tile.get("rotation", 0)
    normalized_rotation = rotation if isinstance(rotation, int) else 0
    return _rotate_edge_names(
        {edge for edge in edges if isinstance(edge, str)},
        normalized_rotation,
    )


def _tile_groups(tile: dict[str, object]) -> list[set[str]]:
    card = tile.get("card")
    if not isinstance(card, dict):
        return []
    rotation = tile.get("rotation", 0)
    normalized_rotation = rotation if isinstance(rotation, int) else 0
    raw_groups = card.get("groups")
    if isinstance(raw_groups, list) and raw_groups:
        groups: list[set[str]] = []
        for raw_group in raw_groups:
            if not isinstance(raw_group, list):
                continue
            group = _rotate_edge_names(
                {edge for edge in raw_group if isinstance(edge, str)},
                normalized_rotation,
            )
            if group:
                groups.append(group)
        return groups
    edges = _tile_edges(tile)
    return [edges] if edges else []


def _player(value: object) -> str:
    return f"P{value}" if isinstance(value, int) else "P?"


def _card_id(value: object) -> str:
    if isinstance(value, dict):
        card_id = value.get("id")
        if isinstance(card_id, str):
            return card_id
    return "unknown"


def _dict_from(value: object) -> dict[object, object]:
    return value if isinstance(value, dict) else {}


def _rotate_edge_names(edges: set[str], rotation: int) -> set[str]:
    normalized = rotation % 360
    if normalized == 0:
        return set(edges)
    if normalized == 180:
        opposite = {"N": "S", "E": "W", "S": "N", "W": "E"}
        return {opposite[edge] for edge in edges if edge in opposite}
    return set(edges)


def _game_dict(result: object) -> dict[str, object]:
    game = result.to_dict() if hasattr(result, "to_dict") else result
    if not isinstance(game, dict):
        raise TypeError("expected a GameResult or mapping")
    return game


def _debug_steps(game: dict[str, object]) -> list[dict[str, object]]:
    debug = game.get("debug", {})
    if not isinstance(debug, dict):
        return []
    steps = debug.get("steps", [])
    return [step for step in steps if isinstance(step, dict)] if isinstance(steps, list) else []


def _initial_public_board() -> dict[tuple[int, int], dict[str, object]]:
    board: dict[tuple[int, int], dict[str, object]] = {
        (0, 0): {
            "x": 0,
            "y": 0,
            "kind": "start",
            "rotation": 0,
            "revealed": True,
            "reachable": True,
            "card": dict(START_CARD),
        }
    }
    for goal_index, (x, y) in enumerate(GOAL_COORDS):
        board[(x, y)] = {
            "x": x,
            "y": y,
            "kind": "goal",
            "goal_index": goal_index,
            "rotation": 0,
            "revealed": False,
            "reachable": False,
            "goal_kind": None,
            "card": None,
        }
    return board


def _apply_public_event(
    board: dict[tuple[int, int], dict[str, object]],
    event: dict[str, object],
) -> None:
    action_type = event.get("action_type")
    if action_type == "play_path":
        x = event.get("x")
        y = event.get("y")
        card = event.get("card")
        if isinstance(x, int) and isinstance(y, int) and isinstance(card, dict):
            board[(x, y)] = {
                "x": x,
                "y": y,
                "kind": "path",
                "rotation": event.get("rotation", 0),
                "revealed": True,
                "reachable": True,
                "card": card,
            }
    elif action_type == "rockfall":
        x = event.get("x")
        y = event.get("y")
        if isinstance(x, int) and isinstance(y, int):
            board.pop((x, y), None)
    elif action_type == "reveal_goal":
        goal_index = event.get("goal_index")
        goal_kind = event.get("revealed_goal_kind")
        if isinstance(goal_index, int) and goal_index in range(len(GOAL_COORDS)):
            x, y = GOAL_COORDS[goal_index]
            card = event.get("card")
            if not isinstance(card, dict):
                card = dict(GOAL_CARD_FALLBACKS["gold" if goal_kind == "gold" else "stone_ne"])
            board[(x, y)] = {
                "x": x,
                "y": y,
                "kind": "goal",
                "goal_index": goal_index,
                "rotation": event.get("rotation", 0),
                "revealed": True,
                "reachable": True,
                "goal_kind": goal_kind,
                "card": card,
            }


def _sorted_tiles(board: dict[tuple[int, int], dict[str, object]]) -> list[dict[str, object]]:
    return [
        _copy_tile(tile)
        for _coord, tile in sorted(board.items(), key=lambda item: (item[0][1], item[0][0]))
    ]


def _copy_tile(tile: dict[str, object]) -> dict[str, object]:
    copied = dict(tile)
    card = copied.get("card")
    if isinstance(card, dict):
        copied["card"] = dict(card)
    return copied
