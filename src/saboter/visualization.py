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
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .shell {{
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      min-height: 100vh;
    }}
    aside {{
      background: var(--panel);
      border-right: 1px solid var(--line);
      padding: 18px;
      overflow: auto;
    }}
    main {{
      min-width: 0;
      padding: 18px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
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
      display: grid;
      grid-template-rows: auto minmax(180px, 320px) minmax(0, 1fr);
    }}
    .event-current {{
      border-bottom: 1px solid var(--line);
      padding: 12px;
      min-height: 68px;
    }}
    .event-current strong {{ display: block; margin-bottom: 4px; }}
    .event-current span {{ color: var(--muted); }}
    .debug-panel {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      overflow: auto;
      display: grid;
      gap: 10px;
      align-content: start;
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
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
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
          <div class="event-list" id="eventList"></div>
        </div>
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
    const slider = document.getElementById("stepSlider");
    const board = document.getElementById("board");
    const eventList = document.getElementById("eventList");

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

    function render(step) {{
      const snapshot = snapshots[step] || [];
      renderPlayers(step);
      const bounds = getBounds(snapshot);
      board.style.gridTemplateColumns = `repeat(${{bounds.width}}, 58px)`;
      board.replaceChildren();
      for (let y = bounds.minY; y <= bounds.maxY; y++) {{
        for (let x = bounds.minX; x <= bounds.maxX; x++) {{
          board.appendChild(renderTile(snapshot.find(tile => tile.x === x && tile.y === y)));
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
      [...eventList.children].forEach((row, index) => {{
        row.classList.toggle("active", index === step - 1);
      }});
    }}

    function renderPlayers(step) {{
      const players = document.getElementById("players");
      players.replaceChildren();
      game.agent_names.forEach((agent, id) => {{
        const finalRole = game.roles[String(id)] ?? game.roles[id] ?? "?";
        const finalReward = game.rewards[String(id)] ?? game.rewards[id] ?? "?";
        const role = finalRole;
        const reward = finalReward;
        const row = document.createElement("div");
        row.className = `player ${{role}}`;
        row.title = `P${{id}} | ${{agent}} | role ${{finalRole}} | reward ${{finalReward}}`;
        row.innerHTML = `
          <div class="player-id">P${{id}}</div>
          <div>
            <div class="player-agent">${{escapeText(agent)}}</div>
            <div class="player-role">${{escapeText(role)}}</div>
          </div>
          <strong>${{reward}}</strong>
        `;
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

      const discard = document.createElement("div");
      discard.innerHTML = `<div class="debug-title">Discard pile</div>`;
      const discardCards = document.createElement("div");
      discardCards.className = "discard-pile";
      const pile = debug.discard_pile_after || debug.discard_pile || [];
      if (!pile.length) discardCards.appendChild(chip("empty"));
      pile.forEach(card => discardCards.appendChild(chip(card ? cardName(card) : "face-down")));
      discard.appendChild(discardCards);
      panel.appendChild(discard);

      const hands = document.createElement("div");
      hands.innerHTML = `<div class="debug-title">Hands</div>`;
      Object.entries(debug.hands || {{}}).forEach(([playerId, hand]) => {{
        const row = document.createElement("div");
        row.className = "hand-row";
        const cards = document.createElement("div");
        cards.className = "cards";
        (hand || []).forEach((card, index) => cards.appendChild(chip(`${{index}}:${{cardName(card)}}`)));
        if (!cards.children.length) cards.appendChild(chip("empty"));
        row.appendChild(strong(`P${{playerId}}`));
        row.appendChild(cards);
        hands.appendChild(row);
      }});
      panel.appendChild(hands);

      const scored = Array.isArray(debug.legal_actions)
        ? debug.legal_actions.filter(action => typeof action.score === "number")
        : [];
      if (scored.length) {{
        const scores = document.createElement("div");
        scores.innerHTML = `<div class="debug-title">Legal action scores</div>`;
        scored
          .slice()
          .sort((left, right) => Number(right.score) - Number(left.score))
          .forEach(action => {{
            const row = document.createElement("div");
            row.className = `score-row${{action.selected ? " selected" : ""}}`;
            row.innerHTML = `<strong>${{escapeText(action.label)}}</strong><span class="score-value">score=${{Number(action.score).toFixed(3)}}</span>`;
            scores.appendChild(row);
          }});
        panel.appendChild(scores);
      }}
    }}

    function renderTile(tile) {{
      const div = document.createElement("div");
      if (!tile) {{
        div.className = "tile empty";
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
      return div;
    }}

    function getBounds(snapshot) {{
      if (!snapshot.length) return {{minX: 0, maxX: 0, minY: 0, maxY: 0, width: 1}};
      const xs = snapshot.map(tile => tile.x);
      const ys = snapshot.map(tile => tile.y);
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
