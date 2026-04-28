from __future__ import annotations

import base64
import json
import re
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = getLogger(__name__)

# Descriptions the VLM produces that carry zero signal.
_USELESS_DESCRIPTIONS: Set[str] = {
    "tool-based analysis",
    "unknown",
    "",
}


def _short_activity(activity: str) -> str:
    """Extract the simple class name from a fully-qualified Android activity."""

    if "/" in activity:
        activity = activity.split("/", 1)[1]
    return activity.rsplit(".", 1)[-1] if "." in activity else activity


def _friendly_activity(activity: str) -> str:
    """Convert an Android activity class into a readable name.

    ``SunriseContainerActivity`` → ``Container``,
    ``SunriseGenericActivity`` → ``Generic Screen``.
    """

    short = _short_activity(activity)
    # Strip common prefixes/suffixes
    friendly = re.sub(r"(?i)^sunrise", "", short)
    friendly = re.sub(r"(?i)activity$", "", friendly)
    friendly = friendly.strip()
    if not friendly:
        friendly = "Screen"
    # CamelCase → spaced
    friendly = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", friendly)
    return friendly.strip()


def _build_screen_names(graph_data: Dict[str, Any]) -> Dict[str, str]:
    """Infer a human-readable name for every node.

    Handles both activity-based nodes (new format) and visual_hash-based nodes (legacy format).

    For activity-based nodes:
    - Use the activity name (e.g., "com.example.app/.MainActivity")

    For visual_hash-based nodes:
    - Strategy (in priority order):
      1. Use the stored ``description`` if meaningful
      2. Infer from incoming edges
      3. Infer from outgoing edges
      4. Fall back to activity class name
    """

    nodes: List[Dict[str, Any]] = graph_data.get("nodes", [])
    edges: List[Dict[str, Any]] = graph_data.get("edges", [])

    # Determine if this is activity-based or visual_hash-based
    is_activity_based = any(n.get("activity") and not n.get("visual_hash") for n in nodes)

    names: Dict[str, str] = {}

    if is_activity_based:
        # New format: activity nodes
        for node in nodes:
            activity = node.get("activity", "")
            if activity:
                # Use friendly activity name
                names[activity] = _friendly_activity(activity)
        return names

    # Legacy format: visual_hash-based nodes
    # Index: hash → incoming non-back action targets
    incoming: Dict[str, List[str]] = defaultdict(list)
    # Index: hash → outgoing non-back action targets
    outgoing: Dict[str, List[str]] = defaultdict(list)

    for edge in edges:
        action_type = (edge.get("action_type") or "").lower()
        target = (edge.get("action_target") or "").strip()
        if not target or action_type == "back":
            continue
        dst = edge.get("destination_hash")
        src = edge.get("source_hash")
        if dst:
            incoming[dst].append(target)
        if src:
            outgoing[src].append(target)

    for node in nodes:
        vhash = node.get("visual_hash")
        if not vhash:
            continue

        # 1. Real description?
        desc = (node.get("description") or "").strip()
        if desc.lower() not in _USELESS_DESCRIPTIONS:
            names[vhash] = desc
            continue

        # 2. Infer from incoming edges
        inc = incoming.get(vhash, [])
        if inc:
            # Pick the most descriptive incoming label (longest)
            best = max(inc, key=len)
            names[vhash] = _clean_target_as_screen_name(best)
            continue

        # 3. Infer from outgoing edges (summarise the screen's content)
        # Take first 3 items in order (preserves frequency/appearance order)
        # rather than sorting by length, which doesn't correlate with descriptiveness
        out = outgoing.get(vhash, [])
        if out:
            items = out[:3]
            summary = ", ".join(_clean_target_as_screen_name(t) for t in items)
            names[vhash] = f"Screen: {summary}"
            continue

        # 4. Friendly activity name
        names[vhash] = _friendly_activity(node.get("activity", "unknown"))

    return names


def _clean_target_as_screen_name(target: str) -> str:
    """Turn an edge target like 'Messages tab' into a screen name 'Messages'.

    Strips trailing noise words: tab, button, card, section, item, icon, link,
    area, etc.  Capitalises the first letter for readability.
    """

    cleaned = target.strip()
    cleaned = re.sub(
        r"\s+(tab|button|card|section|item|icon|link|area|view|field|"
        r"list item|list|navigation|nav)$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip() or target.strip()
    # Title-case if entirely lowercase
    if cleaned == cleaned.lower():
        cleaned = cleaned.title()
    # Always capitalise first character
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _node_label(name: str, node: Dict[str, Any], *, max_len: int = 45) -> str:
    """Format the final node label for rendering.

    Shows the inferred name on the first line, and a small subtitle with the
    short activity + hash on the second.
    """

    short_act = _friendly_activity(node.get("activity", "unknown"))
    short_hash = node["visual_hash"][:6]

    if len(name) > max_len:
        name = name[: max_len - 1] + "\u2026"

    return f"{name}\n({short_act} \u00b7 {short_hash})"


def _edge_label(edge: Dict[str, Any], *, max_target: int = 28) -> str:
    """Build a concise, natural-language edge label."""

    action_type = (edge.get("action_type") or "?").strip()
    target = (edge.get("action_target") or "").strip()

    if target:
        if len(target) > max_target:
            target = target[: max_target - 1] + "\u2026"
        label = f"{action_type} \u2192 {target}"
    else:
        label = action_type

    count = edge.get("count", 1)
    if count > 1:
        label += f" (\u00d7{count})"
    return label


# Dark-theme palette; all tones are dark enough for white text.
_ACTIVITY_PALETTE: tuple[str, ...] = (
    "#1f6feb",
    "#da3633",
    "#bf4b00",
    "#8957e5",
    "#238636",
    "#9e6a03",
    "#0e7490",
    "#6e7681",
    "#6639ba",
    "#57ab5a",
    "#bf5af2",
    "#d29922",
)


def _activity_color_map(nodes: List[Dict[str, Any]]) -> Dict[str, str]:
    """Deterministic activity → hex color mapping, cycling through the palette."""

    activities = sorted({(n.get("activity") or "unknown") for n in nodes})
    return {act: _ACTIVITY_PALETTE[i % len(_ACTIVITY_PALETTE)] for i, act in enumerate(activities)}


# Screenshot filename format (see infrastructure/storage/local.py):
#   {YYYYMMDD_HHMMSS}__{activity_sanitized}.png
# activity_sanitized strips characters outside [A-Za-z0-9._-] — notably the
# `/` between package and activity class is dropped, so we compare activities
# after the same normalization.
_SCREENSHOT_NAME_RE = re.compile(r"^(\d{8}_\d{6})__(.+?)\.png$")


def _sanitize_activity(activity: str) -> str:
    """Mirror ``LocalImageStorage``'s filename-safe sanitization."""

    return "".join(c for c in activity if c.isalnum() or c in "._-")


def _build_screenshot_index(
    screenshots_root: Path,
) -> Dict[str, List[Tuple[int, Path]]]:
    """Scan screenshots into a ``sanitized_activity → [(epoch, path), ...]`` index.

    Sorted ascending by epoch so we can binary-search later if it ever matters;
    for now we linear-scan candidates — the lists are tiny (tens of items).
    """

    from datetime import datetime

    index: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    if not screenshots_root.exists():
        return index

    for path in screenshots_root.rglob("*.png"):
        match = _SCREENSHOT_NAME_RE.match(path.name)
        if not match:
            continue
        ts_str, activity = match.groups()
        try:
            # Local time, same as LocalImageStorage.save()
            epoch = int(datetime.strptime(ts_str, "%Y%m%d_%H%M%S").timestamp())
        except ValueError:
            continue
        index[activity].append((epoch, path))

    for candidates in index.values():
        candidates.sort(key=lambda pair: pair[0])
    return index


def _match_screenshot(
    node: Dict[str, Any],
    index: Dict[str, List[Tuple[int, Path]]],
    *,
    tolerance_sec: int = 600,
) -> Optional[Path]:
    """Find the screenshot file closest in time to the node's ``first_seen``.

    Returns the closest match whose activity matches and whose timestamp is
    within ``tolerance_sec`` of ``first_seen``.  Returns ``None`` if nothing
    qualifies.
    """

    activity = node.get("activity")
    first_seen = node.get("first_seen")
    if not activity or not first_seen:
        return None

    candidates = index.get(_sanitize_activity(activity))
    if not candidates:
        return None

    best_path: Optional[Path] = None
    best_delta = tolerance_sec + 1
    for epoch, path in candidates:
        delta = abs(epoch - int(first_seen))
        if delta < best_delta:
            best_delta = delta
            best_path = path
    return best_path if best_delta <= tolerance_sec else None


def _encode_thumbnail(path: Path, *, max_dimension: int = 360, quality: int = 70) -> Optional[str]:
    """Return a ``data:image/jpeg;base64,...`` URI for a resized thumbnail.

    Returns ``None`` on any failure so the caller can fall back gracefully.
    """

    try:
        from fathom.utils.image import ImageProcessor
    except ImportError:  # pragma: no cover — package always ships PIL
        return None

    try:
        raw = path.read_bytes()
    except OSError:
        return None

    jpeg = ImageProcessor.optimize_for_vision(raw, max_dimension=max_dimension, quality=quality)
    if not jpeg:
        return None

    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def _build_thumbnails(
    nodes: Iterable[Dict[str, Any]],
    screenshots_root: Optional[Path],
) -> Dict[str, str]:
    """Build a ``{node_id → data URI}`` map of thumbnails for the given nodes.

    ``node_id`` is the visual_hash (legacy format) or activity (new format),
    matching the key used by the HTML viewer.  Silently skips nodes without a
    matching screenshot.
    """

    if screenshots_root is None:
        return {}

    index = _build_screenshot_index(screenshots_root)
    if not index:
        return {}

    out: Dict[str, str] = {}
    for node in nodes:
        node_id = node.get("visual_hash") or node.get("activity")
        if not node_id:
            continue
        path = _match_screenshot(node, index)
        if path is None:
            continue
        thumb = _encode_thumbnail(path)
        if thumb:
            out[node_id] = thumb
    return out


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --border: #30363d;
    --text: #f0f6fc;
    --muted: #8b949e;
    --accent: #58a6ff;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
  }
  #app {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  #main {
    display: flex;
    flex: 1 1 auto;
    min-height: 0;
  }
  header {
    flex: 0 0 auto;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    background: var(--panel);
  }
  header h1 { font-size: 14px; margin: 0; font-weight: 600; }
  header .controls { margin-left: auto; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  header .path-controls { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; padding-left: 8px; border-left: 1px solid var(--border); }
  header .path-controls label { color: var(--muted); font-size: 11px; display: inline-flex; gap: 4px; align-items: center; }
  .combo { position: relative; display: inline-block; }
  .combo input { padding-right: 26px !important; }
  .combo::after {
    content: ""; position: absolute; right: 10px; top: 50%; transform: translateY(-25%);
    width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid var(--muted); pointer-events: none;
  }
  .combo-menu {
    display: none; position: absolute; z-index: 1000;
    top: calc(100% + 2px); left: 0; min-width: 100%; max-width: 460px;
    max-height: 320px; overflow-y: auto;
    background: var(--panel); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45);
    font-size: 12px;
  }
  .combo-menu.open { display: block; }
  .combo-menu .opt {
    padding: 6px 10px; cursor: pointer; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
  }
  .combo-menu .opt:hover, .combo-menu .opt.active { background: var(--bg); color: var(--accent); }
  .combo-menu .opt mark { background: rgba(88, 166, 255, 0.25); color: inherit; padding: 0; border-radius: 2px; }
  .combo-menu .empty { padding: 10px; color: var(--muted); font-style: italic; }
  header input[type="search"], header input[type="text"] {
    background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 10px; font-size: 12px; min-width: 260px;
    outline: none;
  }
  header input[type="search"] { min-width: 240px; }
  header input[type="search"]:focus, header input[type="text"]:focus { border-color: var(--accent); }
  header button {
    background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 10px; font-size: 12px; cursor: pointer;
  }
  header button:hover { border-color: var(--accent); }
  header button.active { border-color: var(--accent); color: var(--accent); }
  #canvas {
    flex: 1 1 auto;
    min-width: 0; min-height: 0;
    background: var(--bg);
    position: relative;
  }
  #sidebar {
    flex: 0 0 380px;
    border-left: 1px solid var(--border);
    background: var(--panel);
    overflow-y: auto;
    padding: 16px;
  }
  #sidebar img.thumb {
    display: block; width: 100%; max-height: 320px;
    object-fit: contain;
    background: #000;
    border: 1px solid var(--border); border-radius: 6px;
    margin: 6px 0;
  }
  .tabbar {
    display: flex; gap: 4px; margin: -4px -4px 12px;
    border-bottom: 1px solid var(--border);
  }
  .tab {
    background: transparent; color: var(--muted);
    border: none; border-bottom: 2px solid transparent;
    padding: 8px 12px; font-size: 12px; cursor: pointer;
    font-weight: 500;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab .count { color: var(--muted); font-size: 10px; margin-left: 4px; font-weight: 400; }
  .tab.active .count { color: var(--accent); }
  .pane { display: none; }
  .pane.active { display: block; }
  #paths-summary { color: var(--muted); font-size: 11px; margin-bottom: 8px; }
  #paths-list { display: flex; flex-direction: column; gap: 4px; }
  .path-row {
    padding: 8px 10px; border-radius: 6px; cursor: pointer;
    background: var(--bg); border: 1px solid var(--border);
    font-size: 11px; line-height: 1.5;
  }
  .path-row:hover { border-color: var(--accent); }
  .path-row.selected { border-color: var(--accent); background: rgba(88, 166, 255, 0.08); }
  .path-row .hdr { color: var(--accent); font-weight: 600; margin-bottom: 3px; font-size: 12px; }
  .path-row .step {
    color: var(--text); word-break: break-word;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 10.5px;
  }
  .path-row .arrow { color: var(--muted); margin: 0 2px; }
  #sidebar h2 { font-size: 14px; margin: 0 0 8px; color: var(--accent); }
  #sidebar h3 {
    font-size: 11px; margin: 14px 0 4px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.6px;
  }
  #sidebar .meta {
    color: var(--muted); font-size: 11px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    word-break: break-all; margin: 2px 0;
  }
  #sidebar .desc { margin: 6px 0; line-height: 1.5; }
  #sidebar pre {
    background: var(--bg); padding: 10px; border-radius: 6px;
    border: 1px solid var(--border); font-size: 11px; line-height: 1.45;
    white-space: pre-wrap; word-break: break-word;
    max-height: 420px; overflow-y: auto; margin: 4px 0;
  }
  #sidebar .empty { color: var(--muted); font-style: italic; }
  #legend { display: flex; flex-wrap: wrap; gap: 6px; font-size: 10px; }
  #legend .chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 8px; background: var(--bg);
    border: 1px solid var(--border); border-radius: 11px;
  }
  #legend .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  footer {
    flex: 0 0 auto;
    padding: 8px 16px; border-top: 1px solid var(--border);
    color: var(--muted); font-size: 11px;
    display: flex; justify-content: space-between; gap: 12px;
    background: var(--panel);
  }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1 id="title">{{TITLE}}</h1>
    <div class="controls">
      <input id="search" type="search" placeholder="Filter by activity or description…">
      <button id="fit" title="Fit all nodes to view">Fit</button>
      <button id="reset" title="Clear filters and re-fit">Reset</button>
      <button id="physics" class="active" title="Toggle force simulation">Physics: on</button>
    </div>
    <div class="path-controls">
      <div class="combo">
        <input id="start" type="text" placeholder="Start: activity or keyword…" autocomplete="off">
        <div class="combo-menu" id="start-menu"></div>
      </div>
      <div class="combo">
        <input id="goal" type="text" placeholder="Goal: activity or keyword…" autocomplete="off">
        <div class="combo-menu" id="goal-menu"></div>
      </div>
      <button id="find-paths" title="Find all paths from Start to Goal">Find paths</button>
      <button id="clear-paths" title="Clear highlight">Clear</button>
      <label title="Include back-navigation edges in path search"><input id="allow-back" type="checkbox"> back edges</label>
    </div>
  </header>
  <div id="main">
    <div id="canvas"></div>
    <aside id="sidebar">
      <div class="tabbar">
        <button class="tab active" data-pane="pane-details">Details</button>
        <button class="tab" data-pane="pane-paths">Paths<span id="paths-count" class="count"></span></button>
      </div>
      <div id="pane-details" class="pane active">
        <div id="details"><p class="empty">Click a node or edge to see details.</p></div>
        <h3>Legend · activity</h3>
        <div id="legend"></div>
      </div>
      <div id="pane-paths" class="pane">
        <div id="paths-summary">Pick a <b>Start</b> and <b>Goal</b> screen above, then press <b>Find paths</b>.</div>
        <div id="paths-list"></div>
      </div>
    </aside>
  </div>
  <footer>
    <span id="stats"></span>
    <span>Click to inspect · scroll to zoom · drag to pan</span>
  </footer>
</div>

<script id="fathom-data" type="application/json">{{DATA}}</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("fathom-data").textContent);
  const { graph, meta } = DATA;
  const { screen_names, activity_colors, is_activity_based, title } = meta;
  const thumbnails = meta.thumbnails || {};

  document.getElementById("title").textContent = title;

  function nodeId(n) { return is_activity_based ? n.activity : n.visual_hash; }
  function edgeSrc(e) { return is_activity_based ? e.source_activity : e.source_hash; }
  function edgeDst(e) { return is_activity_based ? e.destination_activity : e.destination_hash; }
  function nodeColor(activity) { return activity_colors[activity] || "#4488ff"; }
  function friendlyActivity(activity) {
    if (!activity) return "Screen";
    const short = activity.includes("/") ? activity.split("/")[1] : activity;
    const cls = short.includes(".") ? short.split(".").pop() : short;
    const trimmed = cls.replace(/^Sunrise/i, "").replace(/Activity$/i, "");
    return trimmed.replace(/(?<=[a-z])(?=[A-Z])/g, " ").trim() || "Screen";
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }
  function fmtTime(t) {
    if (!t) return "";
    try { return new Date(t * 1000).toISOString().replace("T", " ").slice(0, 19); }
    catch (e) { return String(t); }
  }

  // Build vis-network datasets.
  const visNodes = [];
  const rawById = {};
  graph.nodes.forEach((n) => {
    const id = nodeId(n);
    if (!id) return;
    rawById[id] = n;
    const name = screen_names[id] || id.slice(0, 10);
    const visits = n.visit_count || 0;
    const color = nodeColor(n.activity || id);
    visNodes.push({
      id,
      label: name + "\n(visits: " + visits + ")",
      shape: "box",
      color: {
        background: color,
        border: "#c9d1d9",
        highlight: { background: color, border: "#58a6ff" },
      },
      borderWidth: 1.5,
      font: {
        color: "#ffffff",
        size: 11,
        face: "-apple-system",
        strokeWidth: 2,
        strokeColor: "#0d1117",
      },
      margin: 8,
      widthConstraint: { maximum: 180 },
      value: Math.max(1, visits),
    });
  });

  const visEdges = [];
  const edgeRaw = {};
  graph.edges.forEach((e, i) => {
    const from = edgeSrc(e);
    const to = edgeDst(e);
    if (!from || !to) return;
    if (!rawById[from] || !rawById[to]) return;
    const action = e.action_type || "";
    const target = e.action_target || "";
    let label = target ? action + " \u2192 " + target : action;
    if ((e.count || 1) > 1) label += " (\u00d7" + e.count + ")";
    if (label.length > 36) label = label.slice(0, 35) + "\u2026";
    const eid = "e" + i;
    edgeRaw[eid] = e;
    visEdges.push({
      id: eid,
      from, to,
      label,
      arrows: { to: { enabled: true, scaleFactor: 0.6 } },
      color: { color: "#58a6ff", opacity: 0.65, highlight: "#79c0ff" },
      font: {
        color: "#c9d1d9",
        size: 10,
        face: "-apple-system",
        strokeWidth: 3,
        strokeColor: "#0d1117",
        align: "horizontal",
      },
      smooth: { type: "curvedCW", roundness: 0.15 },
      width: 1.1,
    });
  });

  const container = document.getElementById("canvas");
  const dataset = {
    nodes: new vis.DataSet(visNodes),
    edges: new vis.DataSet(visEdges),
  };
  const options = {
    physics: {
      enabled: true,
      solver: "forceAtlas2Based",
      forceAtlas2Based: {
        gravitationalConstant: -70,
        centralGravity: 0.012,
        springLength: 140,
        springConstant: 0.08,
        damping: 0.5,
      },
      stabilization: { iterations: 250 },
    },
    interaction: {
      hover: true,
      tooltipDelay: 200,
      navigationButtons: false,
      multiselect: false,
    },
    nodes: { scaling: { min: 10, max: 40 } },
    edges: { selectionWidth: 2 },
  };
  const network = new vis.Network(container, dataset, options);
  network.once("stabilizationIterationsDone", () => {
    network.fit({ animation: { duration: 400 } });
  });
  // Belt-and-braces: re-fit after the tab has had a tick to lay out.
  setTimeout(() => network.fit(), 150);
  // Debug handle: inspect the live network + datasets from the console.
  window.fathomGraph = { network, dataset, graph, meta };

  // Sidebar rendering.
  const details = document.getElementById("details");
  function clearDetails() {
    details.innerHTML = '<p class="empty">Click a node or edge to see details.</p>';
  }
  function renderNode(id) {
    const n = rawById[id];
    if (!n) return;
    const name = screen_names[id] || id;
    const activity = n.activity ? friendlyActivity(n.activity) : "";
    const parts = ['<h2>' + escapeHtml(name) + '</h2>'];
    if (activity) parts.push('<div class="meta">' + escapeHtml(activity) + '</div>');
    parts.push('<div class="meta">id: ' + escapeHtml(id) + '</div>');
    const thumb = thumbnails[id];
    if (thumb) parts.push('<img class="thumb" src="' + thumb + '" alt="Screenshot of ' + escapeHtml(name) + '">');
    parts.push('<h3>Visits</h3><div>' + (n.visit_count || 0) + '</div>');
    if (n.first_seen) parts.push('<h3>First seen</h3><div class="meta">' + fmtTime(n.first_seen) + '</div>');
    if (n.last_seen && n.last_seen !== n.first_seen) parts.push('<h3>Last seen</h3><div class="meta">' + fmtTime(n.last_seen) + '</div>');
    if (n.description) parts.push('<h3>Description</h3><div class="desc">' + escapeHtml(n.description) + '</div>');
    if (n.rich_description) parts.push('<h3>Rich description</h3><pre>' + escapeHtml(n.rich_description) + '</pre>');
    details.innerHTML = parts.join("");
  }
  function renderEdge(id) {
    const e = edgeRaw[id];
    if (!e) return;
    const action = e.action_type || "action";
    const target = e.action_target || "";
    const parts = ['<h2>' + escapeHtml(action) + '</h2>'];
    if (target) parts.push('<div class="desc">\u2192 ' + escapeHtml(target) + '</div>');
    parts.push('<div class="meta">from: ' + escapeHtml(edgeSrc(e)) + '</div>');
    parts.push('<div class="meta">to: ' + escapeHtml(edgeDst(e)) + '</div>');
    parts.push('<h3>Count</h3><div>' + (e.count || 1) + '</div>');
    if (e.coord_bucket) parts.push('<h3>Coord bucket</h3><div class="meta">' + escapeHtml(e.coord_bucket) + '</div>');
    if (e.first_seen) parts.push('<h3>First seen</h3><div class="meta">' + fmtTime(e.first_seen) + '</div>');
    if (e.last_seen && e.last_seen !== e.first_seen) parts.push('<h3>Last seen</h3><div class="meta">' + fmtTime(e.last_seen) + '</div>');
    details.innerHTML = parts.join("");
  }

  network.on("click", (params) => {
    if (params.nodes.length) renderNode(params.nodes[0]);
    else if (params.edges.length) renderEdge(params.edges[0]);
    else clearDetails();
  });

  // Controls.
  document.getElementById("fit").onclick = () => network.fit({ animation: { duration: 400 } });
  const physicsBtn = document.getElementById("physics");
  let physicsOn = true;
  physicsBtn.onclick = () => {
    physicsOn = !physicsOn;
    network.setOptions({ physics: { enabled: physicsOn } });
    physicsBtn.textContent = "Physics: " + (physicsOn ? "on" : "off");
    physicsBtn.classList.toggle("active", physicsOn);
  };

  // Filter: hide nodes whose activity/description/name doesn't match query.
  const search = document.getElementById("search");
  function applyFilter(q) {
    const needle = q.trim().toLowerCase();
    const visibleIds = new Set();
    graph.nodes.forEach((n) => {
      const id = nodeId(n);
      if (!id) return;
      if (!needle) { visibleIds.add(id); return; }
      const hay = [n.activity, n.description, screen_names[id]]
        .filter(Boolean).join(" ").toLowerCase();
      if (hay.includes(needle)) visibleIds.add(id);
    });
    dataset.nodes.getIds().forEach((id) =>
      dataset.nodes.update({ id, hidden: !visibleIds.has(id) }));
    dataset.edges.forEach((e) => {
      const visible = visibleIds.has(e.from) && visibleIds.has(e.to);
      dataset.edges.update({ id: e.id, hidden: !visible });
    });
  }
  search.addEventListener("input", () => applyFilter(search.value));

  document.getElementById("reset").onclick = () => {
    search.value = "";
    applyFilter("");
    clearDetails();
    network.fit({ animation: { duration: 400 } });
  };

  // Legend (sorted by activity label for readability).
  const legend = document.getElementById("legend");
  Object.entries(activity_colors)
    .sort((a, b) => friendlyActivity(a[0]).localeCompare(friendlyActivity(b[0])))
    .forEach(([act, color]) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.innerHTML =
        '<span class="dot" style="background:' + color + '"></span>' +
        escapeHtml(friendlyActivity(act));
      chip.title = act;
      legend.appendChild(chip);
    });

  // Footer stats.
  const stats = graph.stats || {};
  const screens = stats.unique_screens != null ? stats.unique_screens : graph.nodes.length;
  const trans = stats.total_transitions != null ? stats.total_transitions : graph.edges.length;
  const acts = stats.unique_activities != null
    ? stats.unique_activities
    : Object.keys(activity_colors).length;
  document.getElementById("stats").textContent =
    screens + " screens \u00b7 " + trans + " transitions \u00b7 " + acts + " activities";

  // ── Sidebar tabs ─────────────────────────────────────────────
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".pane").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      const pane = document.getElementById(tab.dataset.pane);
      if (pane) pane.classList.add("active");
    });
  });

  // ── Path-finding ─────────────────────────────────────────────
  // Build combobox options: "<Activity>: <short name> (hash8)" → node id.
  // The label is shaped so users can filter by activity (type "Home") or
  // by any word in the description (type "Facebook", "checkout", …).
  const MAX_LABEL_NAME = 50;
  const labelToId = new Map();
  visNodes.forEach((n) => {
    const raw = rawById[n.id] || {};
    const fullName = screen_names[n.id] || String(n.id).slice(0, 10);
    const shortName = fullName.length > MAX_LABEL_NAME
      ? fullName.slice(0, MAX_LABEL_NAME - 1) + "\u2026"
      : fullName;
    const act = raw.activity ? friendlyActivity(raw.activity) : "";
    const shortId = String(n.id).slice(0, 8);
    const prefix = act ? act + ": " : "";
    let label = prefix + shortName + " (" + shortId + ")";
    let attempt = 2;
    while (labelToId.has(label)) {
      label = prefix + shortName + " (" + shortId + "#" + attempt + ")";
      attempt += 1;
    }
    labelToId.set(label, n.id);
  });
  const allLabels = [...labelToId.keys()].sort((a, b) => a.localeCompare(b));

  function resolveInput(value) {
    if (!value) return null;
    return labelToId.get(value.trim()) || null;
  }

  // Minimal searchable dropdown bound to an <input> + sibling .combo-menu.
  // Shows all options on focus/click; filters by case-insensitive substring
  // as the user types.  Arrow keys + Enter navigate the menu.
  function attachCombo(inputId, menuId) {
    const input = document.getElementById(inputId);
    const menu = document.getElementById(menuId);
    let activeIndex = -1;
    let visibleLabels = allLabels;

    function render(query) {
      const needle = query.trim().toLowerCase();
      visibleLabels = needle
        ? allLabels.filter((l) => l.toLowerCase().includes(needle))
        : allLabels;
      menu.innerHTML = "";
      if (!visibleLabels.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No matches";
        menu.appendChild(empty);
        return;
      }
      visibleLabels.slice(0, 200).forEach((label, i) => {
        const row = document.createElement("div");
        row.className = "opt" + (i === activeIndex ? " active" : "");
        // Highlight the matched substring for readability
        if (needle) {
          const idx = label.toLowerCase().indexOf(needle);
          row.innerHTML =
            escapeHtml(label.slice(0, idx)) +
            "<mark>" + escapeHtml(label.slice(idx, idx + needle.length)) + "</mark>" +
            escapeHtml(label.slice(idx + needle.length));
        } else {
          row.textContent = label;
        }
        row.addEventListener("mousedown", (ev) => {
          ev.preventDefault();
          input.value = label;
          close();
          input.dispatchEvent(new Event("input"));
        });
        menu.appendChild(row);
      });
    }

    function open() { menu.classList.add("open"); render(input.value); }
    function close() { menu.classList.remove("open"); activeIndex = -1; }

    input.addEventListener("focus", open);
    input.addEventListener("click", open);
    input.addEventListener("input", () => { activeIndex = -1; render(input.value); });
    input.addEventListener("blur", () => setTimeout(close, 120));
    input.addEventListener("keydown", (ev) => {
      if (!menu.classList.contains("open")) { if (ev.key !== "Escape") open(); }
      const rows = menu.querySelectorAll(".opt");
      if (ev.key === "ArrowDown") {
        ev.preventDefault();
        activeIndex = Math.min(rows.length - 1, activeIndex + 1);
      } else if (ev.key === "ArrowUp") {
        ev.preventDefault();
        activeIndex = Math.max(0, activeIndex - 1);
      } else if (ev.key === "Enter" && activeIndex >= 0 && rows[activeIndex]) {
        ev.preventDefault();
        input.value = rows[activeIndex].textContent;
        close();
        return;
      } else if (ev.key === "Escape") {
        close();
        return;
      } else {
        return;
      }
      rows.forEach((r, i) => r.classList.toggle("active", i === activeIndex));
      if (rows[activeIndex]) rows[activeIndex].scrollIntoView({ block: "nearest" });
    });
  }
  attachCombo("start", "start-menu");
  attachCombo("goal", "goal-menu");

  // Snapshot starting styles so highlight can be reverted cleanly.
  const initialNodeStyles = visNodes.map((n) => ({
    id: n.id, opacity: 1, borderWidth: n.borderWidth, color: n.color,
  }));
  const initialEdgeStyles = visEdges.map((e) => ({
    id: e.id, color: e.color, width: e.width,
  }));

  function buildAdjacency(includeBack) {
    const adj = new Map();
    graph.edges.forEach((e, i) => {
      const src = edgeSrc(e);
      const dst = edgeDst(e);
      if (!src || !dst) return;
      if (!includeBack && String(e.action_type || "").toLowerCase() === "back") return;
      if (!adj.has(src)) adj.set(src, []);
      adj.get(src).push({ to: dst, edgeId: "e" + i, edge: e });
    });
    return adj;
  }

  function findAllPaths(start, goal, adj, maxDepth, maxPaths) {
    if (start === goal) return { paths: [], sameNode: true, capped: false };
    const results = [];
    let capped = false;
    const visited = new Set([start]);
    const nodesPath = [start];
    const edgesPath = [];
    const stack = [{ neighbors: adj.get(start) || [], idx: 0 }];
    while (stack.length) {
      if (results.length >= maxPaths) { capped = true; break; }
      const top = stack[stack.length - 1];
      if (top.idx >= top.neighbors.length) {
        stack.pop();
        const popped = nodesPath.pop();
        if (popped !== undefined) visited.delete(popped);
        edgesPath.pop();
        continue;
      }
      const step = top.neighbors[top.idx++];
      if (step.to === goal) {
        results.push({
          nodes: [...nodesPath, step.to],
          edges: [...edgesPath, { id: step.edgeId, raw: step.edge }],
        });
        continue;
      }
      if (visited.has(step.to)) continue;
      if (nodesPath.length >= maxDepth) continue;
      visited.add(step.to);
      nodesPath.push(step.to);
      edgesPath.push({ id: step.edgeId, raw: step.edge });
      stack.push({ neighbors: adj.get(step.to) || [], idx: 0 });
    }
    return { paths: results, sameNode: false, capped };
  }

  function renderStep(edge) {
    const raw = edge.raw;
    const action = escapeHtml(raw.action_type || "");
    const target = escapeHtml(raw.action_target || "");
    return target ? action + ' "' + target + '"' : action;
  }

  function pathNodeLabel(id) {
    return screen_names[id] || String(id).slice(0, 10);
  }

  const MAX_DEPTH = 10;
  const MAX_PATHS = 50;
  let currentPath = null;

  function renderPaths(result) {
    const sum = document.getElementById("paths-summary");
    const list = document.getElementById("paths-list");
    list.innerHTML = "";
    currentPath = null;
    if (result.sameNode) {
      sum.textContent = "No paths found — Start and Goal are the same screen.";
      return;
    }
    if (!result.paths.length) {
      sum.textContent = "No paths found within depth " + MAX_DEPTH + ".";
      return;
    }
    // Caller is responsible for ordering; renderPaths preserves it.
    const prefix = result.capped ? "Showing first " : "Found ";
    const suffix = result.capped ? "+ paths" : (" path" + (result.paths.length === 1 ? "" : "s"));
    sum.textContent = prefix + result.paths.length + suffix + " · depth \u2264 " + MAX_DEPTH;
    result.paths.forEach((p, i) => {
      const row = document.createElement("div");
      row.className = "path-row";
      const hopWord = p.edges.length === 1 ? "hop" : "hops";
      const head = "Path " + (i + 1) + " \u00b7 " + p.edges.length + " " + hopWord +
        " \u00b7 " + escapeHtml(pathNodeLabel(p.nodes[0])) + " \u2192 " +
        escapeHtml(pathNodeLabel(p.nodes[p.nodes.length - 1]));
      const steps = p.edges.length
        ? p.edges.map(renderStep).join('<span class="arrow">\u2192</span>')
        : "(no steps)";
      row.innerHTML = '<div class="hdr">' + head + '</div><div class="step">' + steps + '</div>';
      row.addEventListener("click", () => {
        document.querySelectorAll(".path-row").forEach((r) => r.classList.remove("selected"));
        row.classList.add("selected");
        highlightPath(p);
      });
      list.appendChild(row);
    });
  }

  function highlightPath(p) {
    currentPath = p;
    const nodeSet = new Set(p.nodes);
    const edgeSet = new Set(p.edges.map((e) => e.id));
    const nodeUpdates = initialNodeStyles.map((ns) => {
      const inPath = nodeSet.has(ns.id);
      return {
        id: ns.id,
        opacity: inPath ? 1 : 0.12,
        borderWidth: inPath ? 3 : ns.borderWidth,
        color: inPath
          ? Object.assign({}, ns.color, { border: "#ffffff" })
          : ns.color,
        hidden: false,
      };
    });
    dataset.nodes.update(nodeUpdates);
    const edgeUpdates = initialEdgeStyles.map((es) => {
      const inPath = edgeSet.has(es.id);
      return {
        id: es.id,
        color: inPath
          ? { color: "#58a6ff", opacity: 1, highlight: "#79c0ff" }
          : { color: "#58a6ff", opacity: 0.05, highlight: "#58a6ff" },
        width: inPath ? 2.6 : 1.1,
        hidden: false,
      };
    });
    dataset.edges.update(edgeUpdates);
    network.fit({ nodes: [...nodeSet], animation: { duration: 400 } });
  }

  function resetHighlight() {
    if (currentPath === null) return;
    currentPath = null;
    dataset.nodes.update(initialNodeStyles.map((s) => Object.assign({}, s, { hidden: false })));
    dataset.edges.update(initialEdgeStyles.map((s) => Object.assign({}, s, { hidden: false })));
  }

  // Enumerate maximal simple journeys from root-like nodes — paths that can't
  // be extended without revisiting a node or exceeding maxDepth.  Sort by
  // length desc, keep the top `maxResults`.
  function discoverMaximalJourneys(adj, allNodeIds, maxDepth, maxResults) {
    if (!adj.size) return [];
    const inDeg = new Map();
    allNodeIds.forEach((id) => inDeg.set(id, 0));
    adj.forEach((neighbors) => {
      neighbors.forEach((n) => inDeg.set(n.to, (inDeg.get(n.to) || 0) + 1));
    });
    // Roots: out-degree > 0 and in-degree 0 (true entries).
    let roots = allNodeIds.filter((id) => adj.has(id) && (inDeg.get(id) || 0) === 0);
    if (!roots.length) {
      // Fallback: 5 nodes with outbound edges and the lowest in-degree.
      roots = allNodeIds
        .filter((id) => adj.has(id))
        .sort((a, b) => (inDeg.get(a) || 0) - (inDeg.get(b) || 0))
        .slice(0, 5);
    }

    const hardCap = maxResults * 5;
    const all = [];
    outer: for (const start of roots) {
      const visited = new Set([start]);
      const nodesPath = [start];
      const edgesPath = [];
      const stack = [{ neighbors: adj.get(start) || [], idx: 0, hadDescent: false }];
      while (stack.length) {
        if (all.length >= hardCap) break outer;
        const top = stack[stack.length - 1];
        if (top.idx >= top.neighbors.length) {
          // If we never descended from here, this frame is terminal.  Emit
          // the path (skip single-node paths — those aren't journeys).
          if (!top.hadDescent && edgesPath.length) {
            all.push({
              nodes: nodesPath.slice(),
              edges: edgesPath.slice(),
            });
          }
          stack.pop();
          const popped = nodesPath.pop();
          if (popped !== undefined) visited.delete(popped);
          edgesPath.pop();
          continue;
        }
        const step = top.neighbors[top.idx++];
        if (visited.has(step.to)) continue;
        if (nodesPath.length >= maxDepth) continue;
        top.hadDescent = true;
        visited.add(step.to);
        nodesPath.push(step.to);
        edgesPath.push({ id: step.edgeId, raw: step.edge });
        stack.push({ neighbors: adj.get(step.to) || [], idx: 0, hadDescent: false });
      }
    }

    // Dedupe: drop any path whose node sequence is a strict prefix of another.
    all.sort((a, b) => b.nodes.length - a.nodes.length);
    const kept = [];
    for (const p of all) {
      const prefixed = kept.some((q) =>
        q.nodes.length > p.nodes.length &&
        p.nodes.every((n, i) => q.nodes[i] === n),
      );
      if (!prefixed) kept.push(p);
      if (kept.length >= maxResults) break;
    }
    return kept;
  }

  function renderDiscoveredJourneys() {
    const t0 = performance.now();
    const adj = buildAdjacency(false);
    const allNodeIds = graph.nodes.map((n) => nodeId(n)).filter(Boolean);
    const journeys = discoverMaximalJourneys(adj, allNodeIds, MAX_DEPTH, 30);
    renderPaths({ paths: journeys, sameNode: false, capped: false });
    const elapsed = Math.round(performance.now() - t0);
    const sum = document.getElementById("paths-summary");
    const badge = document.getElementById("paths-count");
    if (!journeys.length) {
      sum.textContent = "No journeys discovered in the graph.";
      if (badge) badge.textContent = "";
      return;
    }
    sum.textContent = "Discovered " + journeys.length +
      " journey" + (journeys.length === 1 ? "" : "s") +
      " \u00b7 longest first \u00b7 click a row to highlight \u00b7 pick Start/Goal above to filter" +
      " \u00b7 " + elapsed + " ms";
    if (badge) badge.textContent = "(" + journeys.length + ")";
  }

  document.getElementById("find-paths").addEventListener("click", () => {
    const sum = document.getElementById("paths-summary");
    const list = document.getElementById("paths-list");
    const startId = resolveInput(document.getElementById("start").value);
    const goalId = resolveInput(document.getElementById("goal").value);
    document.querySelector('.tab[data-pane="pane-paths"]').click();
    if (!startId || !goalId) {
      sum.textContent = "Pick Start and Goal screens from the dropdown.";
      list.innerHTML = "";
      return;
    }
    const includeBack = document.getElementById("allow-back").checked;
    const adj = buildAdjacency(includeBack);
    const t0 = performance.now();
    const result = findAllPaths(startId, goalId, adj, MAX_DEPTH, MAX_PATHS);
    const elapsed = Math.round(performance.now() - t0);
    // Shortest first for pair search — fastest route feels most informative.
    result.paths.sort((a, b) => a.edges.length - b.edges.length);
    renderPaths(result);
    if (result.paths.length || result.sameNode) {
      sum.textContent += " · " + elapsed + " ms";
    }
  });

  document.getElementById("clear-paths").addEventListener("click", () => {
    document.querySelectorAll(".path-row").forEach((r) => r.classList.remove("selected"));
    resetHighlight();
    renderDiscoveredJourneys();
  });

  // Prime the Paths pane with maximal journeys so the view is useful on open.
  renderDiscoveredJourneys();
})();
</script>
</body>
</html>
"""


class GraphExportService:
    """
    Stateless service for exporting a knowledge graph to various formats.

    Accepts the JSON dict produced by ``KnowledgeGraph.export_json()`` and
    converts it to DOT, Mermaid, PNG (via networkx/matplotlib), or writes
    all formats to disk.
    """

    # ── Serialization formats ────────────────────────────────────────

    @staticmethod
    def to_json(graph_data: Dict[str, Any], *, indent: int = 2) -> str:
        """Serializes the graph data to a formatted JSON string."""

        return json.dumps(graph_data, indent=indent, default=str)

    @staticmethod
    def to_dot(graph_data: Dict[str, Any]) -> str:
        """Converts graph data to GraphViz DOT format with human-readable labels."""

        screen_names = _build_screen_names(graph_data)

        # Determine if activity-based or visual_hash-based
        nodes = graph_data.get("nodes", [])
        is_activity_based = any(n.get("activity") and not n.get("visual_hash") for n in nodes)

        lines = [
            "digraph KnowledgeGraph {",
            "  rankdir=LR;",
            '  bgcolor="#0d1117";',
            '  node [shape=box, style="rounded,filled", fontsize=9, fontname="Helvetica",',
            '        fillcolor="#1f2937", fontcolor="#f0f6fc", color="#30363d"];',
            '  edge [fontsize=7, fontname="Helvetica", fontcolor="#8b949e",',
            '        color="#58a6ff"];',
            "",
        ]

        for node in nodes:
            if is_activity_based:
                node_id = node.get("activity")
                if not node_id:
                    continue
                name = screen_names.get(node_id, _friendly_activity(node_id))
            else:
                node_id = node.get("visual_hash")
                if not node_id:
                    continue
                name = screen_names.get(node_id, node_id[:8])

            label = f"{name}\\n(visits: {node.get('visit_count', 0)})".replace('"', '\\"')
            lines.append(f'  "{node_id}" [label="{label}"];')

        lines.append("")

        edges = graph_data.get("edges", [])
        if is_activity_based:
            src_key, dst_key = "source_activity", "destination_activity"
        else:
            src_key, dst_key = "source_hash", "destination_hash"

        for edge in edges:
            src = edge.get(src_key)
            dst = edge.get(dst_key)
            if not src or not dst:
                continue

            action = edge.get("action_type", "")
            target = edge.get("action_target", "")
            label = f"{action}: {target}" if target else action
            if edge.get("count", 1) > 1:
                label += f" (×{edge['count']})"
            label = label.replace('"', '\\"')

            lines.append(f'  "{src}" -> "{dst}" [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def to_mermaid(graph_data: Dict[str, Any]) -> str:
        """Converts graph data to a Mermaid flowchart with human-readable labels."""

        screen_names = _build_screen_names(graph_data)

        nodes = graph_data.get("nodes", [])
        is_activity_based = any(n.get("activity") and not n.get("visual_hash") for n in nodes)

        lines = ["graph LR"]

        node_ids: Dict[str, str] = {}
        for i, node in enumerate(nodes):
            if is_activity_based:
                node_key = node.get("activity")
                if not node_key:
                    continue
                name = screen_names.get(node_key, _friendly_activity(node_key))
            else:
                node_key = node.get("visual_hash")
                if not node_key:
                    continue
                name = screen_names.get(node_key, node_key[:8])

            node_id = f"N{i}"
            node_ids[node_key] = node_id

            visits = node.get("visit_count", 0)
            label = f"{name} (visits: {visits})"
            label = label.replace('"', "'")
            lines.append(f'  {node_id}["{label}"]')

        edges = graph_data.get("edges", [])
        if is_activity_based:
            src_key, dst_key = "source_activity", "destination_activity"
        else:
            src_key, dst_key = "source_hash", "destination_hash"

        for edge in edges:
            src = node_ids.get(edge.get(src_key, ""))
            dst = node_ids.get(edge.get(dst_key, ""))
            if not src or not dst:
                continue

            action = edge.get("action_type", "")
            target = edge.get("action_target", "")
            label = f"{action}: {target}" if target else action
            if edge.get("count", 1) > 1:
                label += f" (×{edge['count']})"
            label = label.replace('"', "'")
            lines.append(f'  {src} -->|"{label}"| {dst}')

        return "\n".join(lines)

    # ── Interactive HTML (vis-network, self-contained) ───────────────

    @staticmethod
    def to_html(
        graph_data: Dict[str, Any],
        *,
        title: Optional[str] = None,
        screenshots_root: Optional[Path] = None,
    ) -> str:
        """Renders the knowledge graph as a self-contained interactive HTML page.

        The page loads vis-network from a CDN and embeds the graph data
        inline as JSON — open the file in any modern browser, no server
        required.  Users can pan/zoom, click nodes/edges for details,
        filter by activity or description, and toggle the physics sim.

        Args:
            graph_data: The dict from ``KnowledgeGraph.export_json()``.
            title: Optional page title; auto-generated from stats when omitted.
            screenshots_root: Optional path to a screenshot tree (e.g.
                ``assets/screenshot/YYYY-MM-DD/{package}/``).  When provided,
                each node is matched to the closest same-activity screenshot
                and embedded as a base64 thumbnail in the sidebar.
        """

        nodes: List[Dict[str, Any]] = graph_data.get("nodes", [])
        edges: List[Dict[str, Any]] = graph_data.get("edges", [])
        stats: Dict[str, Any] = graph_data.get("stats", {})

        is_activity_based = any(n.get("activity") and not n.get("visual_hash") for n in nodes)

        if title is None:
            n_screens = stats.get("unique_screens", len(nodes))
            n_trans = stats.get("total_transitions", len(edges))
            title = f"Knowledge Graph \u2014 {n_screens} screens \u00b7 {n_trans} transitions"

        thumbnails = _build_thumbnails(nodes, screenshots_root)

        payload = {
            "graph": graph_data,
            "meta": {
                "screen_names": _build_screen_names(graph_data),
                "activity_colors": _activity_color_map(nodes),
                "is_activity_based": is_activity_based,
                "title": title,
                "thumbnails": thumbnails,
            },
        }

        # Embed as JSON in a <script type="application/json"> block.  Escape
        # </ so a literal </script> inside any description can't close the tag.
        data_json = json.dumps(payload, default=str).replace("</", "<\\/")
        safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        return _HTML_TEMPLATE.replace("{{TITLE}}", safe_title).replace("{{DATA}}", data_json)

    # ── PNG rendering (networkx + matplotlib) ────────────────────────

    @staticmethod
    def to_png(
        graph_data: Dict[str, Any],
        output_path: str,
        *,
        title: Optional[str] = None,
        dpi: int = 150,
    ) -> str:
        """Renders the knowledge graph to a PNG image.

        Uses networkx for layout and matplotlib for rendering.  Produces a
        dark-themed, human-readable diagram with:

        - Nodes sized by visit count
        - Nodes colored by Android activity
        - Descriptive labels (screen description > activity class name)
        - Clean edge labels (action: target)
        - A legend mapping colors to activities

        Args:
            graph_data: Dict from ``KnowledgeGraph.export_json()``.
            output_path: Destination PNG file path.
            title: Optional title override; auto-generated from stats when omitted.
            dpi: Image resolution.

        Returns:
            The absolute path of the written PNG.
        """

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as exc:
            logger.warning(
                "Cannot render PNG — missing dependency: %s. "
                "Install with: pip install networkx matplotlib",
                exc,
            )
            return ""

        nodes: List[Dict[str, Any]] = graph_data.get("nodes", [])
        edges: List[Dict[str, Any]] = graph_data.get("edges", [])

        if not nodes:
            logger.info("No nodes to render — skipping PNG export.")
            return ""

        screen_names = _build_screen_names(graph_data)

        # Determine if activity-based or visual_hash-based
        is_activity_based = any(n.get("activity") and not n.get("visual_hash") for n in nodes)

        G = nx.DiGraph()

        # ── Build graph ──────────────────────────────────────────────
        known_ids: set[str] = set()
        for node in nodes:
            if is_activity_based:
                node_id = node.get("activity")
                if not node_id:
                    continue
                activity_str = _friendly_activity(node_id)
                name = screen_names.get(node_id, activity_str)
            else:
                node_id = node.get("visual_hash")
                if not node_id:
                    continue
                activity_str = _friendly_activity(node.get("activity", "unknown"))
                name = screen_names.get(node_id, node_id[:8])

            known_ids.add(node_id)
            label = f"{name}\n(visits: {node.get('visit_count', 0)})"
            visits = node.get("visit_count", 0)
            G.add_node(node_id, label=label, visits=visits, activity=activity_str)

        if is_activity_based:
            src_key, dst_key = "source_activity", "destination_activity"
        else:
            src_key, dst_key = "source_hash", "destination_hash"

        for edge in edges:
            src = edge.get(src_key)
            dst = edge.get(dst_key)
            if not src or not dst:
                continue
            # Skip edges referencing nodes we don't know about
            if src not in known_ids or dst not in known_ids:
                continue

            action = edge.get("action_type", "")
            target = edge.get("action_target", "")
            label = f"{action}: {target}" if target else action
            if edge.get("count", 1) > 1:
                label += f" (×{edge['count']})"
            G.add_edge(src, dst, label=label)

        # ── Layout ───────────────────────────────────────────────────
        node_count = len(G.nodes())
        k = max(2.5, 4.0 - node_count * 0.05)
        fig_w = max(16, min(28, node_count * 1.8))
        fig_h = max(10, min(20, node_count * 1.2))

        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        pos = nx.spring_layout(G, k=k, iterations=120, seed=42)

        # ── Node styling ─────────────────────────────────────────────
        activities = sorted({G.nodes[n].get("activity", "unknown") for n in G.nodes()})
        cmap = plt.get_cmap("Set2")
        palette = (
            cmap.colors if hasattr(cmap, "colors") else [cmap(i) for i in range(len(activities))]
        )
        color_map = {act: palette[i % len(palette)] for i, act in enumerate(activities)}

        node_list = list(G.nodes())
        node_sizes = [max(G.nodes[n].get("visits", 1) * 350, 1400) for n in node_list]
        node_colors = [
            color_map.get(G.nodes[n].get("activity", "unknown"), "#4488ff") for n in node_list
        ]

        # ── Draw nodes first (so edges render on top) ────────────────
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=node_list,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.92,
            edgecolors="#c9d1d9",
            linewidths=1.5,
        )

        # ── Draw edges ───────────────────────────────────────────────
        # Draw each edge individually so we can set per-edge margins
        # based on the actual source/target node radii.  This ensures
        # arrowheads end at the node boundary, not inside or far away.
        import math

        size_lookup = dict(zip(node_list, node_sizes, strict=True))

        for u, v, _edata in G.edges(data=True):
            src_radius = math.sqrt(size_lookup.get(u, 1400)) / 2 + 5
            tgt_radius = math.sqrt(size_lookup.get(v, 1400)) / 2 + 5

            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edgelist=[(u, v)],
                edge_color="#58a6ff",
                alpha=0.6,
                width=1.3,
                arrows=True,
                arrowsize=20,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.12",
                min_source_margin=src_radius,
                min_target_margin=tgt_radius,
            )

        # ── Node labels ──────────────────────────────────────────────
        labels = {n: G.nodes[n]["label"] for n in G.nodes()}
        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            ax=ax,
            font_size=6.5,
            font_color="#f0f6fc",
            font_weight="bold",
        )

        # ── Edge labels ──────────────────────────────────────────────
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=5.5,
            font_color="#8b949e",
            label_pos=0.35,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "#161b22",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )

        # ── Title ────────────────────────────────────────────────────
        stats = graph_data.get("stats", {})
        if title is None:
            n_screens = stats.get("unique_screens", node_count)
            n_transitions = stats.get("total_transitions", len(edges))
            n_activities = stats.get("unique_activities", len(activities))
            title = (
                f"App Knowledge Graph\n"
                f"{n_screens} screens  \u00b7  {n_transitions} transitions  "
                f"\u00b7  {n_activities} activities"
            )

        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold",
            color="#f0f6fc",
            pad=20,
        )

        # ── Legend ────────────────────────────────────────────────────
        for activity in activities:
            ax.scatter([], [], c=[color_map[activity]], s=100, label=activity)
        ax.legend(
            loc="lower left",
            fontsize=7,
            facecolor="#161b22",
            edgecolor="#30363d",
            labelcolor="#c9d1d9",
            framealpha=0.9,
        )

        ax.margins(0.12)
        ax.axis("off")
        plt.tight_layout()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)

        logger.info("Knowledge graph PNG saved to %s", out)
        return str(out.resolve())

    # ── Disk export ──────────────────────────────────────────────────

    @staticmethod
    def save(
        graph_data: Dict[str, Any],
        output_dir: str = "assets/exports",
        *,
        prefix: str = "knowledge_graph",
        formats: Optional[Dict[str, bool]] = None,
        screenshots_root: Optional[Path] = None,
    ) -> Dict[str, str]:
        """Writes the graph to disk in one or more formats.

        Args:
            graph_data: The dict from ``KnowledgeGraph.export_json()``.
            output_dir: Directory to write files into.
            prefix: Filename prefix.
            formats: Which formats to export.  Defaults to all five.
                     Keys: ``"json"``, ``"dot"``, ``"mermaid"``, ``"png"``,
                     ``"html"``.
            screenshots_root: Optional screenshot tree passed through to the
                HTML exporter for thumbnail embedding.  If omitted, the HTML
                viewer will infer ``assets/screenshot`` as a default when that
                directory exists.

        Returns:
            Dict mapping format name to the written file path.
        """

        if formats is None:
            formats = {
                "json": True,
                "dot": True,
                "mermaid": True,
                "png": True,
                "html": True,
            }

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        written: Dict[str, str] = {}

        if formats.get("json"):
            path = out / f"{prefix}.json"
            path.write_text(GraphExportService.to_json(graph_data))
            written["json"] = str(path)

        if formats.get("dot"):
            path = out / f"{prefix}.dot"
            path.write_text(GraphExportService.to_dot(graph_data))
            written["dot"] = str(path)

        if formats.get("mermaid"):
            path = out / f"{prefix}.mmd"
            path.write_text(GraphExportService.to_mermaid(graph_data))
            written["mermaid"] = str(path)

        if formats.get("png"):
            png_path = str(out / f"{prefix}.png")
            result = GraphExportService.to_png(graph_data, png_path)
            if result:
                written["png"] = result

        if formats.get("html"):
            path = out / f"{prefix}.html"
            resolved_screens = screenshots_root
            if resolved_screens is None:
                default_root = Path("assets/screenshot")
                if default_root.exists():
                    resolved_screens = default_root
            path.write_text(
                GraphExportService.to_html(graph_data, screenshots_root=resolved_screens)
            )
            written["html"] = str(path)

        return written
