from __future__ import annotations

import json
import re
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
    """Infer a human-readable name for every screen node.

    Strategy (in priority order):
    1. Use the stored ``description`` if it is meaningful (not "Tool-based
       analysis" or empty).
    2. Infer from **incoming** edges: the ``action_target`` of the tap/scroll
       that *led to* this screen tells us what it is.  e.g. if the incoming
       edge says ``tap "Messages tab"`` → the screen is ``Messages``.
    3. Infer from **outgoing** edges: the targets of elements *on* the screen
       hint at its purpose (e.g. a screen with "Primary Care", "Urgent Care",
       "Messages tab" is the Home/Dashboard).
    4. Fall back to a cleaned-up activity class name.
    """

    nodes: List[Dict[str, Any]] = graph_data.get("nodes", [])
    edges: List[Dict[str, Any]] = graph_data.get("edges", [])

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

    names: Dict[str, str] = {}

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
        out = outgoing.get(vhash, [])
        if out:
            # Take up to 3 most descriptive items
            items = sorted(out, key=len, reverse=True)[:3]
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

        for node in graph_data.get("nodes", []):
            vhash = node.get("visual_hash")
            if not vhash:
                continue
            name = screen_names.get(vhash, vhash[:8])
            label = _node_label(name, node, max_len=50).replace('"', '\\"')
            visits = node.get("visit_count", 0)
            tooltip = f"visits: {visits}  hash: {vhash[:12]}"
            lines.append(f'  "{vhash}" [label="{label}", tooltip="{tooltip}"];')

        lines.append("")

        for edge in graph_data.get("edges", []):
            src = edge.get("source_hash")
            dst = edge.get("destination_hash")
            if not src or not dst:
                continue
            label = _edge_label(edge, max_target=30).replace('"', '\\"')
            lines.append(f'  "{src}" -> "{dst}" [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def to_mermaid(graph_data: Dict[str, Any]) -> str:
        """Converts graph data to a Mermaid flowchart with human-readable labels."""

        screen_names = _build_screen_names(graph_data)

        lines = ["graph LR"]

        node_ids: Dict[str, str] = {}
        for i, node in enumerate(graph_data.get("nodes", [])):
            vhash = node.get("visual_hash")
            if not vhash:
                continue
            node_id = f"S{i}"
            node_ids[vhash] = node_id

            name = screen_names.get(vhash, vhash[:8])
            label = _node_label(name, node, max_len=45).replace("\n", " | ")
            label = label.replace('"', "'")
            lines.append(f'  {node_id}["{label}"]')

        for edge in graph_data.get("edges", []):
            src = node_ids.get(edge.get("source_hash", ""))
            dst = node_ids.get(edge.get("destination_hash", ""))
            if not src or not dst:
                continue

            label = _edge_label(edge, max_target=25).replace('"', "'")
            lines.append(f'  {src} -->|"{label}"| {dst}')

        return "\n".join(lines)

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

        G = nx.DiGraph()

        # ── Build graph ──────────────────────────────────────────────
        known_hashes: set[str] = set()
        for node in nodes:
            vhash = node.get("visual_hash")
            if not vhash:
                continue
            known_hashes.add(vhash)
            friendly_act = _friendly_activity(node.get("activity", "unknown"))
            name = screen_names.get(vhash, vhash[:8])
            label = _node_label(name, node, max_len=35)
            visits = node.get("visit_count", 0)
            G.add_node(vhash, label=label, visits=visits, activity=friendly_act)

        for edge in edges:
            src = edge.get("source_hash")
            dst = edge.get("destination_hash")
            if not src or not dst:
                continue
            # Skip edges referencing nodes we don't know about to avoid
            # networkx auto-creating attribute-less nodes.
            if src not in known_hashes or dst not in known_hashes:
                continue
            label = _edge_label(edge, max_target=20)
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
    ) -> Dict[str, str]:
        """Writes the graph to disk in one or more formats.

        Args:
            graph_data: The dict from ``KnowledgeGraph.export_json()``.
            output_dir: Directory to write files into.
            prefix: Filename prefix.
            formats: Which formats to export.  Defaults to all four.
                     Keys: ``"json"``, ``"dot"``, ``"mermaid"``, ``"png"``.

        Returns:
            Dict mapping format name to the written file path.
        """

        if formats is None:
            formats = {"json": True, "dot": True, "mermaid": True, "png": True}

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

        return written
