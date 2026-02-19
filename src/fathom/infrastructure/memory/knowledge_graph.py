from __future__ import annotations

import time
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

from fathom.infrastructure.memory.sqlite import SQLiteMemoryProvider
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState

logger = getLogger(__name__)

# Cap "ALREADY TRIED" list in VLM context to bound token usage (per-screen).
MAX_TRIED_IN_CONTEXT = 25

# Maximum Hamming distance (in bits, out of 256 for a 16×16 dHash) for two
# visual hashes to be considered the same logical screen.  12 bits ≈ 95%
# similarity — safely merges minor pixel variations (status-bar clock, cursor
# blink, animation frames) while keeping genuinely different screens apart.
HAMMING_THRESHOLD = 12


@dataclass
class GraphNode:
    """Lightweight in-memory representation of a screen node."""

    visual_hash: str
    activity: str
    description: Optional[str] = None
    first_seen: Optional[int] = None
    last_seen: Optional[int] = None
    visit_count: int = 0


@dataclass
class GraphEdge:
    """Lightweight in-memory representation of a transition edge."""

    source_hash: str
    destination_hash: str
    action_type: str
    action_target: str
    count: int = 1
    first_seen: Optional[int] = None
    last_seen: Optional[int] = None


class KnowledgeGraph:
    """
    Persistent knowledge graph of a mobile application.

    Maintains an in-memory cache backed by SQLite for fast graph traversal
    during a run while persisting all discoveries across runs. The SQLite
    database is the source of truth; the in-memory structures are a
    read-through cache that is loaded on startup and updated on writes.
    """

    def __init__(self, database_path: str = "assets/memory/knowledge.db") -> None:
        self.__provider = SQLiteMemoryProvider(database_path=database_path)
        self.__nodes: Dict[str, GraphNode] = {}
        self.__edges: Dict[str, List[GraphEdge]] = {}  # source_hash -> edges
        self.__hash_aliases: Dict[str, str] = {}  # raw_hash -> canonical_hash
        self.__loaded = False

    # ── Fuzzy hash resolution ─────────────────────────────────────

    @staticmethod
    def _hamming_distance(hash1: str, hash2: str) -> int:
        if len(hash1) != len(hash2):
            return 256
        try:
            return bin(int(hash1, 16) ^ int(hash2, 16)).count("1")
        except ValueError:
            return 256

    def _resolve_canonical(self, visual_hash: str) -> str:
        """Map *visual_hash* to an existing node within HAMMING_THRESHOLD."""
        if visual_hash in self.__nodes:
            return visual_hash
        if visual_hash in self.__hash_aliases:
            return self.__hash_aliases[visual_hash]

        best_hash: Optional[str] = None
        best_distance = HAMMING_THRESHOLD + 1

        for existing_hash in self.__nodes:
            d = self._hamming_distance(visual_hash, existing_hash)
            if d < best_distance:
                best_distance = d
                best_hash = existing_hash

        if best_hash is not None and best_distance <= HAMMING_THRESHOLD:
            self.__hash_aliases[visual_hash] = best_hash
            return best_hash

        return visual_hash

    @property
    def provider(self) -> SQLiteMemoryProvider:
        """Returns the underlying SQLiteMemoryProvider for backward compat."""
        return self.__provider

    @property
    def nodes(self) -> Dict[str, GraphNode]:
        """All screen nodes currently in the graph."""
        return self.__nodes

    @property
    def node_count(self) -> int:
        """Number of unique screens in the graph."""
        return len(self.__nodes)

    @property
    def edge_count(self) -> int:
        """Total number of transition edges in the graph."""
        return sum(len(edges) for edges in self.__edges.values())

    async def load(self) -> None:
        """
        Hydrates the in-memory graph from the SQLite database.
        Call this once at the start of a run to pick up knowledge from prior runs.

        Performs fuzzy deduplication during load so that screens whose
        visual hashes differ by ≤ HAMMING_THRESHOLD bits are merged into
        a single canonical node.
        """

        raw_screen_count = 0
        screens = await self.__provider.get_all_screens()
        for screen in screens:
            raw_screen_count += 1
            raw_hash = screen["visual_hash"]
            canonical = self._resolve_canonical(raw_hash)
            existing = self.__nodes.get(canonical)

            if existing and canonical != raw_hash:
                existing.visit_count += screen["visit_count"] or 0
                existing.last_seen = max(existing.last_seen or 0, screen["last_seen"] or 0)
                if not existing.description and screen["description"]:
                    existing.description = screen["description"]
            else:
                node = GraphNode(
                    visual_hash=canonical,
                    activity=screen["activity"],
                    description=screen["description"],
                    first_seen=screen["first_seen"],
                    last_seen=screen["last_seen"],
                    visit_count=screen["visit_count"] or 0,
                )
                self.__nodes[canonical] = node

        transitions = await self.__provider.get_all_transitions()
        for t in transitions:
            src = self._resolve_canonical(t["source_hash"])
            dst = self._resolve_canonical(t["destination_hash"])
            action_type = t["action_type"]
            action_target = t["action_target"] or ""

            edges = self.__edges.setdefault(src, [])
            merged = False
            for edge in edges:
                if edge.action_type == action_type and edge.action_target == action_target:
                    edge.destination_hash = dst
                    edge.count += t["count"] or 1
                    edge.last_seen = max(edge.last_seen or 0, t["last_seen"] or 0)
                    merged = True
                    break
            if not merged:
                edges.append(
                    GraphEdge(
                        source_hash=src,
                        destination_hash=dst,
                        action_type=action_type,
                        action_target=action_target,
                        count=t["count"] or 1,
                        first_seen=t["first_seen"],
                        last_seen=t["last_seen"],
                    )
                )

        self.__loaded = True
        logger.info(
            "Knowledge graph loaded: %d raw → %d canonical screens, %d transitions  (aliases=%d)",
            raw_screen_count,
            self.node_count,
            self.edge_count,
            len(self.__hash_aliases),
        )

    async def add_screen(
        self,
        state: ScreenState,
        description: Optional[str] = None,
    ) -> GraphNode:
        """
        Registers a screen observation. Persists to SQLite and updates the
        in-memory cache. Increments visit_count on repeat visits.

        Uses fuzzy hash matching: if *state.visual_hash* is within
        HAMMING_THRESHOLD of an existing node, the observation is merged
        into that canonical node instead of creating a duplicate.
        """

        # Persist to SQLite (handles upsert + visit_count increment)
        await self.__provider.store_observation(screen=state, description=description)

        # Resolve to canonical hash for in-memory dedup
        canonical = self._resolve_canonical(state.visual_hash)
        now = int(time.time())
        existing = self.__nodes.get(canonical)

        if existing:
            existing.visit_count += 1
            existing.last_seen = now
            if description:
                existing.description = description
            return existing

        node = GraphNode(
            visual_hash=state.visual_hash,
            activity=state.activity,
            description=description,
            first_seen=now,
            last_seen=now,
            visit_count=1,
        )
        self.__nodes[state.visual_hash] = node
        return node

    async def record_transition(
        self,
        source_hash: str,
        action: Action,
        destination_hash: str,
    ) -> None:
        """
        Records a screen-to-screen transition. Persists to SQLite and
        updates the in-memory edge cache. Deduplicates by
        (canonical_source, action_type, action_target).
        """

        canonical_src = self._resolve_canonical(source_hash)
        canonical_dst = self._resolve_canonical(destination_hash)

        # Persist to SQLite (uses canonical hashes for cleaner storage)
        await self.__provider.store_transition(
            source_hash=canonical_src,
            action=action,
            destination_hash=canonical_dst,
        )

        # Update in-memory cache
        action_type = (
            action.action_type.value
            if hasattr(action.action_type, "value")
            else str(action.action_type)
        )
        action_target = action.natural_language_target or action.target or ""
        now = int(time.time())

        edges = self.__edges.setdefault(canonical_src, [])
        for edge in edges:
            if edge.action_type == action_type and edge.action_target == action_target:
                edge.destination_hash = canonical_dst
                edge.count += 1
                edge.last_seen = now
                return

        edges.append(
            GraphEdge(
                source_hash=canonical_src,
                destination_hash=canonical_dst,
                action_type=action_type,
                action_target=action_target,
                count=1,
                first_seen=now,
                last_seen=now,
            )
        )

    def get_neighbors(self, visual_hash: str) -> List[GraphEdge]:
        """
        Returns all outgoing transition edges from a screen.
        """

        return list(self.__edges.get(self._resolve_canonical(visual_hash), []))

    def get_inbound_edge(self, destination_hash: str) -> Optional[Tuple[str, GraphEdge]]:
        """
        Returns the first inbound edge leading to *destination_hash*.

        Returns a ``(source_hash, edge)`` tuple, or ``None`` if no
        inbound transition is known.  Used by DFS orphan-recovery to
        locate a path to an unexplored screen.
        """

        canonical_dst = self._resolve_canonical(destination_hash)
        for source_hash, edges in self.__edges.items():
            for edge in edges:
                if edge.destination_hash == canonical_dst:
                    return source_hash, edge
        return None

    def get_unexplored_screens(self, max_visits: int = 2) -> List[GraphNode]:
        """
        Returns screens that have been seen fewer than max_visits times.
        Useful for directing exploration toward under-visited areas.
        """

        return [node for node in self.__nodes.values() if node.visit_count < max_visits]

    def get_screen(self, visual_hash: str) -> Optional[GraphNode]:
        """
        Returns a single screen node, or None if unknown.
        """

        return self.__nodes.get(self._resolve_canonical(visual_hash))

    def has_screen(self, visual_hash: str) -> bool:
        """
        Checks whether a screen (or a fuzzy match) has been seen before.
        """

        return self._resolve_canonical(visual_hash) in self.__nodes

    def resolve_hash(self, visual_hash: str) -> str:
        """
        Public API: resolve a raw visual hash to its canonical form.

        Callers that compare hashes directly (e.g. ``pre != post``) should
        resolve both sides through this method first so that minor pixel
        variations are collapsed.
        """

        return self._resolve_canonical(visual_hash)

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns summary statistics about the knowledge graph.
        """

        total_edges = self.edge_count
        total_visits = sum(n.visit_count for n in self.__nodes.values())
        activities = {n.activity for n in self.__nodes.values()}
        unexplored = len(self.get_unexplored_screens())

        return {
            "unique_screens": self.node_count,
            "total_transitions": total_edges,
            "total_visits": total_visits,
            "unique_activities": len(activities),
            "activities": sorted(activities),
            "unexplored": unexplored,
        }

    def get_tried_actions(self, visual_hash: str) -> List[Tuple[str, str, Optional[str]]]:
        """
        Returns actions already recorded from a screen.

        Each entry is a tuple of ``(action_type, action_target,
        destination_description)`` where *destination_description* is the
        description stored on the destination screen node (or ``None``).
        """

        edges = self.__edges.get(self._resolve_canonical(visual_hash), [])
        result: List[Tuple[str, str, Optional[str]]] = []
        for edge in edges:
            dest_node = self.__nodes.get(edge.destination_hash)
            dest_desc = dest_node.description if dest_node else None
            result.append((edge.action_type, edge.action_target, dest_desc))
        return result

    def build_exploration_context(
        self,
        current_hash: Optional[str] = None,
        *,
        depth: Optional[int] = None,
        parent_description: Optional[str] = None,
    ) -> str:
        """
        Formats the current knowledge graph state as LLM-readable context
        for the exploration prompt (DFS or BFS).

        The output is injected into the VLM user payload so the model knows
        which actions have already been tried on the current screen and can
        pick an untried element (or signal ``content_exhausted``).

        For DFS: pass depth (path length from root) and parent_description
        (description of the screen we navigated from) when available.
        """

        lines: List[str] = []

        lines.append(f"EXPLORED SO FAR: {self.node_count} screens, {self.edge_count} transitions")

        if depth is not None:
            lines.append(f"DEPTH: {depth}")

        if parent_description:
            lines.append(f"PARENT SCREEN: {parent_description}")

        if current_hash:
            current_hash = self._resolve_canonical(current_hash)
            node = self.__nodes.get(current_hash)
            if node and node.description:
                lines.append(f"CURRENT SCREEN: {node.description}")

            tried = self.get_tried_actions(current_hash)
            if tried:
                lines.append("ALREADY TRIED FROM THIS SCREEN:")
                excess = len(tried) - MAX_TRIED_IN_CONTEXT
                for action_type, action_target, dest_desc in tried[:MAX_TRIED_IN_CONTEXT]:
                    entry = f"- {action_type}"
                    if action_target:
                        entry += f' "{action_target}"'
                    if dest_desc:
                        entry += f" -> {dest_desc}"
                    lines.append(entry)
                if excess > 0:
                    lines.append(f"... and {excess} more tried")
            else:
                lines.append("ALREADY TRIED FROM THIS SCREEN: (none -- this is a fresh screen)")

        return "\n".join(lines)

    def export_json(self) -> Dict[str, Any]:
        """
        Exports the full knowledge graph as a JSON-serializable dictionary.
        """

        nodes = []
        for node in self.__nodes.values():
            nodes.append(
                {
                    "visual_hash": node.visual_hash,
                    "activity": node.activity,
                    "description": node.description,
                    "first_seen": node.first_seen,
                    "last_seen": node.last_seen,
                    "visit_count": node.visit_count,
                }
            )

        edges = []
        for edge_list in self.__edges.values():
            for edge in edge_list:
                edges.append(
                    {
                        "source_hash": edge.source_hash,
                        "destination_hash": edge.destination_hash,
                        "action_type": edge.action_type,
                        "action_target": edge.action_target,
                        "count": edge.count,
                        "first_seen": edge.first_seen,
                        "last_seen": edge.last_seen,
                    }
                )

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": self.get_stats(),
        }

    def export_dot(self) -> str:
        """
        Exports the knowledge graph in GraphViz DOT format.

        Delegates to :class:`GraphExportService` for consistent,
        human-readable rendering across all export paths.
        """

        from fathom.services.export import GraphExportService

        return GraphExportService.to_dot(self.export_json())

    def export_mermaid(self) -> str:
        """
        Exports the knowledge graph as a Mermaid flowchart.

        Delegates to :class:`GraphExportService` for consistent,
        human-readable rendering across all export paths.
        """

        from fathom.services.export import GraphExportService

        return GraphExportService.to_mermaid(self.export_json())
