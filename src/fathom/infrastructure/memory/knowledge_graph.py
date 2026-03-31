from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, List, Optional, Set, Tuple

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


def normalize_activity(activity: str) -> str:
    """
    Normalize an Android activity string to a canonical form.

    ADB sometimes returns ``"pkg/pkg.Activity"`` and sometimes just
    ``"pkg.Activity"``.  Strip the ``"package/"`` prefix so comparisons
    are consistent.  E.g.::

        "in.swiggy.android/in.swiggy.android.activities.HomeActivity"
        → "in.swiggy.android.activities.HomeActivity"
    """

    if "/" in activity:
        return activity.split("/", 1)[1]
    return activity


@dataclass
class GraphNode:
    """Lightweight in-memory representation of a screen node."""

    visual_hash: str
    activity: str
    description: Optional[str] = None
    rich_description: Optional[str] = None
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
                if not existing.rich_description and screen.get("rich_description"):
                    existing.rich_description = screen["rich_description"]
            else:
                node = GraphNode(
                    visual_hash=canonical,
                    activity=screen["activity"],
                    description=screen["description"],
                    rich_description=screen.get("rich_description"),
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

    async def update_rich_description(
        self,
        visual_hash: str,
        rich_description: str,
    ) -> None:
        """
        Stores a rich (detailed markdown) screen description on the node
        and persists it to SQLite.
        """

        canonical = self._resolve_canonical(visual_hash)
        node = self.__nodes.get(canonical)
        if node:
            node.rich_description = rich_description
        await self.__provider.update_rich_description(
            visual_hash=canonical,
            rich_description=rich_description,
        )

    async def append_activity_description(
        self,
        activity: str,
        observation: str,
    ) -> None:
        """
        Appends a new observation to the activity's rich description.

        The LLM is responsible for deduplication — it receives the existing
        description in context and is instructed to only output NEW details.
        This method simply appends whatever the LLM returns.
        """

        # Find the canonical node that owns the description for this activity
        target_node: Optional[GraphNode] = None
        for node in self.__nodes.values():
            if (
                normalize_activity(node.activity) == normalize_activity(activity)
                and node.rich_description is not None
            ):
                target_node = node
                break

        if target_node is None:
            # First observation for this activity — find any node with this activity
            for node in self.__nodes.values():
                if normalize_activity(node.activity) == normalize_activity(activity):
                    target_node = node
                    break

        if target_node is None:
            logger.warning("No node found for activity %s — dropping observation", activity)
            return

        if target_node.rich_description:
            target_node.rich_description += f"\n\n---\n\n### Additional Observation\n{observation}"
        else:
            target_node.rich_description = observation

        await self.__provider.update_rich_description(
            visual_hash=target_node.visual_hash,
            rich_description=target_node.rich_description,
        )

    def _get_tried_actions_for_activity(
        self, activity: str
    ) -> List[Tuple[str, str, Optional[str]]]:
        """
        Aggregates tried actions across ALL screen nodes that share the
        given activity.  Deduplicates by (action_type, action_target).
        """

        seen: Set[Tuple[str, str]] = set()
        result: List[Tuple[str, str, Optional[str]]] = []

        for node_hash, node in self.__nodes.items():
            if node.activity != activity:
                continue
            canonical = self._resolve_canonical(node_hash)
            for edge in self.__edges.get(canonical, []):
                if edge.action_type == "back":
                    continue
                key = (edge.action_type, edge.action_target)
                if key not in seen:
                    seen.add(key)
                    dest_node = self.__nodes.get(edge.destination_hash)
                    dest_desc = dest_node.description if dest_node else None
                    result.append((edge.action_type, edge.action_target, dest_desc))

        return result

    def _get_activity_description(self, activity: str) -> Optional[str]:
        """Returns the existing rich description for an activity, or None."""
        for node in self.__nodes.values():
            if (
                normalize_activity(node.activity) == normalize_activity(activity)
                and node.rich_description
            ):
                return node.rich_description
        return None

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

    def has_activity_description(self, activity: str) -> bool:
        """
        Returns True if ANY node with the given activity already has a
        rich description.  Used to deduplicate descriptions per-activity
        rather than per-visual-hash.
        """

        return any(
            n.activity == activity and n.rich_description is not None for n in self.__nodes.values()
        )

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
            if edge.action_type == "back":
                continue
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
        fully_scanned_count: Optional[int] = None,
        recent_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Formats the current knowledge graph state as LLM-readable context
        for the exploration prompt (DFS or BFS).

        The output is injected into the VLM user payload so the model knows
        which actions have already been tried on the current screen and can
        pick an untried element (or signal ``content_exhausted``).

        Parameters
        ----------
        current_hash:
            Canonical visual hash of the screen being scanned.
        depth:
            Path length from root (for DFS depth display).
        parent_description:
            Description of the screen we navigated from.
        fully_scanned_count:
            Number of screens already marked fully explored.
        recent_steps:
            Optional list of recent action dicts (up to 5).  Each dict has
            keys: ``type`` (str), ``target`` (str), ``success`` (bool),
            ``screen_changed`` (bool).  Injected as reactive feedback so
            the LLM can course-correct based on recent trajectory.
        """

        lines: List[str] = []

        # ── Recent-step reactive feedback (anchored at the top) ──────
        if recent_steps:
            step_lines = []
            for step in recent_steps:
                action = step.get("type", "")
                target = step.get("target", "")
                success = step.get("success", True)
                changed = step.get("screen_changed", True)

                if success and changed:
                    indicator = "ok, new screen"
                elif success and not changed:
                    indicator = "ok, NO screen change"
                else:
                    indicator = "FAILED"
                step_lines.append(f'- {action} "{target}" -> {indicator}')

            lines.append("RECENT ACTIONS (oldest to newest):\n" + "\n".join(step_lines))

        # Progress header
        scanned_part = ""
        if fully_scanned_count is not None:
            scanned_part = f", {fully_scanned_count} fully explored"
        lines.append(
            f"EXPLORATION PROGRESS: {self.node_count} screens discovered"
            f"{scanned_part}, {self.edge_count} transitions"
        )

        if depth is not None:
            lines.append(f"DEPTH: {depth}")

        if parent_description:
            lines.append(f"PARENT SCREEN: {parent_description}")

        if current_hash:
            current_hash = self._resolve_canonical(current_hash)
            node = self.__nodes.get(current_hash)
            if node and node.description:
                lines.append(f"CURRENT SCREEN: {node.description}")

            # Inject existing activity description so the LLM knows what
            # has already been captured and only outputs NEW observations.
            if node:
                existing_desc = self._get_activity_description(node.activity)
                if existing_desc:
                    lines.append(
                        "EXISTING DESCRIPTION FOR THIS ACTIVITY (do NOT repeat — only describe what is NEW):\n"
                        + existing_desc
                    )

            # Aggregate tried actions across ALL screens sharing the same
            # activity, so revisiting an activity on a different visual hash
            # still shows the full history of what was tried.
            activity = node.activity if node else None
            tried = self._get_tried_actions_for_activity(activity) if activity else []
            if not tried:
                # Fallback to per-screen tried actions
                tried = self.get_tried_actions(current_hash)

            if tried:
                lines.append("ALREADY TRIED IN THIS ACTIVITY:")
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
                lines.append(f"ACTIONS TRIED: {len(tried)}")

                # Hard-constraint forbidden list (exact target names)
                forbidden = sorted({t for _, t, _ in tried if t})
                if forbidden:
                    lines.append(
                        "FORBIDDEN TARGETS (do NOT select any of these): "
                        + ", ".join(f'"{t}"' for t in forbidden)
                    )
            else:
                lines.append("ALREADY TRIED IN THIS ACTIVITY: (none -- this is a fresh activity)")

        # Recent discoveries — last 5 screens by first_seen (descending)
        nodes_with_ts = [
            (h, n) for h, n in self.__nodes.items() if n.first_seen is not None and n.description
        ]
        if nodes_with_ts:
            nodes_with_ts.sort(key=lambda x: x[1].first_seen or 0, reverse=True)
            recent = nodes_with_ts[:5]
            lines.append("RECENT DISCOVERIES:")
            for _h, n in recent:
                lines.append(f"- {n.description}")

        return "\n".join(lines)

    # ── Query & Navigation Features ───────────────────────────────

    def find_path(
        self,
        start_hash: str,
        end_hash: str,
        max_depth: int = 50,
    ) -> Optional[List[Tuple[str, Optional[GraphEdge]]]]:
        """
        Finds the shortest path from start_hash to end_hash using BFS.

        Returns a list of (node_hash, edge_taken) tuples representing the path,
        or None if no path exists. The start node is always first with edge=None.
        The end node is last with the edge that led to it.

        Parameters
        ----------
        start_hash:
            The starting screen's visual hash.
        end_hash:
            The target screen's visual hash.
        max_depth:
            Maximum search depth to prevent infinite exploration.

        Returns
        -------
        List[Tuple[str, Optional[GraphEdge]]] or None
            Path as [(node, edge_taken), ...] or None if unreachable.
            Edge is None for the start node.
        """
        start_hash = self._resolve_canonical(start_hash)
        end_hash = self._resolve_canonical(end_hash)

        if start_hash not in self.__nodes:
            logger.warning(f"Start hash {start_hash} not in graph")
            return None

        if end_hash not in self.__nodes:
            logger.warning(f"End hash {end_hash} not in graph")
            return None

        if start_hash == end_hash:
            return [(start_hash, None)]

        # BFS queue: (current_hash, path_to_current)
        queue: deque[Tuple[str, List[Tuple[str, Optional[GraphEdge]]]]] = deque()
        visited: Set[str] = set()

        initial_path: List[Tuple[str, Optional[GraphEdge]]] = [(start_hash, None)]
        queue.append((start_hash, initial_path))
        visited.add(start_hash)

        while queue:
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            for edge in self.__edges.get(current, []):
                next_hash = edge.destination_hash

                if next_hash == end_hash:
                    return path + [(next_hash, edge)]

                if next_hash not in visited:
                    visited.add(next_hash)
                    new_path = path + [(next_hash, edge)]
                    queue.append((next_hash, new_path))

        return None

    def find_all_paths(
        self,
        start_hash: str,
        end_hash: str,
        max_depth: int = 10,
    ) -> List[List[Tuple[str, Optional[GraphEdge]]]]:
        """
        Finds all paths from start_hash to end_hash up to max_depth.

        Returns a list of paths, where each path is a list of
        (node_hash, edge_taken) tuples.

        Parameters
        ----------
        start_hash:
            The starting screen's visual hash.
        end_hash:
            The target screen's visual hash.
        max_depth:
            Maximum search depth (smaller than find_path to avoid explosion).

        Returns
        -------
        List[List[Tuple[str, Optional[GraphEdge]]]]
            All found paths, or empty list if none exist.
        """
        start_hash = self._resolve_canonical(start_hash)
        end_hash = self._resolve_canonical(end_hash)

        if start_hash not in self.__nodes or end_hash not in self.__nodes:
            return []

        all_paths: List[List[Tuple[str, Optional[GraphEdge]]]] = []

        def dfs(
            current: str,
            target: str,
            path: List[Tuple[str, Optional[GraphEdge]]],
            visited: Set[str],
            depth: int,
        ) -> None:
            if depth > max_depth:
                return

            if current == target:
                all_paths.append(path[:])
                return

            for edge in self.__edges.get(current, []):
                next_hash = edge.destination_hash
                if next_hash not in visited:
                    visited.add(next_hash)
                    path.append((next_hash, edge))
                    dfs(next_hash, target, path, visited, depth + 1)
                    path.pop()
                    visited.remove(next_hash)

        initial_path: List[Tuple[str, Optional[GraphEdge]]] = [(start_hash, None)]
        visited_set: Set[str] = {start_hash}
        dfs(start_hash, end_hash, initial_path, visited_set, 0)

        return all_paths

    def is_reachable(self, start_hash: str, end_hash: str, max_depth: int = 100) -> bool:
        """
        Checks if end_hash is reachable from start_hash.

        Uses BFS for efficiency, stopping as soon as the target is found.

        Parameters
        ----------
        start_hash:
            The starting screen's visual hash.
        end_hash:
            The target screen's visual hash.
        max_depth:
            Maximum search depth to prevent infinite exploration.

        Returns
        -------
        bool
            True if a path exists, False otherwise.
        """
        return self.find_path(start_hash, end_hash, max_depth) is not None

    def detect_cycles(self, start_hash: Optional[str] = None) -> List[List[str]]:
        """
        Detects all cycles in the graph using DFS.

        If start_hash is provided, only searches for cycles reachable from that node.
        Otherwise, searches the entire graph for all cycles.

        Returns a list of cycles, where each cycle is a list of node hashes
        representing the cycle path (first and last node are the same).

        Parameters
        ----------
        start_hash:
            Optional starting point for cycle detection. If None, searches entire graph.

        Returns
        -------
        List[List[str]]
            List of cycles found, each cycle is a list of node hashes.
        """
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        parent_map: Dict[str, str] = {}

        def dfs_visit(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for edge in self.__edges.get(node, []):
                next_node = edge.destination_hash

                if next_node not in visited:
                    parent_map[next_node] = node
                    dfs_visit(next_node, path)
                elif next_node in rec_stack:
                    # Found a cycle: extract it from the path
                    cycle_start_idx = path.index(next_node)
                    cycle = path[cycle_start_idx:] + [next_node]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        # Determine which nodes to start from
        start_nodes: List[str] = []
        if start_hash:
            resolved = self._resolve_canonical(start_hash)
            if resolved in self.__nodes:
                start_nodes = [resolved]
        else:
            start_nodes = list(self.__nodes.keys())

        for node in start_nodes:
            if node not in visited:
                dfs_visit(node, [])

        return cycles

    def get_connected_component(self, start_hash: str) -> Set[str]:
        """
        Returns all nodes reachable from start_hash (forward reachability).

        Uses BFS to find all nodes in the connected component.

        Parameters
        ----------
        start_hash:
            The starting screen's visual hash.

        Returns
        -------
        Set[str]
            Set of all reachable node hashes (including start_hash).
        """
        start_hash = self._resolve_canonical(start_hash)
        if start_hash not in self.__nodes:
            return set()

        reachable: Set[str] = set()
        queue: deque[str] = deque([start_hash])
        reachable.add(start_hash)

        while queue:
            current = queue.popleft()
            for edge in self.__edges.get(current, []):
                next_node = edge.destination_hash
                if next_node not in reachable:
                    reachable.add(next_node)
                    queue.append(next_node)

        return reachable

    def get_reverse_connected_component(self, end_hash: str) -> Set[str]:
        """
        Returns all nodes that can reach end_hash (backward reachability).

        Builds reverse edges and uses BFS to find all nodes that lead to end_hash.

        Parameters
        ----------
        end_hash:
            The target screen's visual hash.

        Returns
        -------
        Set[str]
            Set of all nodes that can reach end_hash (including end_hash).
        """
        end_hash = self._resolve_canonical(end_hash)
        if end_hash not in self.__nodes:
            return set()

        # Build reverse edge map
        reverse_edges: Dict[str, List[str]] = {}
        for source, edges in self.__edges.items():
            for edge in edges:
                dest = edge.destination_hash
                reverse_edges.setdefault(dest, []).append(source)

        # BFS backward from end_hash
        can_reach: Set[str] = set()
        queue: deque[str] = deque([end_hash])
        can_reach.add(end_hash)

        while queue:
            current = queue.popleft()
            for source in reverse_edges.get(current, []):
                if source not in can_reach:
                    can_reach.add(source)
                    queue.append(source)

        return can_reach

    def get_graph_diameter(self) -> Optional[int]:
        """
        Computes the diameter of the graph (longest shortest path between any two nodes).

        Returns None if the graph is empty or disconnected.
        For large graphs, this may be expensive. Consider caching the result.

        Returns
        -------
        int or None
            The graph diameter, or None if not computable.
        """
        if not self.__nodes:
            return None

        nodes = list(self.__nodes.keys())
        max_distance = 0

        for start in nodes:
            for end in nodes:
                if start != end:
                    path = self.find_path(start, end)
                    if path:
                        distance = len(path) - 1
                        max_distance = max(max_distance, distance)

        return max_distance if max_distance > 0 else None

    def get_visualization_context(self, visual_hash: str, depth: int = 2) -> Dict[str, Any]:
        """
        Generates context for visualizing a screen and its neighborhood in the graph.

        Returns information about the node, its incoming/outgoing edges,
        and paths to/from nearby nodes.

        Parameters
        ----------
        visual_hash:
            The target screen's visual hash.
        depth:
            How many hops away to include in the context.

        Returns
        -------
        Dict[str, Any]
            Context dict with node info, neighbors, and reachability analysis.
        """
        visual_hash = self._resolve_canonical(visual_hash)
        node = self.__nodes.get(visual_hash)

        if not node:
            return {}

        # Get forward and backward reachability
        forward = self.get_connected_component(visual_hash)
        backward = self.get_reverse_connected_component(visual_hash)

        # Get immediate neighbors
        outgoing: List[Dict[str, Any]] = []
        for edge in self.__edges.get(visual_hash, []):
            dest_node = self.__nodes.get(edge.destination_hash)
            outgoing.append(
                {
                    "destination": edge.destination_hash,
                    "action_type": edge.action_type,
                    "action_target": edge.action_target,
                    "count": edge.count,
                    "destination_description": dest_node.description if dest_node else None,
                }
            )

        # Get inbound neighbors
        inbound: List[Dict[str, Any]] = []
        for source, edges in self.__edges.items():
            for edge in edges:
                if edge.destination_hash == visual_hash:
                    source_node = self.__nodes.get(source)
                    inbound.append(
                        {
                            "source": source,
                            "action_type": edge.action_type,
                            "action_target": edge.action_target,
                            "count": edge.count,
                            "source_description": source_node.description if source_node else None,
                        }
                    )

        return {
            "node": {
                "visual_hash": node.visual_hash,
                "activity": node.activity,
                "description": node.description,
                "visit_count": node.visit_count,
                "first_seen": node.first_seen,
                "last_seen": node.last_seen,
            },
            "outgoing_edges": outgoing,
            "inbound_edges": inbound,
            "forward_reachable": len(forward),
            "backward_reachable": len(backward),
            "in_cycle": len(backward) > 1 and visual_hash in backward,
        }

    def export_json(self) -> Dict[str, Any]:
        """
        Exports the full knowledge graph as a JSON-serializable dictionary.
        """

        nodes = []
        for node in self.__nodes.values():
            node_dict: Dict[str, Any] = {
                "visual_hash": node.visual_hash,
                "activity": node.activity,
                "description": node.description,
                "first_seen": node.first_seen,
                "last_seen": node.last_seen,
                "visit_count": node.visit_count,
            }
            if node.rich_description:
                node_dict["rich_description"] = node.rich_description
            nodes.append(node_dict)

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
