from __future__ import annotations

import time
from dataclasses import dataclass
from logging import getLogger
from typing import AbstractSet, Any, Dict, List, Optional, Set, Tuple

from fathom.constants import ActionType
from fathom.constants.exploration import FocusRelevance
from fathom.constants.screen import ScreenCategory
from fathom.infrastructure.memory.algorithms import GraphAlgorithms
from fathom.infrastructure.memory.canonical import ScreenCanonicalizer
from fathom.interfaces import IMemoryProvider
from fathom.schemas.actions import Action
from fathom.schemas.content import ScreenContent
from fathom.schemas.document import LinkSemantics
from fathom.schemas.exploration import ActionOutcome, TriedAction
from fathom.schemas.screens import ScreenState

logger = getLogger(__name__)

# Cap the "ALREADY TRIED" list injected into VLM context to bound tokens per screen.
MAX_TRIED_IN_CONTEXT = 25

# Descriptions the VLM emits that carry no signal; treated as "no description"
# so a later meaningful one can replace them. "fallback state" is the sentinel a
# failed analysis writes, so it must not survive as a screen's description.
USELESS_DESCRIPTIONS = frozenset({"", "unknown", "tool-based analysis", "fallback state"})

# Action types that "sample" a list item (counted by the sampling guard).
SAMPLING_ACTION_TYPES = frozenset(
    {ActionType.TAP.value, ActionType.LONG_PRESS.value, ActionType.TYPE.value}
)


@dataclass
class GraphNode:
    """
    Lightweight in-memory representation of a screen node.
    """

    visual_hash: str
    activity: str
    description: Optional[str] = None
    rich_description: Optional[str] = None
    first_seen: Optional[int] = None
    last_seen: Optional[int] = None
    visit_count: int = 0
    activity_hash: Optional[str] = None
    xml_hash: Optional[str] = None
    interaction_hash: Optional[str] = None
    structure_hash: Optional[str] = None
    exhausted: bool = False
    relevance: FocusRelevance = FocusRelevance.UNSCOPED
    category: ScreenCategory = ScreenCategory.OTHER
    content: Optional[ScreenContent] = None


@dataclass
class GraphEdge:
    """
    Lightweight in-memory representation of a transition edge.
    """

    source_hash: str
    destination_hash: str
    action_type: str
    action_target: str
    count: int = 1
    first_seen: Optional[int] = None
    last_seen: Optional[int] = None
    coord_bucket: Optional[str] = None
    coord_region: Optional[str] = None
    element_category: Optional[str] = None
    value: Optional[str] = None
    semantics: Optional[LinkSemantics] = None


class KnowledgeGraph:
    """
    Persistent knowledge graph of a mobile application.

    Maintains an in-memory cache backed by a memory provider for fast graph
    traversal during a run while persisting discoveries across runs. The
    provider is the source of truth; the in-memory structures are a
    read-through cache loaded on startup and updated on writes.
    """

    def __init__(self, *, provider: IMemoryProvider) -> None:
        self.__provider = provider
        self.__nodes: Dict[str, GraphNode] = {}
        self.__edges: Dict[str, List[GraphEdge]] = {}
        self.__aliases: Dict[str, str] = {}
        self.__canonicalizer = ScreenCanonicalizer()

    @staticmethod
    def normalize_activity(activity: str) -> str:
        """
        Strips the package prefix so "pkg/pkg.Activity" and "pkg.Activity" match.
        """

        if "/" in activity:
            return activity.split("/", 1)[1]
        return activity

    @staticmethod
    def package_of(activity: str) -> str:
        """
        Returns the package component of a "package/activity" identifier.
        """

        return activity.split("/", 1)[0]

    @staticmethod
    def has_meaningful_description(description: Optional[str]) -> bool:
        """
        Whether a description carries signal worth preserving across revisits.
        """

        if not description:
            return False
        return description.strip().lower() not in USELESS_DESCRIPTIONS

    @property
    def provider(self) -> IMemoryProvider:
        """
        Returns the underlying persistence provider.
        """

        return self.__provider

    @property
    def nodes(self) -> Dict[str, GraphNode]:
        """
        All screen nodes currently in the graph.
        """

        return self.__nodes

    @property
    def edges(self) -> Dict[str, List[GraphEdge]]:
        """
        All transition edges, keyed by canonical source hash.
        """

        return self.__edges

    @property
    def node_count(self) -> int:
        """
        Number of unique screens in the graph.
        """

        return len(self.__nodes)

    @property
    def edge_count(self) -> int:
        """
        Total number of transition edges in the graph.
        """

        return sum(len(edges) for edges in self.__edges.values())

    def __resolve(self, visual_hash: str) -> str:
        """
        Maps a visual hash to an existing node within the Hamming threshold.
        """

        return self.__canonicalizer.resolve(
            visual_hash=visual_hash, nodes=self.__nodes, aliases=self.__aliases
        )

    def __resolve_for_state(self, state: ScreenState) -> str:
        """
        Layered MLSIA dedup for a screen state, falling back to Hamming.
        """

        return self.__canonicalizer.resolve_for_state(
            state=state, nodes=self.__nodes, aliases=self.__aliases
        )

    def __resolve_identity(
        self,
        *,
        visual_hash: str,
        activity_hash: Optional[str],
        structure_hash: Optional[str],
    ) -> str:
        """
        Resolves a raw visual hash without crossing activity or structural identity.

        Used where the full screen state is unavailable (persisted-row hydration)
        so a near-duplicate merge cannot collapse structurally distinct screens.
        """

        return self.__canonicalizer.resolve_identity(
            visual_hash=visual_hash,
            activity_hash=activity_hash,
            structure_hash=structure_hash,
            nodes=self.__nodes,
            aliases=self.__aliases,
        )

    def resolve_hash(self, visual_hash: str) -> str:
        """
        Resolves a raw visual hash to its canonical form.
        """

        return self.__resolve(visual_hash)

    def canonical_for_state(self, *, state: ScreenState) -> str:
        """
        Resolves a freshly observed screen to its canonical node via gated identity.

        Callers that re-capture a screen (edge endpoints, DFS routing) get a fresh
        perceptual hash that misses the node and alias maps; resolving it by bare
        visual distance would snap it onto the nearest-looking node in any activity.
        Routing through the same gated resolution as add_screen keeps every such
        lookup on the correct node, so transitions never cross-staple features.
        """

        return self.__resolve_for_state(state)

    async def load(self) -> None:
        """
        Hydrates the in-memory graph from persistence, deduplicating on load.
        """

        screens = await self.__provider.get_all_screens()
        for screen in screens:
            self.__hydrate_screen(screen=screen)

        transitions = await self.__provider.get_all_transitions()
        for transition in transitions:
            self.__hydrate_transition(transition=transition)

        logger.info(
            "Knowledge graph loaded: %d screens, %d transitions (aliases=%d)",
            self.node_count,
            self.edge_count,
            len(self.__aliases),
        )

    def __hydrate_screen(self, *, screen: Dict[str, Any]) -> None:
        """
        Folds one persisted screen row into a canonical node.
        """

        raw_hash = screen["visual_hash"]
        canonical = self.__resolve_identity(
            visual_hash=raw_hash,
            activity_hash=screen.get("activity_hash"),
            structure_hash=screen.get("structure_hash"),
        )
        existing = self.__nodes.get(canonical)

        if existing and canonical != raw_hash:
            existing.visit_count += screen["visit_count"] or 0
            existing.last_seen = max(existing.last_seen or 0, screen["last_seen"] or 0)
            if not existing.description and screen["description"]:
                existing.description = screen["description"]
            if not existing.rich_description and screen.get("rich_description"):
                existing.rich_description = screen["rich_description"]
            if not existing.content and screen.get("content_json"):
                existing.content = self.__coerce_content(screen.get("content_json"))
            if screen.get("exhausted"):
                existing.exhausted = True
            persisted_relevance = self.__coerce_relevance(screen.get("relevance"))
            if persisted_relevance is not FocusRelevance.UNSCOPED:
                existing.relevance = persisted_relevance
            persisted_category = self.__coerce_category(screen.get("category"))
            if persisted_category is not ScreenCategory.OTHER:
                existing.category = persisted_category
            self.__backfill_hashes(node=existing, source=screen)
            return

        self.__nodes[canonical] = GraphNode(
            visual_hash=canonical,
            activity=screen["activity"],
            description=screen["description"],
            rich_description=screen.get("rich_description"),
            first_seen=screen["first_seen"],
            last_seen=screen["last_seen"],
            visit_count=screen["visit_count"] or 0,
            activity_hash=screen.get("activity_hash"),
            xml_hash=screen.get("xml_hash"),
            interaction_hash=screen.get("interaction_hash"),
            structure_hash=screen.get("structure_hash"),
            exhausted=bool(screen.get("exhausted", False)),
            relevance=self.__coerce_relevance(screen.get("relevance")),
            category=self.__coerce_category(screen.get("category")),
            content=self.__coerce_content(screen.get("content_json")),
        )

    def __hydrate_transition(self, *, transition: Dict[str, Any]) -> None:
        """
        Folds one persisted transition row into a canonical edge.
        """

        source = self.__resolve(transition["source_hash"])
        destination = self.__resolve(transition["destination_hash"])
        edges = self.__edges.setdefault(source, [])

        edge = GraphEdge(
            source_hash=source,
            destination_hash=destination,
            action_type=transition["action_type"],
            action_target=transition["action_target"] or "",
            coord_bucket=transition.get("coord_bucket"),
            coord_region=transition.get("coord_region"),
            element_category=transition.get("element_category"),
            value=transition.get("action_value"),
            semantics=self.__coerce_semantics(transition.get("semantics_json")),
            count=transition["count"] or 1,
            first_seen=transition["first_seen"],
            last_seen=transition["last_seen"],
        )
        if not self.__merge_edge(edges=edges, incoming=edge):
            edges.append(edge)

    @staticmethod
    def __merge_edge(*, edges: List[GraphEdge], incoming: GraphEdge) -> bool:
        """
        Merges an incoming edge into an existing one, preferring coord-bucket identity.
        """

        if incoming.coord_bucket:
            for edge in edges:
                if (
                    edge.action_type == incoming.action_type
                    and edge.coord_bucket == incoming.coord_bucket
                ):
                    KnowledgeGraph.__absorb_edge(target=edge, incoming=incoming)
                    return True

        for edge in edges:
            if (
                edge.action_type == incoming.action_type
                and edge.action_target == incoming.action_target
            ):
                KnowledgeGraph.__absorb_edge(target=edge, incoming=incoming)
                return True

        return False

    @staticmethod
    def __absorb_edge(*, target: GraphEdge, incoming: GraphEdge) -> None:
        """
        Folds an incoming edge's destination, count, and metadata into a target.
        """

        target.destination_hash = incoming.destination_hash
        target.count += incoming.count
        target.last_seen = max(target.last_seen or 0, incoming.last_seen or 0)
        if not target.coord_bucket and incoming.coord_bucket:
            target.coord_bucket = incoming.coord_bucket
        if not target.coord_region and incoming.coord_region:
            target.coord_region = incoming.coord_region
        if not target.element_category and incoming.element_category:
            target.element_category = incoming.element_category
        if not target.value and incoming.value:
            target.value = incoming.value
        if target.semantics is None and incoming.semantics is not None:
            target.semantics = incoming.semantics

    @staticmethod
    def __backfill_hashes(*, node: GraphNode, source: Dict[str, Any]) -> None:
        """
        Fills any missing MLSIA hashes on a node from a persisted row.
        """

        if not node.activity_hash and source.get("activity_hash"):
            node.activity_hash = source["activity_hash"]
        if not node.xml_hash and source.get("xml_hash"):
            node.xml_hash = source["xml_hash"]
        if not node.interaction_hash and source.get("interaction_hash"):
            node.interaction_hash = source["interaction_hash"]
        if not node.structure_hash and source.get("structure_hash"):
            node.structure_hash = source["structure_hash"]

    async def add_screen(
        self, *, state: ScreenState, description: Optional[str] = None
    ) -> GraphNode:
        """
        Registers a screen observation, merging fuzzy duplicates and counting visits.
        """

        await self.__provider.store_observation(screen=state, description=description)

        canonical = self.__resolve_for_state(state)
        now = int(time.time())
        existing = self.__nodes.get(canonical)

        if existing:
            existing.visit_count += 1
            existing.last_seen = now
            if description and not self.has_meaningful_description(existing.description):
                existing.description = description
            self.__backfill_state_hashes(node=existing, state=state)
            return existing

        node = GraphNode(
            visual_hash=state.visual_hash,
            activity=state.activity,
            description=description,
            first_seen=now,
            last_seen=now,
            visit_count=1,
            activity_hash=state.activity_hash,
            xml_hash=state.xml_hash,
            interaction_hash=state.interaction_hash,
            structure_hash=state.structure_hash,
        )
        self.__nodes[state.visual_hash] = node
        return node

    @staticmethod
    def __backfill_state_hashes(*, node: GraphNode, state: ScreenState) -> None:
        """
        Fills any missing MLSIA hashes on a node from a fresh screen state.
        """

        if not node.activity_hash and state.activity_hash:
            node.activity_hash = state.activity_hash
        if not node.xml_hash and state.xml_hash:
            node.xml_hash = state.xml_hash
        if not node.interaction_hash and state.interaction_hash:
            node.interaction_hash = state.interaction_hash
        if not node.structure_hash and state.structure_hash:
            node.structure_hash = state.structure_hash

    async def update_rich_description(self, *, visual_hash: str, rich_description: str) -> None:
        """
        Stores a rich markdown description on the node and persists it.
        """

        canonical = self.__resolve(visual_hash)
        node = self.__nodes.get(canonical)
        if node:
            node.rich_description = rich_description
        await self.__provider.update_rich_description(
            visual_hash=canonical, rich_description=rich_description
        )

    async def mark_exhausted(self, *, visual_hash: str) -> None:
        """
        Records that a screen is fully explored, in memory and in persistence.
        """

        canonical = self.__resolve(visual_hash)
        node = self.__nodes.get(canonical)
        if node:
            node.exhausted = True
        await self.__provider.mark_exhausted(visual_hash=canonical)

    async def record_relevance(self, *, visual_hash: str, relevance: FocusRelevance) -> None:
        """
        Records how a screen relates to the focus, in memory and in persistence.
        """

        canonical = self.__resolve(visual_hash)
        node = self.__nodes.get(canonical)
        if node:
            node.relevance = relevance
        await self.__provider.set_relevance(visual_hash=canonical, relevance=relevance.value)

    def relevance_of(self, *, visual_hash: str) -> FocusRelevance:
        """
        Returns the recorded focus relevance for a screen, UNSCOPED when unknown.
        """

        node = self.__nodes.get(self.__resolve(visual_hash))
        return node.relevance if node else FocusRelevance.UNSCOPED

    @staticmethod
    def __coerce_relevance(value: Optional[str]) -> FocusRelevance:
        """
        Coerces a persisted relevance string into the enum, defaulting to UNSCOPED.
        """

        if value is None:
            return FocusRelevance.UNSCOPED
        try:
            return FocusRelevance(value)
        except ValueError:
            return FocusRelevance.UNSCOPED

    async def record_category(self, *, visual_hash: str, category: ScreenCategory) -> None:
        """
        Records a screen's functional category, in memory and in persistence.
        """

        canonical = self.__resolve(visual_hash)
        node = self.__nodes.get(canonical)
        if node:
            node.category = category
        await self.__provider.set_category(visual_hash=canonical, category=category.value)

    def category_of(self, *, visual_hash: str) -> ScreenCategory:
        """
        Returns the recorded category for a screen, OTHER when unknown.
        """

        node = self.__nodes.get(self.__resolve(visual_hash))
        return node.category if node else ScreenCategory.OTHER

    @staticmethod
    def __coerce_category(value: Optional[str]) -> ScreenCategory:
        """
        Coerces a persisted category string into the enum, defaulting to OTHER.
        """

        if value is None:
            return ScreenCategory.OTHER
        try:
            return ScreenCategory(value)
        except ValueError:
            return ScreenCategory.OTHER

    async def record_content(self, *, visual_hash: str, content: ScreenContent) -> None:
        """
        Records a screen's structured content, in memory and in persistence.
        """

        canonical = self.__resolve(visual_hash)
        node = self.__nodes.get(canonical)
        if node:
            node.content = content
        await self.__provider.set_content(
            visual_hash=canonical, content_json=content.model_dump_json()
        )

    def content_of(self, *, visual_hash: str) -> Optional[ScreenContent]:
        """
        Returns the recorded structured content for a screen, None when unknown.
        """

        node = self.__nodes.get(self.__resolve(visual_hash))
        return node.content if node else None

    @staticmethod
    def __coerce_content(value: Optional[str]) -> Optional[ScreenContent]:
        """
        Parses persisted content JSON into the value object, None when absent or invalid.
        """

        if not value:
            return None
        try:
            return ScreenContent.model_validate_json(value)
        except ValueError:
            return None

    @staticmethod
    def __coerce_semantics(value: Optional[str]) -> Optional[LinkSemantics]:
        """
        Parses persisted link-semantics JSON into the model, None when absent or invalid.
        """

        if not value:
            return None
        try:
            return LinkSemantics.model_validate_json(value)
        except ValueError:
            return None

    async def record_transition(
        self, *, source_hash: str, action: Action, destination_hash: str
    ) -> None:
        """
        Records a screen-to-screen transition, deduplicating by source and target.
        """

        canonical_src = self.__resolve(source_hash)
        canonical_dst = self.__resolve(destination_hash)

        action_type = action.action_type.value
        action_target = action.natural_language_target or action.target or ""
        coord_bucket = action.bounds.coord_bucket() if action.bounds else None
        now = int(time.time())

        edges = self.__edges.setdefault(canonical_src, [])

        canonical_target = action_target
        if coord_bucket:
            for edge in edges:
                if edge.action_type == action_type and edge.coord_bucket == coord_bucket:
                    canonical_target = edge.action_target
                    break

        persisted_action = (
            action
            if canonical_target == action_target
            else action.model_copy(update={"natural_language_target": canonical_target})
        )
        await self.__provider.store_transition(
            source_hash=canonical_src, action=persisted_action, destination_hash=canonical_dst
        )

        for edge in edges:
            if edge.action_type == action_type and edge.action_target == canonical_target:
                edge.destination_hash = canonical_dst
                edge.count += 1
                edge.last_seen = now
                if not edge.coord_bucket and coord_bucket:
                    edge.coord_bucket = coord_bucket
                if not edge.coord_region and action.region:
                    edge.coord_region = action.region
                if not edge.element_category and action.element_category:
                    edge.element_category = action.element_category
                if not edge.value and action.text:
                    edge.value = action.text
                if edge.semantics is None:
                    edge.semantics = LinkSemantics.of(action=action)
                return

        edges.append(
            GraphEdge(
                source_hash=canonical_src,
                destination_hash=canonical_dst,
                action_type=action_type,
                action_target=canonical_target,
                coord_bucket=coord_bucket,
                coord_region=action.region,
                element_category=action.element_category,
                value=action.text,
                semantics=LinkSemantics.of(action=action),
                count=1,
                first_seen=now,
                last_seen=now,
            )
        )

    def get_neighbors(self, *, visual_hash: str) -> List[GraphEdge]:
        """
        Returns all outgoing transition edges from a screen.
        """

        return list(self.__edges.get(self.__resolve(visual_hash), []))

    def get_inbound_edge(self, *, destination_hash: str) -> Optional[Tuple[str, GraphEdge]]:
        """
        Returns the first known inbound edge leading to a destination.
        """

        canonical_dst = self.__resolve(destination_hash)
        for source_hash, edges in self.__edges.items():
            for edge in edges:
                if edge.destination_hash == canonical_dst:
                    return source_hash, edge
        return None

    def get_unexplored_screens(self, *, max_visits: int = 2) -> List[GraphNode]:
        """
        Returns screens visited fewer than max_visits times.
        """

        return [node for node in self.__nodes.values() if node.visit_count < max_visits]

    def exhausted_hashes(self) -> Set[str]:
        """
        Canonical hashes of screens already marked fully explored.
        """

        return {hash_value for hash_value, node in self.__nodes.items() if node.exhausted}

    def get_screen(self, *, visual_hash: str) -> Optional[GraphNode]:
        """
        Returns a single screen node, or None if unknown.
        """

        return self.__nodes.get(self.__resolve(visual_hash))

    def has_screen(self, *, visual_hash: str) -> bool:
        """
        Whether a screen (or a fuzzy match) has been seen before.
        """

        return self.__resolve(visual_hash) in self.__nodes

    def prune_foreign_screens(self, *, package: str) -> int:
        """
        Drops screens captured on a different app package, returning the count.

        A crawl can briefly land on the device launcher or another app; those
        screens are not part of the target app and must not appear in its graph,
        report, or documentation. Their nodes, edges, and aliases are removed so
        every downstream artifact is scoped to the target package.
        """

        foreign = {
            visual_hash
            for visual_hash, node in self.__nodes.items()
            if self.__is_foreign(activity=node.activity, package=package)
        }
        if not foreign:
            return 0

        for visual_hash in foreign:
            self.__nodes.pop(visual_hash, None)
            self.__edges.pop(visual_hash, None)

        for edges in self.__edges.values():
            edges[:] = [edge for edge in edges if edge.destination_hash not in foreign]

        self.__aliases = {
            raw: canonical for raw, canonical in self.__aliases.items() if canonical not in foreign
        }
        return len(foreign)

    @staticmethod
    def __is_foreign(*, activity: str, package: str) -> bool:
        """
        Whether an activity belongs to a known package other than the target.

        Unknown or empty activities are kept: only a clearly different package
        (e.g. the launcher) is treated as foreign.
        """

        observed = KnowledgeGraph.package_of(activity)
        return bool(observed) and observed not in (package, "unknown")

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns summary statistics about the knowledge graph.
        """

        activities = {node.activity for node in self.__nodes.values()}

        return {
            "unique_screens": self.node_count,
            "total_transitions": self.edge_count,
            "total_visits": sum(node.visit_count for node in self.__nodes.values()),
            "unique_activities": len(activities),
            "activities": sorted(activities),
            "unexplored": len(self.get_unexplored_screens()),
        }

    def count_category_taps(self, *, visual_hash: str, category: str) -> int:
        """
        Counts sampling taps for an element category across the current activity.
        """

        canonical = self.__resolve(visual_hash)
        current_node = self.__nodes.get(canonical)
        activity = current_node.activity if current_node else None

        if activity is None:
            return self.__count_edges(hash_value=canonical, category=category)

        normalized = self.normalize_activity(activity)
        total = 0
        for node_hash, node in self.__nodes.items():
            if self.normalize_activity(node.activity) != normalized:
                continue
            total += self.__count_edges(hash_value=self.__resolve(node_hash), category=category)
        return total

    def __count_edges(self, *, hash_value: str, category: str) -> int:
        """
        Counts sampling-type edges on a screen matching an element category.
        """

        return sum(
            1
            for edge in self.__edges.get(hash_value, [])
            if edge.element_category == category and edge.action_type in SAMPLING_ACTION_TYPES
        )

    def get_tried_actions(self, *, visual_hash: str) -> List[TriedAction]:
        """
        Returns the actions already exercised on a screen and where each led.
        """

        result: List[TriedAction] = []
        for edge in self.__edges.get(self.__resolve(visual_hash), []):
            if edge.action_type == ActionType.BACK.value:
                continue
            destination = self.__nodes.get(edge.destination_hash)
            result.append(
                TriedAction(
                    action_type=edge.action_type,
                    target=edge.action_target,
                    coord_bucket=edge.coord_bucket,
                    destination_hash=edge.destination_hash,
                    destination_description=destination.description if destination else None,
                )
            )
        return result

    def build_exploration_context(
        self,
        *,
        current_hash: Optional[str] = None,
        depth: Optional[int] = None,
        parent_description: Optional[str] = None,
        fully_scanned_count: Optional[int] = None,
        fully_scanned: Optional[Set[str]] = None,
        recent_actions: Optional[List[ActionOutcome]] = None,
        depth_floor_active: bool = False,
        min_dfs_depth: int = 0,
        focus: Optional[str] = None,
    ) -> str:
        """
        Formats the knowledge-graph state as LLM-readable context for the scan prompt.
        """

        lines: List[str] = []

        if focus and focus.strip():
            lines.append(f"FOCUS: {focus.strip()}")

        if recent_actions:
            lines.append(self.__format_recent_actions(recent_actions=recent_actions))

        scanned = (
            f", {fully_scanned_count} fully explored" if fully_scanned_count is not None else ""
        )
        lines.append(
            f"EXPLORATION PROGRESS: {self.node_count} screens discovered{scanned}, "
            f"{self.edge_count} transitions"
        )

        if depth is not None:
            lines.append(f"DEPTH: {depth}")

        if depth_floor_active:
            lines.append(
                f"DEPTH FLOOR - the previous turn declared content_exhausted at depth {depth} "
                f"(< minimum {min_dfs_depth}). Pick ANY untried interactive element on this "
                "screen. Do NOT set content_exhausted=true unless the screen has ZERO clickable items."
            )

        if parent_description:
            lines.append(f"PARENT SCREEN: {parent_description}")

        node = self.__nodes.get(self.__resolve(current_hash)) if current_hash else None
        self.__append_activity_coverage(lines=lines, current_hash=current_hash, node=node)

        if node and node.description:
            lines.append(f"CURRENT SCREEN: {node.description}")

        if current_hash:
            self.__append_tried_actions(
                lines=lines, current_hash=current_hash, fully_scanned=fully_scanned
            )

        self.__append_recent_discoveries(lines=lines)

        return "\n".join(lines)

    @staticmethod
    def __format_recent_actions(*, recent_actions: List[ActionOutcome]) -> str:
        """
        Renders recent action outcomes as reactive feedback for the prompt.
        """

        lines = []
        for outcome in recent_actions:
            label = f'- {outcome.kind.value} "{outcome.target}"'
            if outcome.expected is not None:
                label += f" (expected {outcome.expected.value})"
            lines.append(f"{label} -> {KnowledgeGraph.__outcome_indicator(outcome=outcome)}")
        return "RECENT ACTIONS (oldest to newest):\n" + "\n".join(lines)

    @staticmethod
    def __outcome_indicator(*, outcome: ActionOutcome) -> str:
        """
        Describes how an action's result compares to what it predicted.
        """

        if not outcome.success:
            return "FAILED"
        if outcome.screen_changed:
            return "ok, new screen"
        if outcome.expected is not None and outcome.expected.implies_transition:
            return "NO change despite expecting a transition - element may be inert, do not retap"
        return "ok, NO screen change"

    def __append_activity_coverage(
        self, *, lines: List[str], current_hash: Optional[str], node: Optional[GraphNode]
    ) -> None:
        """
        Adds known-activity coverage and a revisit nudge toward unseen sections.
        """

        known = sorted(
            {self.normalize_activity(n.activity) for n in self.__nodes.values() if n.activity}
        )
        if known:
            lines.append(f"KNOWN ACTIVITIES ({len(known)}): " + ", ".join(known))

        if not (node and len(known) > 1):
            return

        current_activity = self.normalize_activity(node.activity)
        canonical = self.__resolve(current_hash) if current_hash else None
        siblings = [
            h
            for h, n in self.__nodes.items()
            if self.normalize_activity(n.activity) == current_activity and h != canonical
        ]
        if siblings or node.visit_count > 1:
            lines.append(
                "REVISIT - this activity is already mapped. PRIORITIZE navigating to an UNSEEN "
                "activity over tapping another element inside this one.\n"
                "- Prefer P1 global_navigation tabs/drawer items that jump to a different section.\n"
                "- Skip P3 content items (they lead to detail screens within this same activity).\n"
                "- If every visible P1 element leads back to a KNOWN activity, use BACK to climb "
                "out and find a different entry point."
            )

    def __append_tried_actions(
        self, *, lines: List[str], current_hash: str, fully_scanned: Optional[Set[str]] = None
    ) -> None:
        """
        Adds the already-tried actions and forbidden targets for the current screen.
        """

        tried = self.get_tried_actions(visual_hash=current_hash)
        if not tried:
            lines.append("ALREADY TRIED ON THIS SCREEN: (none -- this screen is fresh)")
            return

        explored: AbstractSet[str] = fully_scanned or frozenset()
        lines.append("ALREADY TRIED ON THIS SCREEN:")
        for action in tried[:MAX_TRIED_IN_CONTEXT]:
            entry = f"- {action.action_type}"
            if action.target:
                entry += f' "{action.target}"'
            if action.destination_description:
                entry += f" -> {action.destination_description}"
            if action.destination_hash in explored:
                entry += " [already fully explored - low value]"
            lines.append(entry)

        excess = len(tried) - MAX_TRIED_IN_CONTEXT
        if excess > 0:
            lines.append(f"... and {excess} more tried")
        lines.append(f"ACTIONS TRIED: {len(tried)}")

        forbidden = sorted({action.target for action in tried if action.target})
        if forbidden:
            lines.append(
                "FORBIDDEN TARGETS (do NOT select any of these): "
                + ", ".join(f'"{target}"' for target in forbidden)
            )

    def __append_recent_discoveries(self, *, lines: List[str]) -> None:
        """
        Adds the five most recently discovered described screens.
        """

        described = [
            node
            for node in self.__nodes.values()
            if node.first_seen is not None and node.description
        ]
        if not described:
            return

        described.sort(key=lambda node: node.first_seen or 0, reverse=True)
        lines.append("RECENT DISCOVERIES:")
        for node in described[:5]:
            lines.append(f"- {node.description}")

    def find_path(
        self, *, start_hash: str, end_hash: str, max_depth: int = 50
    ) -> Optional[List[Tuple[str, Optional[GraphEdge]]]]:
        """
        Returns the shortest path between two screens, or None if unreachable.
        """

        return GraphAlgorithms.find_path(
            nodes=self.__nodes,
            edges=self.__edges,
            start_hash=self.__resolve(start_hash),
            end_hash=self.__resolve(end_hash),
            max_depth=max_depth,
        )

    def find_all_paths(
        self, *, start_hash: str, end_hash: str, max_depth: int = 10
    ) -> List[List[Tuple[str, Optional[GraphEdge]]]]:
        """
        Returns every path between two screens up to max_depth.
        """

        return GraphAlgorithms.find_all_paths(
            nodes=self.__nodes,
            edges=self.__edges,
            start_hash=self.__resolve(start_hash),
            end_hash=self.__resolve(end_hash),
            max_depth=max_depth,
        )

    def is_reachable(self, *, start_hash: str, end_hash: str, max_depth: int = 100) -> bool:
        """
        Whether end_hash is reachable from start_hash.
        """

        return (
            self.find_path(start_hash=start_hash, end_hash=end_hash, max_depth=max_depth)
            is not None
        )

    def detect_cycles(self, *, start_hash: Optional[str] = None) -> List[List[str]]:
        """
        Detects cycles in the graph, optionally restricted to those reachable from a node.
        """

        if start_hash:
            resolved = self.__resolve(start_hash)
            start_nodes = [resolved] if resolved in self.__nodes else []
        else:
            start_nodes = list(self.__nodes.keys())

        return GraphAlgorithms.detect_cycles(edges=self.__edges, start_nodes=start_nodes)

    def get_connected_component(self, *, start_hash: str) -> Set[str]:
        """
        Returns all nodes forward-reachable from a screen.
        """

        resolved = self.__resolve(start_hash)
        if resolved not in self.__nodes:
            return set()
        return GraphAlgorithms.connected_component(edges=self.__edges, start_hash=resolved)

    def get_reverse_connected_component(self, *, end_hash: str) -> Set[str]:
        """
        Returns all nodes that can reach a screen.
        """

        resolved = self.__resolve(end_hash)
        if resolved not in self.__nodes:
            return set()
        return GraphAlgorithms.reverse_connected_component(edges=self.__edges, end_hash=resolved)

    def get_graph_diameter(self) -> Optional[int]:
        """
        Returns the longest shortest-path between any two screens, or None.
        """

        return GraphAlgorithms.diameter(nodes=self.__nodes, edges=self.__edges)
