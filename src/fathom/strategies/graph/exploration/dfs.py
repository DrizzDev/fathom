"""
Depth-first-search bookkeeping and navigation planning for exploration.

The exploration graph drives a depth-first walk of an application's screens.
The mutable DFS state -- the current phase, the path back to the root, the
recovery queue, and the set of fully scanned screens -- lives on
:class:`DfsState`, owned by the node provider for the lifetime of a run.
:class:`DfsNavigator` reads that state together with the persistent
:class:`KnowledgeGraph` to reconstruct paths, recover orphaned screens, and
compute the BACK-then-forward action sequence between two paths.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from logging import getLogger
from typing import Deque, Dict, List, Optional, Set, Tuple

from fathom.constants import ActionType
from fathom.constants.exploration import BFSPhase
from fathom.infrastructure.memory.knowledge_graph import GraphEdge, KnowledgeGraph
from fathom.schemas.actions import Action
from fathom.schemas.checkpoint import ExplorationCheckpoint
from fathom.schemas.exploration import BFSQueueEntry

logger = getLogger(__name__)


@dataclass
class DfsState:
    """
    Mutable depth-first-search bookkeeping for a single exploration run.

    Held on the node provider (not the LangGraph state dict) because the
    recovery queue, the path of ``(hash, action)`` pairs, and the per-screen
    exhaustion counters are awkward to thread through a serialisable state and
    are read directly by the navigator and routers.
    """

    phase: BFSPhase = BFSPhase.SCAN
    root_hash: Optional[str] = None
    scanning_hash: Optional[str] = None
    current_path: List[Tuple[str, Action]] = field(default_factory=list)
    pending_nav: List[Action] = field(default_factory=list)
    bfs_queue: Deque[BFSQueueEntry] = field(default_factory=deque)
    fully_scanned: Set[str] = field(default_factory=set)
    exhaustion_retries: Dict[str, int] = field(default_factory=dict)
    stalled_routes: int = 0
    steps_since_new_screen: int = 0

    @property
    def depth(self) -> int:
        """
        Number of edges traversed from the root on the current DFS path.
        """

        return len(self.current_path)

    def to_checkpoint(self) -> ExplorationCheckpoint:
        """
        Captures the resumable DFS state as a serializable checkpoint.

        Transient per-step fields (the screen being scanned, pending navigation,
        the stall counter) are intentionally omitted: they are rebuilt each run.
        """

        return ExplorationCheckpoint(
            phase=self.phase,
            root_hash=self.root_hash,
            current_path=list(self.current_path),
            bfs_queue=list(self.bfs_queue),
            fully_scanned=sorted(self.fully_scanned),
            exhaustion_retries=dict(self.exhaustion_retries),
        )

    @classmethod
    def from_checkpoint(cls, *, checkpoint: ExplorationCheckpoint) -> "DfsState":
        """
        Rehydrates a DFS state from a saved checkpoint, restoring the deque and set.
        """

        return cls(
            phase=checkpoint.phase,
            root_hash=checkpoint.root_hash,
            current_path=list(checkpoint.current_path),
            bfs_queue=deque(checkpoint.bfs_queue),
            fully_scanned=set(checkpoint.fully_scanned),
            exhaustion_retries=dict(checkpoint.exhaustion_retries),
        )


class DfsNavigator:
    """
    Path reconstruction and recovery-navigation planning over the screen graph.
    """

    def __init__(self, *, dfs: DfsState, knowledge_graph: KnowledgeGraph) -> None:
        self.__dfs = dfs
        self.__knowledge_graph = knowledge_graph

    def path_to_screen(self, *, screen_hash: Optional[str]) -> List[Tuple[str, Action]]:
        """
        Best-effort reconstruction of the path to an already-visited screen.
        """

        dfs = self.__dfs
        if not screen_hash or screen_hash == dfs.root_hash:
            return []

        for entry in dfs.bfs_queue:
            if entry.screen_hash == screen_hash:
                return list(entry.path_from_root)

        return list(dfs.current_path[:-1])

    def find_orphaned_screens(self) -> List[BFSQueueEntry]:
        """
        Discover unscanned screens, each with a replayable path from the root.

        Used when backtracking reaches the root but the knowledge graph still
        holds screens that were never fully scanned -- including, on a relaunch,
        the whole persisted frontier. Each entry carries the full root-anchored
        action path so recovery can navigate to a deep screen, not just a child
        of the current one.
        """

        dfs = self.__dfs
        orphans: List[BFSQueueEntry] = []

        for visual_hash in self.__knowledge_graph.nodes:
            if visual_hash in dfs.fully_scanned or visual_hash == dfs.root_hash:
                continue

            path = self.__path_from_root(screen_hash=visual_hash)
            if not path:
                continue

            source_hash, action = path[-1]
            orphans.append(
                BFSQueueEntry(
                    screen_hash=visual_hash,
                    parent_hash=source_hash,
                    action_from_parent=action,
                    depth=len(path),
                    path_from_root=path,
                )
            )

        # On a focused run, recover on-focus screens before off-focus ones so the
        # frontier sweep heads toward the target section first; off-focus screens
        # sink to the bottom but stay reachable. Depth breaks ties, keeping the
        # nearest-first ordering (and, on a broad-coverage run where every screen
        # is UNSCOPED, the relevance key is constant so depth alone decides).
        orphans.sort(key=lambda entry: (self.__recovery_priority(entry=entry), entry.depth))
        return orphans

    def __recovery_priority(self, *, entry: BFSQueueEntry) -> int:
        """
        Focus-relevance recovery rank for a frontier screen; lower is recovered first.
        """

        relevance = self.__knowledge_graph.relevance_of(visual_hash=entry.screen_hash)
        return relevance.recovery_priority

    def __path_from_root(self, *, screen_hash: str) -> List[Tuple[str, Action]]:
        """
        Best-effort replayable path from the run's root to a screen.

        Prefers a root-anchored shortest path so recovery can be replayed from
        anywhere; falls back to the single known inbound hop when no rooted path
        exists. Returns an empty list when the screen is unreachable.
        """

        dfs = self.__dfs
        knowledge_graph = self.__knowledge_graph

        if dfs.root_hash:
            graph_path = knowledge_graph.find_path(start_hash=dfs.root_hash, end_hash=screen_hash)
            if graph_path and len(graph_path) > 1:
                hops: List[Tuple[str, Action]] = []
                for index in range(1, len(graph_path)):
                    source_hash = graph_path[index - 1][0]
                    edge = graph_path[index][1]
                    if edge is None:
                        continue
                    hops.append((source_hash, self.__edge_action(edge=edge)))
                return hops

        inbound = knowledge_graph.get_inbound_edge(destination_hash=screen_hash)
        if inbound is None:
            return []
        source_hash, edge = inbound
        return [(source_hash, self.__edge_action(edge=edge))]

    @staticmethod
    def __edge_action(*, edge: GraphEdge) -> Action:
        """
        Builds a target-grounded replay action from a transition edge.
        """

        try:
            action_type = ActionType(edge.action_type)
        except ValueError:
            action_type = ActionType.TAP

        return Action(
            action_type=action_type,
            confidence=1.0,
            target=edge.action_target or "frontier recovery",
            rationale=f"DFS recovery: navigate to frontier via {action_type.value}",
        )

    @staticmethod
    def compute_navigation(
        *,
        current_path: List[Tuple[str, Action]],
        target_path: List[Tuple[str, Action]],
    ) -> List[Action]:
        """
        BACK-then-forward action plan to move from ``current_path`` to ``target_path``.

        Ascends to the lowest common ancestor of the two paths with hardware
        BACK presses, then replays the target's forward edges from there.
        """

        common_length = 0
        for index in range(min(len(current_path), len(target_path))):
            current_screen, current_action = current_path[index]
            target_screen, target_action = target_path[index]
            if current_screen == target_screen and current_action == target_action:
                common_length = index + 1
            else:
                break

        actions: List[Action] = []

        backs_needed = len(current_path) - common_length
        for _ in range(backs_needed):
            actions.append(
                Action(
                    action_type=ActionType.BACK,
                    confidence=1.0,
                    target="back navigation",
                    rationale="DFS recovery: navigating to common ancestor",
                )
            )

        for index in range(common_length, len(target_path)):
            _, forward_action = target_path[index]
            actions.append(forward_action)

        return actions
