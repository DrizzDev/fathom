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
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.schemas.actions import Action
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

    @property
    def depth(self) -> int:
        """
        Number of edges traversed from the root on the current DFS path.
        """

        return len(self.current_path)


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
        Discover unscanned screens reachable via a known inbound transition.

        Used when backtracking reaches the root but the knowledge graph still
        holds screens that were discovered via transitions yet never fully
        scanned -- typically caused by BACK overshooting or landing on an
        already-scanned screen that had unexplored neighbours.
        """

        dfs = self.__dfs
        knowledge_graph = self.__knowledge_graph
        orphans: List[BFSQueueEntry] = []

        for visual_hash in knowledge_graph.nodes:
            if visual_hash in dfs.fully_scanned or visual_hash == dfs.root_hash:
                continue

            inbound = knowledge_graph.get_inbound_edge(destination_hash=visual_hash)
            if inbound is None:
                continue
            source_hash, edge = inbound

            try:
                action_type = ActionType(edge.action_type)
            except ValueError:
                action_type = ActionType.TAP

            action = Action(
                action_type=action_type,
                confidence=1.0,
                target=edge.action_target or "orphan recovery",
                rationale=f"DFS recovery: navigate to orphaned screen via {action_type.value}",
            )
            orphans.append(
                BFSQueueEntry(
                    screen_hash=visual_hash,
                    parent_hash=source_hash,
                    action_from_parent=action,
                    depth=1,
                    path_from_root=[(source_hash, action)],
                )
            )

        return orphans

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
