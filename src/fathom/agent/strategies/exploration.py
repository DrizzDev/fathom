from __future__ import annotations

import asyncio
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union

from fathom.agent.strategies.base import ExecutionStrategy
from fathom.constants import ActionType, StrategyStatus
from fathom.infrastructure.memory.knowledge_graph import GraphNode, KnowledgeGraph
from fathom.prompts.modes import PromptMode
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.results import AnalysisResult, StrategyResult
from fathom.schemas.screens import ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool
from fathom.utils.execution import ensure_target_package, execute_device_action

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Backward-compatible legacy classes (retained for non-BFS fallback)
# ---------------------------------------------------------------------------


class ScreenNode:
    """
    Node in the screen graph representing a unique screen state.
    Retained for backward compatibility with ExplorationGraph.
    """

    def __init__(self, fingerprint: str, activity: str) -> None:
        self.__activity = activity
        self.__fingerprint = fingerprint

        self.__visits = 0
        self.__last = 0.0
        self.__actions: Set[str] = set()
        self.__transitions: Dict[str, str] = {}

    @property
    def fingerprint(self) -> str:
        """
        Unique identifier for this screen state.
        """

        return self.__fingerprint

    @property
    def activity(self) -> str:
        """
        Activity name for this screen.
        """

        return self.__activity

    @property
    def visits(self) -> int:
        """
        Number of times this screen has been visited.
        """

        return self.__visits

    @property
    def actions(self) -> Set[str]:
        """
        Actions that can be performed from this screen.
        """

        return self.__actions

    @property
    def transitions(self) -> Dict[str, str]:
        """
        Transitions from this screen to other screens.
        """

        return self.__transitions

    def record_visit(self) -> None:
        """
        Records a visit to this screen.
        """

        self.__visits += 1
        self.__last = time.time()

    def record_action(self, description: str, destination: str) -> None:
        """
        Records an action and its result.
        """

        self.__actions.add(description)
        self.__transitions[description] = destination

    def should_explore(self, limit: int = 5) -> bool:
        """
        Checks if exploration limit reached.
        """

        return self.__visits < limit


class ExplorationGraph:
    """
    In-memory graph of discovered screens and transitions.
    Retained for backward compatibility; new code should use KnowledgeGraph.
    """

    HAMMING_THRESHOLD = 12

    def __init__(self) -> None:
        self.__nodes: Dict[str, ScreenNode] = {}
        self.__edges: List[Tuple[str, str, str]] = []
        self.__hash_aliases: Dict[str, str] = {}

    @staticmethod
    def _hamming_distance(hash1: str, hash2: str) -> int:
        if len(hash1) != len(hash2):
            return 256
        try:
            return bin(int(hash1, 16) ^ int(hash2, 16)).count("1")
        except ValueError:
            return 256

    def _resolve_canonical(self, visual_hash: str) -> str:
        if visual_hash in self.__nodes:
            return visual_hash
        if visual_hash in self.__hash_aliases:
            return self.__hash_aliases[visual_hash]

        best_hash = None
        best_distance = self.HAMMING_THRESHOLD + 1
        for existing_hash in self.__nodes:
            d = self._hamming_distance(visual_hash, existing_hash)
            if d < best_distance:
                best_distance = d
                best_hash = existing_hash

        if best_hash is not None and best_distance <= self.HAMMING_THRESHOLD:
            self.__hash_aliases[visual_hash] = best_hash
            return best_hash
        return visual_hash

    @property
    def nodes(self) -> Dict[str, ScreenNode]:
        """
        All discovered screen nodes.
        """

        return self.__nodes

    @property
    def edges(self) -> List[Tuple[str, str, str]]:
        """
        All transitions between screens.
        """

        return self.__edges

    def resolve_hash(self, visual_hash: str) -> str:
        """Public API: resolve a raw hash to its canonical form."""
        return self._resolve_canonical(visual_hash)

    def add_screen(self, state: ScreenState) -> ScreenNode:
        """
        Adds or updates a screen, using fuzzy hash matching.
        """

        key = self._resolve_canonical(state.visual_hash)
        if key not in self.__nodes:
            self.__nodes[key] = ScreenNode(fingerprint=key, activity=state.activity)

        node = self.__nodes[key]
        node.record_visit()

        return node

    def record_transition(self, origin: str, destination: str, action: str) -> None:
        """
        Records a transition using canonical hashes.
        """

        canonical_origin = self._resolve_canonical(origin)
        canonical_dest = self._resolve_canonical(destination)

        if canonical_origin in self.__nodes:
            self.__nodes[canonical_origin].record_action(
                description=action, destination=canonical_dest
            )

        self.__edges.append((canonical_origin, action, canonical_dest))

    def get_stats(self) -> Dict[str, Any]:
        """
        Calculates coverage stats.
        """

        total = sum(len(node.actions) for node in self.__nodes.values())
        unexplored = sum(1 for node in self.__nodes.values() if node.should_explore())
        activities = len({node.activity for node in self.__nodes.values()})

        return {
            "total_actions": total,
            "unexplored": unexplored,
            "activities": activities,
            "unique_screens": len(self.__nodes),
            "total_transitions": len(self.__edges),
        }


class ActionGenerator:
    """
    Generates exploratory actions for unknown UI states.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.__rng = random.Random(seed)  # nosec
        self.__failures: Dict[str, int] = defaultdict(int)

    def generate(self, node: Union[ScreenNode, GraphNode], width: int, height: int) -> Action:
        """
        Selects the best exploratory action based on visit count.
        """

        visits = node.visits if isinstance(node, ScreenNode) else node.visit_count

        if visits <= 2:
            return self.__tap(width=width, height=height)

        if visits <= 4:
            return self.__scroll()

        return self.__back()

    def __tap(self, width: int, height: int) -> Action:
        """
        Random tap.
        """

        _ = width
        _ = height

        x = self.__rng.randint(50, 950)
        y = self.__rng.randint(100, 900)

        return Action(
            confidence=0.3,
            rationale="Exploratory tap",
            action_type=ActionType.TAP,
            target=f"random tap at ({x}, {y})",
            bounds=Bounds(x=x, y=y, width=50, height=50),
        )

    def __scroll(self) -> Action:
        """
        Random scroll.
        """

        direction = self.__rng.choice(["up", "down"])

        return Action(
            confidence=0.4,
            action_type=ActionType.SCROLL,
            rationale=f"Scrolling {direction}",
            target=f"exploration scroll {direction}",
        )

    def __back(self) -> Action:
        """
        Back navigation.
        """

        return Action(
            confidence=0.5,
            target="back navigation",
            action_type=ActionType.BACK,
            rationale="Exploring parent path",
        )


# ---------------------------------------------------------------------------
# DFS data structures
# ---------------------------------------------------------------------------


class BFSPhase(Enum):
    """
    State machine phases for DFS-driven exploration.

    SCAN      — On the target screen. VLM identifies and taps the next untried
                element.  If the tap navigates to a new screen, DFS follows it
                (stays in SCAN on the new screen).
    BACKTRACK — Current screen fully scanned.  Press BACK to return to the
                parent screen.  If the parent is also exhausted, keep
                backtracking until we find a screen with untried elements.
    ADVANCE   — Recovery only.  When BACKTRACK reaches the root and all
                screens on the DFS path are scanned, but the KG has orphaned
                unexplored screens that were skipped (e.g. due to BACK
                overshooting), navigate to them via path replay.
    """

    SCAN = "scan"
    BACKTRACK = "backtrack"
    ADVANCE = "advance"


@dataclass
class BFSQueueEntry:
    """
    An entry in the BFS frontier queue.

    Stores enough information to navigate back to this screen from any
    position in the app using the simple-BACK strategy.
    """

    screen_hash: str
    parent_hash: str
    action_from_parent: Action
    depth: int
    path_from_root: List[Tuple[str, Action]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DFS Exploration Strategy
# ---------------------------------------------------------------------------


class ExplorationStrategy(ExecutionStrategy):
    """
    DFS-driven strategy for autonomous application mapping.

    Uses a depth-first traversal to systematically discover all screens
    reachable from the starting screen.  At each screen the VLM identifies
    untried interactive elements; tapping an element that navigates to a
    new screen causes the strategy to follow it (go deeper).  When a
    screen is exhausted the strategy backtracks via BACK until it finds a
    screen with untried elements.

    When no VisionTool or KnowledgeGraph is provided, falls back to the
    legacy random ActionGenerator behaviour for backward compatibility.
    """

    def __init__(
        self,
        device: DeviceTool,
        capture: CaptureTool,
        vision: Optional[VisionTool] = None,
        *,
        max_steps: int = 100,
        timeout: float = 3600.0,
        seed: Optional[int] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        target_package: Optional[str] = None,
    ) -> None:
        self.__device = device
        self.__capture = capture

        self.__vision = vision
        self.__max_steps = max_steps

        self.__steps = 0
        self.__timeout = timeout

        self.__target_package = target_package
        self.__knowledge_graph = knowledge_graph
        self.__graph = ExplorationGraph()
        self.__generator = ActionGenerator(seed=seed)

        self.__start = time.time()
        self.__last: Optional[Action] = None
        self.__current: Optional[ScreenState] = None

        # --- DFS state ---
        self.__bfs_enabled = vision is not None and knowledge_graph is not None
        self.__bfs_queue: Deque[BFSQueueEntry] = deque()
        self.__phase: BFSPhase = BFSPhase.SCAN
        self.__scanning_hash: Optional[str] = None
        self.__current_path: List[Tuple[str, Action]] = []
        self.__pending_nav: List[Action] = []
        self.__root_hash: Optional[str] = None
        self.__fully_scanned: Set[str] = set()

    @property
    def name(self) -> str:
        """
        Strategy name.
        """

        return "exploration"

    @property
    def graph(self) -> ExplorationGraph:
        """
        Session-level exploration graph (backward compat).
        """

        return self.__graph

    @property
    def knowledge_graph(self) -> Optional[KnowledgeGraph]:
        """
        Persistent knowledge graph, if configured.
        """

        return self.__knowledge_graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_step(self) -> StrategyResult:
        """
        Executes one discovery step.

        When DFS is enabled (VisionTool + KnowledgeGraph present) the step
        is governed by the DFS state machine. Otherwise falls back to the
        legacy random-action approach.
        """

        if self.__bfs_enabled:
            return await self.__execute_bfs_step()

        return await self.__execute_legacy_step()

    async def should_continue(self) -> bool:
        """
        Stop conditions.
        """

        if self.__steps >= self.__max_steps:
            return False

        if (time.time() - self.__start) >= self.__timeout:
            return False

        # DFS-specific: exploration is complete when the recovery queue is
        # empty and no pending navigation remains.
        return not (
            self.__bfs_enabled
            and self.__phase == BFSPhase.ADVANCE
            and not self.__bfs_queue
            and not self.__pending_nav
        )

    def get_progress(self) -> Dict[str, Any]:
        """
        Discovery metrics.
        """

        stats = (
            self.__knowledge_graph.get_stats()
            if self.__knowledge_graph
            else self.__graph.get_stats()
        )

        progress: Dict[str, Any] = {
            "steps": self.__steps,
            "stats": stats,
            "elapsed": time.time() - self.__start,
        }

        if self.__bfs_enabled:
            progress["bfs_queue_size"] = len(self.__bfs_queue)
            progress["bfs_phase"] = self.__phase.value
            progress["current_depth"] = len(self.__current_path)
            progress["screens_fully_scanned"] = len(self.__fully_scanned)

        return progress

    # ------------------------------------------------------------------
    # Package scope enforcement
    # ------------------------------------------------------------------

    async def __enforce_package_scope(self) -> bool:
        """
        Check the foreground package matches the target.

        Returns ``True`` if the device is (or was recovered to) the target
        package, ``False`` if recovery failed.  When no ``target_package``
        is set, always returns ``True``.
        """

        if not self.__target_package:
            return True

        return await ensure_target_package(
            device=self.__device,
            target_package=self.__target_package,
        )

    # ------------------------------------------------------------------
    # DFS Step Execution
    # ------------------------------------------------------------------

    def __require_bfs_deps(self) -> Tuple[VisionTool, KnowledgeGraph]:
        """Verify DFS dependencies are initialised and return them narrowed.

        Returns the non-``None`` vision tool and knowledge graph so callers
        get properly narrowed types for mypy.

        Raises ``RuntimeError`` when called without vision or knowledge
        graph — both are required for DFS-driven exploration.
        """

        if self.__vision is None or self.__knowledge_graph is None:
            raise RuntimeError(
                "DFS exploration requires both a VisionTool and a "
                "KnowledgeGraph, but one or both are None."
            )
        return self.__vision, self.__knowledge_graph

    async def __execute_bfs_step(self) -> StrategyResult:
        """
        Single DFS step dispatched by phase.
        """

        self.__require_bfs_deps()  # validates; individual methods also narrow

        if self.__phase == BFSPhase.SCAN:
            return await self.__execute_scan()
        elif self.__phase == BFSPhase.BACKTRACK:
            return await self.__execute_backtrack()
        else:
            return await self.__execute_advance()

    # ---- SCAN --------------------------------------------------------

    async def __execute_scan(self) -> StrategyResult:
        """
        SCAN phase: ask VLM to identify and tap the next untried element.

        If the VLM signals ``content_exhausted`` (no more untried elements),
        transition to ADVANCE.
        """

        vision, kg = self.__require_bfs_deps()

        capture = await self.__capture.capture()
        state = self.__capture.compute_state(capture=capture)
        fingerprint = self.__graph.resolve_hash(state.visual_hash)

        # First step ever — establish root
        if self.__root_hash is None:
            self.__root_hash = fingerprint
            self.__scanning_hash = fingerprint

        # Register screen in both graphs
        self.__graph.add_screen(state=state)
        await kg.add_screen(state=state)

        # Build exploration context for VLM (DFS: depth + parent for flow awareness)
        parent_hash = self.__current_path[-1][0] if self.__current_path else None
        parent_node = kg.nodes.get(parent_hash) if parent_hash and kg else None
        parent_description = parent_node.description if parent_node else None

        kg_context = kg.build_exploration_context(
            current_hash=fingerprint,
            depth=len(self.__current_path),
            parent_description=parent_description,
        )

        # Ask VLM for next untried element
        analysis: AnalysisResult = await vision.analyze(
            intent="Explore this app to discover all screens and features",
            capture=capture,
            context=kg_context,
            mode=PromptMode.EXPLORATION,
        )

        # Persist VLM's screen description to the KG
        if analysis.screen_description:
            await kg.add_screen(state=state, description=analysis.screen_description)

        # VLM signals all elements exhausted on this screen
        if analysis.content_exhausted:
            self.__fully_scanned.add(fingerprint)
            self.__phase = BFSPhase.BACKTRACK
            logger.info(
                "Screen %s fully scanned, backtracking (depth=%d)",
                fingerprint[:8],
                len(self.__current_path),
            )

            self.__steps += 1
            return StrategyResult(
                status=StrategyStatus.CONTINUE,
                message=f"Screen {fingerprint[:8]} fully scanned, backtracking",
            )

        # Execute the VLM's recommended action with proper coordinate conversion
        action = analysis.action
        step = Step(
            action=action,
            screen_hash=fingerprint,
            step_number=self.__steps,
        )

        result = await execute_device_action(device=self.__device, action=action)
        self.__steps += 1
        self.__last = action

        await asyncio.sleep(delay=0.5)

        # Package scope enforcement
        if not await self.__enforce_package_scope():
            return StrategyResult(
                status=StrategyStatus.COMPLETE,
                message=f"Left target package {self.__target_package} and could not recover",
            )

        # Capture post-state
        post_capture = await self.__capture.capture()
        post_state = self.__capture.compute_state(capture=post_capture)
        post_hash = self.__graph.resolve_hash(post_state.visual_hash)

        # Record transition in both graphs
        self.__graph.record_transition(
            origin=fingerprint,
            destination=post_hash,
            action=action.to_description(),
        )
        await kg.record_transition(
            source_hash=fingerprint,
            action=action,
            destination_hash=post_hash,
        )

        step_result = StepResult(
            step=step,
            error=result.error,
            pre_hash=fingerprint,
            success=result.success,
            duration=result.duration,
            post_hash=post_hash,
            screen_changed=fingerprint != post_hash,
        )

        # Determine next phase (DFS: follow new screens instead of returning)
        if fingerprint != post_hash:
            # Navigated to a different screen
            is_new = not kg.has_screen(post_hash)

            # Register the new screen
            self.__graph.add_screen(state=post_state)
            await kg.add_screen(state=post_state)

            # Extend the DFS path
            new_path = list(self.__current_path) + [(fingerprint, action)]
            self.__current_path = new_path

            if post_hash in self.__fully_scanned:
                # Already exhausted — backtrack immediately
                self.__phase = BFSPhase.BACKTRACK
                logger.debug(
                    "Navigated to already-scanned screen %s, backtracking",
                    post_hash[:8],
                )
            else:
                # DFS: stay in SCAN on the new screen (go deeper)
                self.__phase = BFSPhase.SCAN
                if is_new:
                    logger.info(
                        "DFS: discovered new screen %s at depth %d",
                        post_hash[:8],
                        len(new_path),
                    )
        # else: action stayed on same screen (e.g. scroll, dropdown) → stay in SCAN

        return StrategyResult(
            step_result=step_result,
            status=StrategyStatus.CONTINUE,
            message=f"DFS scan: {action.to_description()}",
        )

    # ---- BACKTRACK ----------------------------------------------------

    async def __execute_backtrack(self) -> StrategyResult:
        """
        BACKTRACK phase: press BACK to ascend the DFS tree.

        After pressing BACK, checks where we landed:
        - Screen with untried elements → switch to SCAN.
        - Fully scanned screen with depth remaining → stay in BACKTRACK.
        - Root fully scanned → check KG for orphaned screens → ADVANCE or DONE.
        """

        _, kg = self.__require_bfs_deps()

        action = Action(
            confidence=1.0,
            target="back navigation",
            action_type=ActionType.BACK,
            rationale="DFS: backtracking from exhausted screen",
        )

        pre_capture = await self.__capture.capture()
        pre_state = self.__capture.compute_state(capture=pre_capture)
        pre_hash = self.__graph.resolve_hash(pre_state.visual_hash)

        step = Step(
            action=action,
            screen_hash=pre_hash,
            step_number=self.__steps,
        )

        result = await execute_device_action(device=self.__device, action=action)
        self.__steps += 1
        self.__last = action

        await asyncio.sleep(delay=0.5)

        # Package scope enforcement
        if not await self.__enforce_package_scope():
            return StrategyResult(
                status=StrategyStatus.COMPLETE,
                message=f"Left target package {self.__target_package} and could not recover",
            )

        post_capture = await self.__capture.capture()
        post_state = self.__capture.compute_state(capture=post_capture)
        post_hash = self.__graph.resolve_hash(post_state.visual_hash)

        # Register wherever we landed
        self.__graph.add_screen(state=post_state)
        await kg.add_screen(state=post_state)

        step_result = StepResult(
            step=step,
            error=result.error,
            pre_hash=pre_hash,
            success=result.success,
            duration=result.duration,
            post_hash=post_hash,
            screen_changed=pre_hash != post_hash,
        )

        # Pop from DFS path
        if self.__current_path:
            self.__current_path = self.__current_path[:-1]

        if post_hash not in self.__fully_scanned:
            # Landed on a screen with untried elements — scan it
            self.__phase = BFSPhase.SCAN
            logger.debug("BACKTRACK landed on unexplored screen %s", post_hash[:8])
        elif self.__current_path:
            # Still have depth — keep backtracking
            self.__phase = BFSPhase.BACKTRACK
            logger.debug("BACKTRACK: screen %s fully scanned, continuing up", post_hash[:8])
        else:
            # At root level, everything on the DFS path is scanned.
            # Check KG for orphaned unexplored screens.
            orphans = self.__find_orphaned_screens(kg)
            if orphans:
                for entry in orphans:
                    self.__bfs_queue.append(entry)
                self.__phase = BFSPhase.ADVANCE
                logger.info(
                    "DFS tree exhausted, %d orphaned screens found for recovery",
                    len(orphans),
                )
            else:
                return StrategyResult(
                    step_result=step_result,
                    status=StrategyStatus.COMPLETE,
                    message="DFS exploration complete — all reachable screens scanned",
                )

        return StrategyResult(
            step_result=step_result,
            status=StrategyStatus.CONTINUE,
            message="DFS: backtracking",
        )

    def __find_orphaned_screens(self, kg: KnowledgeGraph) -> List[BFSQueueEntry]:
        """Find screens in the KG that are not in ``fully_scanned`` and
        have a known path (recorded transitions) so we can navigate to them."""

        orphans: List[BFSQueueEntry] = []
        for visual_hash in kg.nodes:
            if visual_hash in self.__fully_scanned:
                continue
            if visual_hash == self.__root_hash:
                continue
            result = kg.get_inbound_edge(visual_hash)
            if result is None:
                continue
            source_hash, edge = result
            try:
                _resolved_type = ActionType(edge.action_type)
            except (ValueError, KeyError):
                _resolved_type = ActionType.TAP
            action = Action(
                confidence=1.0,
                target=edge.action_target or "orphan recovery",
                action_type=_resolved_type,
                rationale=f"DFS recovery: navigate to orphaned screen via {_resolved_type.value}",
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

    # ---- ADVANCE -----------------------------------------------------

    async def __execute_advance(self) -> StrategyResult:
        """
        ADVANCE phase (recovery): navigate to an orphaned unexplored screen.

        Uses ``__pending_nav`` (a pre-computed list of actions) to replay
        the route.  When ``__pending_nav`` is empty, dequeues the next
        recovery entry and computes the navigation sequence.
        """

        self.__require_bfs_deps()  # validate; this method uses no direct kg/vision refs

        # If we have pending navigation actions, execute the next one
        if self.__pending_nav:
            return await self.__execute_nav_action()

        # Dequeue next recovery entry
        if not self.__bfs_queue:
            return StrategyResult(
                status=StrategyStatus.COMPLETE,
                message="DFS exploration complete — all reachable screens scanned",
            )

        entry = self.__bfs_queue.popleft()

        # Skip if already fully scanned (might have been scanned via a different path)
        if entry.screen_hash in self.__fully_scanned:
            logger.debug("Skipping already-scanned screen %s", entry.screen_hash[:8])
            # Stay in ADVANCE to dequeue next
            return StrategyResult(
                status=StrategyStatus.CONTINUE,
                message=f"DFS recovery: skipping already-scanned {entry.screen_hash[:8]}",
            )

        # Compute navigation from current position to the target screen
        self.__pending_nav = self.__compute_navigation(target_entry=entry)
        self.__scanning_hash = entry.screen_hash
        self.__current_path = list(entry.path_from_root)

        logger.info(
            "ADVANCE to screen %s (depth=%d, nav_steps=%d)",
            entry.screen_hash[:8],
            entry.depth,
            len(self.__pending_nav),
        )

        if self.__pending_nav:
            return await self.__execute_nav_action()

        # Target is already the current screen (edge case)
        self.__phase = BFSPhase.SCAN
        return StrategyResult(
            status=StrategyStatus.CONTINUE,
            message=f"DFS recovery: already at target {entry.screen_hash[:8]}, scanning",
        )

    async def __execute_nav_action(self) -> StrategyResult:
        """
        Executes a single navigation action from ``__pending_nav``.
        """

        action = self.__pending_nav.pop(0)

        pre_capture = await self.__capture.capture()
        pre_state = self.__capture.compute_state(capture=pre_capture)
        pre_hash = self.__graph.resolve_hash(pre_state.visual_hash)

        step = Step(
            action=action,
            screen_hash=pre_hash,
            step_number=self.__steps,
        )

        result = await execute_device_action(device=self.__device, action=action)
        self.__steps += 1
        self.__last = action

        await asyncio.sleep(delay=0.5)

        # Package scope enforcement
        if not await self.__enforce_package_scope():
            return StrategyResult(
                status=StrategyStatus.COMPLETE,
                message=f"Left target package {self.__target_package} and could not recover",
            )

        post_capture = await self.__capture.capture()
        post_state = self.__capture.compute_state(capture=post_capture)
        post_hash = self.__graph.resolve_hash(post_state.visual_hash)

        # Register screen
        self.__graph.add_screen(state=post_state)
        if self.__knowledge_graph:
            await self.__knowledge_graph.add_screen(state=post_state)

        step_result = StepResult(
            step=step,
            error=result.error,
            pre_hash=pre_hash,
            success=result.success,
            duration=result.duration,
            post_hash=post_hash,
            screen_changed=pre_hash != post_hash,
        )

        # If we've arrived at the target OR exhausted pending nav, start scanning
        if not self.__pending_nav or post_hash == self.__scanning_hash:
            self.__pending_nav.clear()
            self.__phase = BFSPhase.SCAN
            logger.debug("Navigation complete, scanning %s", post_hash[:8])

        return StrategyResult(
            step_result=step_result,
            status=StrategyStatus.CONTINUE,
            message=f"DFS recovery navigate: {action.to_description()}",
        )

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def __compute_navigation(self, target_entry: BFSQueueEntry) -> List[Action]:
        """
        Computes the sequence of BACK presses + forward taps to reach the
        target screen from the current position using simple-BACK strategy.

        Algorithm:
        1. Find the longest common prefix between ``__current_path`` and
           ``target_entry.path_from_root``.  Entries match only when both
           the source screen hash **and** the action taken are identical
           (i.e. the same transition was followed).
        2. For each excess level in ``__current_path`` beyond the common
           ancestor, append a BACK action.
        3. For each level from the common ancestor to the target, append
           the forward action stored in the path.
        """

        target_path = target_entry.path_from_root
        current = self.__current_path

        # Find common prefix length — entries must share the same source
        # screen AND the same action (same transition) to be considered
        # part of the common prefix.
        common_len = 0
        for i in range(min(len(current), len(target_path))):
            c_screen, c_action = current[i]
            t_screen, t_action = target_path[i]
            if c_screen == t_screen and c_action == t_action:
                common_len = i + 1
            else:
                break

        actions: List[Action] = []

        # BACK actions to ascend from current depth to common ancestor
        backs_needed = len(current) - common_len
        for _ in range(backs_needed):
            actions.append(
                Action(
                    confidence=1.0,
                    target="back navigation",
                    action_type=ActionType.BACK,
                    rationale="DFS recovery: navigating to common ancestor",
                )
            )

        # Forward actions from common ancestor to target
        for i in range(common_len, len(target_path)):
            _, forward_action = target_path[i]
            actions.append(forward_action)

        return actions

    # ------------------------------------------------------------------
    # Legacy (non-BFS) step execution
    # ------------------------------------------------------------------

    async def __execute_legacy_step(self) -> StrategyResult:
        """
        Original random-action exploration step.
        Used when VisionTool or KnowledgeGraph is not provided.
        """

        capture = await self.__capture.capture()
        state = self.__capture.compute_state(capture=capture)

        fingerprint = self.__graph.resolve_hash(state.visual_hash)

        # Update in-memory graph (backward compat)
        node = self.__graph.add_screen(state=state)

        # Persist to knowledge graph
        kg_node: Optional[GraphNode] = None
        if self.__knowledge_graph:
            kg_node = await self.__knowledge_graph.add_screen(state=state)

        if self.__last and self.__current:
            current_hash = self.__graph.resolve_hash(self.__current.visual_hash)
            self.__graph.record_transition(
                destination=fingerprint,
                origin=current_hash,
                action=self.__last.to_description(),
            )
            # Persist transition to knowledge graph
            if self.__knowledge_graph:
                await self.__knowledge_graph.record_transition(
                    source_hash=current_hash,
                    action=self.__last,
                    destination_hash=fingerprint,
                )

        self.__current = state
        size = await self.__device.get_screen_size()

        # Use KnowledgeGraph node for action generation when available
        # (it has cross-run visit counts for smarter decisions)
        gen_node: Union[ScreenNode, GraphNode] = kg_node if kg_node else node
        action = self.__generator.generate(node=gen_node, width=size[0], height=size[1])

        step = Step(
            action=action,
            screen_hash=fingerprint,
            step_number=self.__steps,
        )

        # Execute with proper coordinate conversion
        result = await execute_device_action(device=self.__device, action=action)

        self.__steps += 1
        self.__last = action

        # Stability wait
        await asyncio.sleep(delay=0.5)

        # Package scope enforcement
        if not await self.__enforce_package_scope():
            return StrategyResult(
                status=StrategyStatus.COMPLETE,
                message=f"Left target package {self.__target_package} and could not recover",
            )

        post_capture = await self.__capture.capture()
        post_state = self.__capture.compute_state(capture=post_capture)

        resolved_post = self.__graph.resolve_hash(post_state.visual_hash)
        step_result = StepResult(
            step=step,
            error=result.error,
            pre_hash=fingerprint,
            success=result.success,
            duration=result.duration,
            post_hash=resolved_post,
            screen_changed=fingerprint != resolved_post,
        )

        return StrategyResult(
            step_result=step_result,
            status=StrategyStatus.CONTINUE,
            message=f"Explored: {action.to_description()}",
        )
