"""
LangGraph node functions for the BFS exploration graph.

Mirrors the architecture of :mod:`fathom.graph.nodes` (intent graph) but
replaces the VLM planner + goal-reasoning loop with BFS-driven screen
scanning.  All the same services (audit, history, tracing, metrics, UX)
are reused identically.

Topology::

    ground → bfs_route ─── SCAN ───→ scan → execute → record → ground
                       ├── RETURN ──→ navigate → record → ground
                       ├── ADVANCE ─→ navigate → record → ground
                       └── COMPLETE → END
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime
from logging import getLogger
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from fathom.agent.state import AgentState
from fathom.agent.strategies.exploration import BFSPhase, BFSQueueEntry
from fathom.constants import ActionType
from fathom.graph.exploration_state import ExplorationGraphState
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.infrastructure.memory.ledger import Ledger
from fathom.interfaces import ILedger, IMemoryProvider
from fathom.prompts.modes import PromptMode
from fathom.schemas.actions import Action
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.services.audit import AuditService
from fathom.services.history import HistoryService
from fathom.services.ux import UXService
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool
from fathom.tools.vision.processing.annotator import ImageAnnotator
from fathom.utils.coordinates import CoordinateConverter
from fathom.utils.execution import (
    ensure_target_package,
    execute_device_action,
    get_action_coordinates,
)

logger = getLogger(__name__)


# ── Context ─────────────────────────────────────────────────────────────


class ExplorationNodeContext:
    """
    Shared mutable context for all exploration graph nodes.

    Parallel to :class:`~fathom.graph.nodes.NodeContext` but carries
    BFS-specific state and the persistent :class:`KnowledgeGraph`.
    Reuses the same service classes for audit, history, tracing, etc.
    """

    def __init__(
        self,
        device: DeviceTool,
        capture: CaptureTool,
        vision: VisionTool,
        knowledge_graph: KnowledgeGraph,
        memory: IMemoryProvider,
        *,
        max_steps: int = 100,
        workflow_id: str = "default",
        cancel_event: Optional[asyncio.Event] = None,
        target_package: Optional[str] = None,
    ) -> None:
        self.device = device
        self.capture_tool = capture
        self.vision = vision
        self.knowledge_graph = knowledge_graph
        self.memory = memory
        self.max_steps = max_steps
        self.target_package = target_package
        self._cancel_event = cancel_event or asyncio.Event()

        # Reused services (adapted for exploration)
        self.ledger: ILedger = Ledger()
        self.agent_state = AgentState(
            intent="Explore this app to discover all screens and features",
            max_steps=max_steps,
            # Disable loop detection — BFS exploration deliberately revisits
            # screens and the LoopDetector would falsely terminate the run.
            loop_threshold=999_999,
        )
        self.metrics = ExecutionMetrics()
        self.ux_service = UXService()
        self.audit_service = AuditService()
        self.history = HistoryService(
            workflow_id=workflow_id,
            intent="exploration",
        )

        # ── BFS state (mutable, lives on context not graph dict) ──────
        self.bfs_queue: Deque[BFSQueueEntry] = deque()
        self.phase: BFSPhase = BFSPhase.SCAN
        self.scanning_hash: Optional[str] = None
        self.current_path: List[Tuple[str, Action]] = []
        self.pending_nav: List[Action] = []
        self.root_hash: Optional[str] = None
        self.fully_scanned: Set[str] = set()

    @property
    def is_cancelled(self) -> bool:
        """Fast non-blocking check for cancellation."""
        return self._cancel_event.is_set()


# ── Node factory ────────────────────────────────────────────────────────


def build_exploration_nodes(
    ctx: ExplorationNodeContext,
) -> Dict[str, Callable[..., Any]]:
    """
    Return a dict of ``{node_name: async_callable}`` for the exploration
    graph.  All nodes close over *ctx*.
    """

    # ── ground ──────────────────────────────────────────────────────

    async def ground_node(state: ExplorationGraphState) -> ExplorationGraphState:
        """Capture the screen and compute the screen state."""

        if ctx.is_cancelled:
            logger.info("exploration ground_node: cancelled")
            return {
                **state,
                "capture": None,
                "is_complete": True,
                "completion_reason": "Workflow cancelled by user",
            }

        start = time.time()
        screen = await ctx.capture_tool.capture_stable(timeout=2000)
        grounding_duration = time.time() - start
        ctx.metrics.record(operation="screenshot", duration=grounding_duration)

        if not screen:
            return {
                **state,
                "capture": None,
                "is_complete": False,
                "grounding_duration": grounding_duration,
                "completion_reason": "Capture failed",
            }

        screen_state = ctx.capture_tool.compute_state(capture=screen)
        screen = screen.model_copy(update={"state": screen_state})
        is_new = ctx.agent_state.update_screen(screen=screen_state)

        return {
            **state,
            "capture": screen,
            "screen_state": screen_state,
            "is_new_screen": is_new,
            "grounding_duration": grounding_duration,
            # Reset per-step fields
            "action": None,
            "analysis": None,
            "step_result": None,
            "content_exhausted": False,
        }

    # ── bfs_route ───────────────────────────────────────────────────

    async def bfs_route_node(state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Read the current BFS phase from the context and propagate it
        into the graph state for the conditional router to inspect.
        """

        screen_state: Optional[ScreenState] = state.get("screen_state")
        fingerprint = screen_state.visual_hash if screen_state else None

        # First step — establish root
        if ctx.root_hash is None and fingerprint:
            ctx.root_hash = fingerprint
            ctx.scanning_hash = fingerprint
            ctx.phase = BFSPhase.SCAN

        return {
            **state,
            "bfs_phase": ctx.phase.value,
        }

    # ── scan ────────────────────────────────────────────────────────

    async def scan_node(state: ExplorationGraphState) -> ExplorationGraphState:
        """
        SCAN phase: use VLM to identify the next untried interactive
        element on the current screen.
        """

        if ctx.is_cancelled:
            return {
                **state,
                "is_complete": True,
                "completion_reason": "Workflow cancelled by user",
            }

        capture: Optional[ScreenCapture] = state.get("capture")
        screen_state: Optional[ScreenState] = state.get("screen_state")

        if not capture or not screen_state:
            return {
                **state,
                "action": None,
                "content_exhausted": True,
                "analysis_duration": 0.0,
            }

        fingerprint = screen_state.visual_hash
        ctx.scanning_hash = fingerprint

        # Register screen in knowledge graph
        await ctx.knowledge_graph.add_screen(state=screen_state)

        # Build exploration context for VLM
        kg_context = ctx.knowledge_graph.build_exploration_context(
            current_hash=fingerprint,
            bfs_queue_size=len(ctx.bfs_queue),
        )

        # Ask VLM for next untried element
        start = time.time()
        analysis: AnalysisResult = await ctx.vision.analyze(
            intent="Explore this app to discover all screens and features",
            capture=capture,
            context=kg_context,
            mode=PromptMode.EXPLORATION,
        )
        analysis_duration = time.time() - start
        ctx.metrics.record(operation="analysis", duration=analysis_duration)

        # Persist VLM screen description
        if analysis.screen_description:
            await ctx.knowledge_graph.add_screen(
                state=screen_state, description=analysis.screen_description
            )

        # VLM signals all elements exhausted
        if analysis.content_exhausted:
            ctx.fully_scanned.add(fingerprint)
            ctx.phase = BFSPhase.ADVANCE
            logger.info(
                "Screen %s fully scanned, advancing BFS (queue=%d)",
                fingerprint[:8],
                len(ctx.bfs_queue),
            )
            return {
                **state,
                "action": None,
                "analysis": analysis,
                "kg_context": kg_context,
                "content_exhausted": True,
                "screen_description": analysis.screen_description,
                "analysis_duration": analysis_duration,
                "bfs_phase": ctx.phase.value,
            }

        action = analysis.action

        # HARDWARE BACK override (clear bounds for back actions)
        if action.action_type == ActionType.BACK:
            action = action.model_copy(update={"bounds": None})

        return {
            **state,
            "action": action,
            "analysis": analysis,
            "kg_context": kg_context,
            "content_exhausted": False,
            "screen_description": analysis.screen_description,
            "analysis_duration": analysis_duration,
        }

    # ── navigate ────────────────────────────────────────────────────

    async def navigate_node(state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Execute navigation actions for RETURN (BACK) or ADVANCE (forward
        replay) phases.  Consumes one action from ``ctx.pending_nav``
        or synthesises a BACK action for RETURN.
        """

        if ctx.is_cancelled:
            return {
                **state,
                "is_complete": True,
                "completion_reason": "Workflow cancelled by user",
            }

        # Determine action
        if ctx.phase == BFSPhase.RETURN:
            action = Action(
                confidence=1.0,
                target="back navigation",
                action_type=ActionType.BACK,
                rationale="BFS: returning to scanning screen after probe",
            )
        elif ctx.pending_nav:
            action = ctx.pending_nav.pop(0)
        else:
            # ADVANCE with empty pending_nav — try to dequeue
            if not ctx.bfs_queue:
                return {
                    **state,
                    "is_complete": True,
                    "completion_reason": "BFS exploration complete — all reachable screens scanned",
                }

            entry = ctx.bfs_queue.popleft()

            # Skip already-scanned screens
            while entry.screen_hash in ctx.fully_scanned and ctx.bfs_queue:
                logger.debug("Skipping already-scanned screen %s", entry.screen_hash[:8])
                entry = ctx.bfs_queue.popleft()

            if entry.screen_hash in ctx.fully_scanned:
                return {
                    **state,
                    "is_complete": True,
                    "completion_reason": "BFS exploration complete — all reachable screens scanned",
                }

            # Compute navigation
            ctx.pending_nav = _compute_navigation(
                current_path=ctx.current_path,
                target_path=entry.path_from_root,
            )
            ctx.scanning_hash = entry.screen_hash
            ctx.current_path = list(entry.path_from_root)

            logger.info(
                "ADVANCE to screen %s (depth=%d, nav_steps=%d)",
                entry.screen_hash[:8],
                entry.depth,
                len(ctx.pending_nav),
            )

            if ctx.pending_nav:
                action = ctx.pending_nav.pop(0)
            else:
                # Already at target
                ctx.phase = BFSPhase.SCAN
                return {
                    **state,
                    "bfs_phase": ctx.phase.value,
                }

        # Execute the navigation action
        capture: Optional[ScreenCapture] = state.get("capture")
        screen_state: Optional[ScreenState] = state.get("screen_state")
        pre_hash = screen_state.visual_hash if screen_state else "0"

        step = Step(
            action=action,
            screen_hash=pre_hash,
            step_number=ctx.agent_state.step_count,
        )

        step_start = time.time()

        # Tracing
        if capture:
            coordinates = await get_action_coordinates(ctx.device, action)
            await _trace_exploration(ctx, action, capture.image, coordinates)

        action_result = await execute_device_action(device=ctx.device, action=action)
        execution_duration = time.time() - step_start
        ctx.metrics.record(operation="action", duration=execution_duration)

        step_result = StepResult(
            step=step,
            error=action_result.error,
            pre_hash=pre_hash,
            success=action_result.success,
            duration=int(execution_duration * 1000),
            post_hash="0",  # Will be updated by record_node after re-capture
            screen_changed=True,
        )

        return {
            **state,
            "action": action,
            "step_result": step_result,
            "execution_duration": execution_duration,
        }

    # ── execute ─────────────────────────────────────────────────────

    async def execute_node(state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Execute the VLM-recommended action from the SCAN phase.
        Includes coordinate conversion, tracing, and metrics — identical
        to the intent graph's execute_node.
        """

        if ctx.is_cancelled:
            return {
                **state,
                "step_result": None,
                "execution_duration": 0.0,
            }

        action: Optional[Action] = state.get("action")
        capture: Optional[ScreenCapture] = state.get("capture")
        screen_state: Optional[ScreenState] = state.get("screen_state")

        if not action or not capture:
            return {**state, "step_result": None, "execution_duration": 0.0}

        pre_hash = screen_state.visual_hash if screen_state else "0"

        # UX rendering
        ctx.ux_service.render_fallback(
            reasoning=action.rationale,
            action=action.to_description(),
            step_number=ctx.agent_state.step_count + 1,
        )

        step = Step(
            action=action,
            screen_hash=pre_hash,
            step_number=ctx.agent_state.step_count,
        )

        step_start = time.time()

        # Tracing (same as intent graph)
        coordinates = await get_action_coordinates(ctx.device, action)
        await _trace_exploration(ctx, action, capture.image, coordinates)

        # Execute with proper coordinate conversion
        action_result = await execute_device_action(device=ctx.device, action=action)

        execution_duration = time.time() - step_start
        ctx.metrics.record(operation="action", duration=execution_duration)

        step_result = StepResult(
            step=step,
            error=action_result.error,
            pre_hash=pre_hash,
            success=action_result.success,
            duration=int(execution_duration * 1000),
            post_hash="0",  # Filled in by record_node after stability wait + recapture
            screen_changed=True,
        )

        return {
            **state,
            "step_result": step_result,
            "execution_duration": execution_duration,
        }

    # ── record ──────────────────────────────────────────────────────

    async def record_node(state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Record the step result: update knowledge graph, agent state,
        history, audit, and determine BFS phase transitions.
        """

        if ctx.is_cancelled:
            return {
                **state,
                "is_complete": True,
                "completion_reason": "Workflow cancelled by user",
            }

        step_result: Optional[StepResult] = state.get("step_result")
        action: Optional[Action] = state.get("action")
        screen_state: Optional[ScreenState] = state.get("screen_state")

        if not step_result:
            return state

        pre_hash = step_result.pre_hash

        # Stability wait
        await asyncio.sleep(0.5)

        # ── Package scope enforcement ───────────────────────────────
        if ctx.target_package:
            recovered = await ensure_target_package(
                device=ctx.device,
                target_package=ctx.target_package,
            )
            if not recovered:
                logger.error(
                    "Could not recover to target package %s — terminating",
                    ctx.target_package,
                )
                return {
                    **state,
                    "is_complete": True,
                    "completion_reason": (
                        f"Left target package {ctx.target_package} and could not recover"
                    ),
                }

        # Re-capture post-state
        post_capture = await ctx.capture_tool.capture()
        post_state = ctx.capture_tool.compute_state(capture=post_capture)
        post_hash = post_state.visual_hash

        # Update step_result with actual post_hash
        step_result = step_result.model_copy(
            update={
                "post_hash": post_hash,
                "screen_changed": pre_hash != post_hash,
            }
        )

        # Record in agent state
        ctx.agent_state.record_step(result=step_result)

        # Check if post_hash is new BEFORE adding it to the KG
        post_is_new = not ctx.knowledge_graph.has_screen(post_hash)

        # Persist to knowledge graph
        await ctx.knowledge_graph.add_screen(state=post_state)
        if action and pre_hash != "0":
            await ctx.knowledge_graph.record_transition(
                source_hash=pre_hash,
                action=action,
                destination_hash=post_hash,
            )

        # Persist experience to memory provider
        if action:
            asyncio.create_task(
                ctx.memory.store_experience(
                    action=action,
                    success=step_result.success,
                    visual_hash=pre_hash,
                )
            )

        # History export (same as intent graph)
        center: Optional[List[int]] = None
        if action and action.bounds:
            size = await ctx.device.get_screen_size()
            converter = CoordinateConverter(screen_width=size[0], screen_height=size[1])
            cx, cy = converter.center_to_pixels(bounds=action.bounds)
            center = [cx, cy]

        ctx.history.save_step(result=step_result, absolute_center=center)

        # Audit (same as intent graph)
        if screen_state:
            ctx.audit_service.record_context(
                knowledge={},
                success=step_result.success,
                visual_hash=screen_state.visual_hash,
                step_number=ctx.agent_state.step_count,
                context=ctx.agent_state.build_context(),
                action_description=action.to_description() if action else "unknown",
            )

        # Append to accumulated results
        results: List[StepResult] = list(state.get("step_results", []))
        results.append(step_result)

        # ── BFS phase transitions ─────────────────────────────────
        if ctx.phase == BFSPhase.SCAN:
            if pre_hash != post_hash:
                # Navigated to a different screen — enqueue if new
                if post_is_new and post_hash not in ctx.fully_scanned and action:
                    new_path = list(ctx.current_path) + [(pre_hash, action)]
                    ctx.bfs_queue.append(
                        BFSQueueEntry(
                            screen_hash=post_hash,
                            parent_hash=pre_hash,
                            action_from_parent=action,
                            depth=len(new_path),
                            path_from_root=new_path,
                        )
                    )
                    logger.info(
                        "Discovered new screen %s at depth %d (queue=%d)",
                        post_hash[:8],
                        len(new_path),
                        len(ctx.bfs_queue),
                    )

                # Must return to scanning screen
                ctx.phase = BFSPhase.RETURN
            # else: stayed on same screen — remain in SCAN

        elif ctx.phase == BFSPhase.RETURN:
            if post_hash == ctx.scanning_hash:
                # Successfully returned
                ctx.phase = BFSPhase.SCAN
                logger.debug("BACK returned to scanning screen %s", post_hash[:8])
            else:
                # BACK overshot — switch to ADVANCE for re-navigation
                ctx.phase = BFSPhase.ADVANCE
                logger.warning(
                    "BACK overshot: expected %s, got %s — switching to ADVANCE",
                    (ctx.scanning_hash or "?")[:8],
                    post_hash[:8],
                )

        elif ctx.phase == BFSPhase.ADVANCE:
            # Check if we arrived at the target
            if post_hash == ctx.scanning_hash:
                ctx.pending_nav.clear()
                ctx.phase = BFSPhase.SCAN
                logger.debug("Navigation complete, scanning %s", post_hash[:8])
            elif not ctx.pending_nav:
                # Ran out of nav actions — start scanning wherever we landed
                ctx.phase = BFSPhase.SCAN
                ctx.scanning_hash = post_hash
                logger.debug("Nav exhausted, scanning landed screen %s", post_hash[:8])

        # Check step limit
        is_complete = ctx.agent_state.step_count >= ctx.max_steps

        return {
            **state,
            "step_result": step_result,
            "step_results": results,
            "step_number": ctx.agent_state.step_count,
            "bfs_phase": ctx.phase.value,
            "is_complete": is_complete,
            "completion_reason": "Max steps reached" if is_complete else None,
        }

    return {
        "ground": ground_node,
        "bfs_route": bfs_route_node,
        "scan": scan_node,
        "navigate": navigate_node,
        "execute": execute_node,
        "record": record_node,
    }


# ── Routing functions ───────────────────────────────────────────────────


def make_route_after_ground(
    ctx: ExplorationNodeContext,
) -> Callable[[ExplorationGraphState], str]:
    """Router after ground: proceed to bfs_route or bail out."""

    def route(state: ExplorationGraphState) -> str:
        if ctx.is_cancelled:
            return "done"
        if state.get("capture") is None:
            return "done"
        return "bfs_route"

    return route


def make_route_after_bfs_route(
    ctx: ExplorationNodeContext,
) -> Callable[[ExplorationGraphState], str]:
    """Router after bfs_route: dispatch by BFS phase."""

    def route(state: ExplorationGraphState) -> str:
        if ctx.is_cancelled:
            return "done"

        phase = state.get("bfs_phase", "scan")

        if phase == "scan":
            return "scan"
        elif phase == "return":
            return "navigate"
        elif phase == "advance":
            # Check if BFS is complete
            if not ctx.bfs_queue and not ctx.pending_nav:
                ctx.audit_service.print_session_summary()
                return "done"
            return "navigate"

        return "done"

    return route


def make_route_after_scan(
    ctx: ExplorationNodeContext,
) -> Callable[[ExplorationGraphState], str]:
    """Router after scan: execute action or loop back if exhausted."""

    def route(state: ExplorationGraphState) -> str:
        if ctx.is_cancelled:
            return "done"

        if state.get("content_exhausted"):
            # Screen fully scanned — loop back to bfs_route (now in ADVANCE)
            return "bfs_route"

        if state.get("action") is None:
            return "bfs_route"

        return "execute"

    return route


def make_route_after_record(
    ctx: ExplorationNodeContext,
) -> Callable[[ExplorationGraphState], str]:
    """Router after record: continue or terminate."""

    def route(state: ExplorationGraphState) -> str:
        if ctx.is_cancelled:
            return "done"

        if state.get("is_complete"):
            ctx.audit_service.print_session_summary()
            return "done"

        if not ctx.agent_state.can_continue:
            ctx.audit_service.print_session_summary()
            return "done"

        return "ground"

    return route


# ── Private helpers ─────────────────────────────────────────────────────


def _compute_navigation(
    current_path: List[Tuple[str, Action]],
    target_path: List[Tuple[str, Action]],
) -> List[Action]:
    """
    Compute BACK + forward actions to navigate from current_path to
    target_path using the simple-BACK strategy.
    """

    common_len = 0
    for i in range(min(len(current_path), len(target_path))):
        c_screen, c_action = current_path[i]
        t_screen, t_action = target_path[i]
        if c_screen == t_screen and c_action == t_action:
            common_len = i + 1
        else:
            break

    actions: List[Action] = []

    # BACK actions to ascend
    backs_needed = len(current_path) - common_len
    for _ in range(backs_needed):
        actions.append(
            Action(
                confidence=1.0,
                target="back navigation",
                action_type=ActionType.BACK,
                rationale="BFS: navigating to common ancestor",
            )
        )

    # Forward actions from common ancestor to target
    for i in range(common_len, len(target_path)):
        _, forward_action = target_path[i]
        actions.append(forward_action)

    return actions


async def _trace_exploration(
    ctx: ExplorationNodeContext,
    action: Action,
    image_data: bytes,
    coordinates: Tuple[int, ...],
) -> None:
    """Write a visual trace image — same as the intent graph's _trace_background."""

    if not coordinates:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = (
        f"explore__{ctx.agent_state.step_count + 1}__{action.action_type.value}__{timestamp}.png"
    )
    path = f"assets/traces/{filename}"
    ImageAnnotator.trace(
        output_path=path,
        coords=coordinates,
        image_data=image_data,
        label=action.to_description(),
        action_type=action.action_type.value,
    )
