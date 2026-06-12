"""
LangGraph node functions for the DFS exploration graph.

Mirrors the architecture of :mod:`fathom.graph.nodes` (intent graph) but
replaces the VLM planner + goal-reasoning loop with DFS-driven screen
scanning.  All the same services (audit, history, tracing, metrics, UX)
are reused identically.

Topology::

    ground → dfs_route ─── SCAN ──────→ scan → execute → record → ground
                        ├── BACKTRACK ─→ navigate → record → ground
                        ├── ADVANCE ──→ navigate → record → ground  (recovery)
                        └── COMPLETE ──→ END
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
from fathom.constants.events import ExplorationEvent
from fathom.domain.exploration import (
    ActionKey,
    DedupPolicy,
    DepthFloorConfig,
    DepthFloorPolicy,
    ExplorationPolicyConfig,
)
from fathom.graph.exploration_state import ExplorationGraphState
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.infrastructure.memory.ledger import Ledger
from fathom.interfaces import ILedger, IMemoryProvider
from fathom.prompts.modes import PromptMode
from fathom.schemas.actions import Action
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.results import ActionResult, AnalysisResult
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
from fathom.utils.hitl import prompt_human_review

logger = getLogger(__name__)


# Minimum DFS path length (in edges traversed from root) before the agent
# is allowed to honour a VLM ``content_exhausted=True`` flag.  The canonical
# home for this value is :class:`DepthFloorConfig.minimum`; it is mirrored
# here as a module constant for backward-compatible imports.
MIN_DFS_DEPTH: int = DepthFloorConfig().minimum


# ── Context ─────────────────────────────────────────────────────────────


class ExplorationNodeContext:
    """
    Shared mutable context for all exploration graph nodes.

    Parallel to :class:`~fathom.graph.nodes.NodeContext` but carries
    DFS-specific state and the persistent :class:`KnowledgeGraph`.
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
        timeout: float = 3600.0,
        workflow_id: str = "default",
        cancel_event: Optional[asyncio.Event] = None,
        pause_event: Optional[asyncio.Event] = None,
        target_package: Optional[str] = None,
        focus: Optional[str] = None,
        event_bus: Optional[Callable[[ExplorationEvent, Dict[str, Any]], None]] = None,
        policy_config: Optional[ExplorationPolicyConfig] = None,
    ) -> None:
        self.device = device
        self.capture_tool = capture
        self.vision = vision
        self.knowledge_graph = knowledge_graph
        self.memory = memory
        self.max_steps = max_steps
        self.timeout = timeout
        self.start_time = time.time()
        self.target_package = target_package
        self._cancel_event = cancel_event or asyncio.Event()
        self._pause_event = pause_event or asyncio.Event()
        self._pause_event.set()
        self._event_bus: Callable[[ExplorationEvent, Dict[str, Any]], None] = (
            event_bus if event_bus is not None else (lambda _event, _ctx: None)
        )

        # Injected decision policies (pure domain rules — no I/O).
        config = policy_config or ExplorationPolicyConfig()
        self.policy_config = config
        self.depth_floor_policy = DepthFloorPolicy(config=config.depth)
        self.dedup_policy = DedupPolicy(dedup=config.dedup, sampling=config.sampling)

        # Focused exploration: when `focus` is set, bias the agent toward the
        # named section via the cached-prompt GOAL; otherwise fall back to the
        # full-breadth mapping intent.
        focus_clean = (focus or "").strip()
        # Keep the raw focus phrase so per-turn KG context can surface it as a
        # short reminder (the cached system GOAL handles the long-form intent).
        self.focus: Optional[str] = focus_clean or None
        if focus_clean:
            intent = (
                f"Focus on exploring {focus_clean} of this app. Map every screen "
                f"and element within that section; skip unrelated sections."
            )
        else:
            intent = "Explore this app to discover all screens and features"

        # Reused services (adapted for exploration)
        self.ledger: ILedger = Ledger()
        self.agent_state = AgentState(
            intent=intent,
            max_steps=max_steps,
            # Disable loop detection — DFS exploration deliberately revisits
            # screens and the LoopDetector would falsely terminate the run.
            loop_threshold=999_999,
        )
        self.metrics = ExecutionMetrics()
        self.ux_service = UXService()
        self.audit_service = AuditService()
        self.history = HistoryService(
            workflow_id=workflow_id,
            intent="exploration",
            package_name=target_package or "",
        )

        # ── DFS state (mutable, lives on context not graph dict) ──────
        self.bfs_queue: Deque[BFSQueueEntry] = deque()
        self.phase: BFSPhase = BFSPhase.SCAN
        self.scanning_hash: Optional[str] = None
        self.current_path: List[Tuple[str, Action]] = []
        self.pending_nav: List[Action] = []
        self.root_hash: Optional[str] = None
        self.fully_scanned: Set[str] = set()
        # Per-screen counter of how many times the VLM has declared
        # ``content_exhausted`` for it.  Used by the depth-floor guard to
        # tolerate one premature exhaustion before honouring it.
        self.exhaustion_retries: Dict[str, int] = {}

    @property
    def is_cancelled(self) -> bool:
        """Fast non-blocking check for cancellation."""
        return self._cancel_event.is_set()

    @property
    def pause_event(self) -> asyncio.Event:
        """Async-friendly pause event."""

        return self._pause_event

    def update_intent(self, intent: str) -> None:
        """Update the exploration intent used in planning."""

        self.agent_state.set_intent(intent)

    def emit(self, event: ExplorationEvent, /, **fields: Any) -> None:
        """Publish an event to the bus. Best-effort — never raises."""

        try:
            self._event_bus(event, fields)
        except Exception:  # noqa: BLE001 — bus failures must not break the graph
            logger.debug("event bus raised for %s", event, exc_info=True)


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

        ctx.emit(
            ExplorationEvent.SCREEN_CAPTURED,
            activity=screen_state.activity,
            screen_hash=screen_state.visual_hash,
            is_new=is_new,
        )

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
        Read the current DFS phase from the context and propagate it
        into the graph state for the conditional router to inspect.
        """

        screen_state: Optional[ScreenState] = state.get("screen_state")
        fingerprint = (
            ctx.knowledge_graph.resolve_hash(screen_state.visual_hash) if screen_state else None
        )

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

        fingerprint = ctx.knowledge_graph.resolve_hash(screen_state.visual_hash)
        ctx.scanning_hash = fingerprint

        # Build exploration context for VLM (DFS: depth + parent for flow awareness)
        parent_hash = ctx.current_path[-1][0] if ctx.current_path else None
        parent_node = ctx.knowledge_graph.nodes.get(parent_hash) if parent_hash else None
        parent_description = parent_node.description if parent_node else None

        # Assemble recent step feedback from agent state so the LLM can
        # course-correct based on the trajectory of recent actions.
        recent_steps: Optional[List[Dict[str, Any]]] = None
        agent_ctx = ctx.agent_state.build_context()
        raw_actions = agent_ctx.get("recent_actions")
        if isinstance(raw_actions, list) and raw_actions:
            recent_steps = raw_actions[-5:]

        # Depth-floor active when (a) we're below the minimum chain length
        # and (b) the VLM has already declared this screen exhausted at
        # least once.  The injected directive nudges the model to find any
        # untried element on the retry pass.
        depth_floor_active = ctx.depth_floor_policy.is_active(
            depth=len(ctx.current_path),
            retries=ctx.exhaustion_retries.get(fingerprint, 0),
        )
        kg_context = ctx.knowledge_graph.build_exploration_context(
            current_hash=fingerprint,
            depth=len(ctx.current_path),
            parent_description=parent_description,
            fully_scanned_count=len(ctx.fully_scanned),
            recent_steps=recent_steps,
            depth_floor_active=depth_floor_active,
            min_dfs_depth=ctx.depth_floor_policy.minimum,
            focus=ctx.focus,
        )

        # Ask VLM for next untried element
        start = time.time()
        dedup_failures: Optional[List[str]] = None
        analysis: AnalysisResult = await ctx.vision.analyze(
            intent=ctx.agent_state.intent,
            capture=capture,
            context=kg_context,
            mode=PromptMode.EXPLORATION,
            resolved_fingerprint=fingerprint,
        )

        # ── Dedup guard: reject actions already tried on this screen ──
        # Keyed on (action_type, coord_bucket) — stable across LLM label drift.
        # Falls back to (action_type, target_name) for legacy edges persisted
        # before coord_bucket was recorded.
        tried_keys = {
            ActionKey(kind=atype.lower(), label=(bucket or atarget).lower())
            for atype, atarget, bucket, _ in ctx.knowledge_graph.get_tried_actions(fingerprint)
        }

        # Repeatability, per-category sampling caps, and the retry budget are
        # owned by the injected DedupPolicy (see fathom.domain.exploration).
        for _retry in range(ctx.dedup_policy.retries):
            if not analysis.action or analysis.content_exhausted:
                break
            if ctx.dedup_policy.is_repeatable(analysis.action):
                break

            # Category-sampling guard — fires when freeform target_name would
            # otherwise let the LLM enumerate a list (every card has a
            # distinct bucket + label, so coord-bucket dedup doesn't catch it).
            proposed_category = analysis.action.element_category
            category_limit = ctx.dedup_policy.limit_for(proposed_category)
            if proposed_category is not None and category_limit is not None:
                already_sampled = ctx.knowledge_graph.count_category_taps(
                    fingerprint, proposed_category
                )
                if ctx.dedup_policy.is_over_sampled(
                    category=proposed_category, sampled=already_sampled
                ):
                    limit = category_limit
                    logger.warning(
                        "Sampling guard: %s already sampled %d×/%d on screen %s — rejecting",
                        proposed_category,
                        already_sampled,
                        limit,
                        fingerprint[:8],
                    )
                    dedup_failures = [
                        f"You have already sampled {already_sampled} {proposed_category} "
                        f"elements on this screen (limit {limit}). Per LIST SAMPLING, the "
                        f"rest are effectively tried — pick a DIFFERENT category "
                        f"(P1 global_navigation, P2 primary_action, P5 secondary_control) "
                        f"or press BACK."
                    ]
                    analysis = await ctx.vision.analyze(
                        intent=ctx.agent_state.intent,
                        capture=capture,
                        context=kg_context,
                        mode=PromptMode.EXPLORATION,
                        resolved_fingerprint=fingerprint,
                        failures=dedup_failures,
                    )
                    continue  # re-check the fresh analysis from the top

            action_key = ctx.dedup_policy.key_for(analysis.action)
            if ctx.dedup_policy.is_novel(action=analysis.action, tried=tried_keys):
                break  # Novel action — proceed

            logger.warning(
                "Dedup guard: LLM repeated tried action %s on screen %s (retry %d/%d)",
                action_key,
                fingerprint[:8],
                _retry + 1,
                ctx.dedup_policy.retries,
            )
            repeated_label = (
                analysis.action.natural_language_target
                or analysis.action.target
                or action_key.label
            )
            dedup_failures = [
                f'You already tried {action_key.kind} "{repeated_label}" — pick a DIFFERENT untried element.'
            ]
            analysis = await ctx.vision.analyze(
                intent=ctx.agent_state.intent,
                capture=capture,
                context=kg_context,
                mode=PromptMode.EXPLORATION,
                resolved_fingerprint=fingerprint,
                failures=dedup_failures,
            )
        else:
            # Check if the final retry still picked a tried action — but skip
            # the check for repeatable actions (scroll/swipe/back), which are
            # meant to be re-issued.
            if (
                analysis.action
                and not analysis.content_exhausted
                and not ctx.dedup_policy.is_repeatable(analysis.action)
                and not ctx.dedup_policy.is_novel(action=analysis.action, tried=tried_keys)
            ):
                logger.warning(
                    "Dedup guard exhausted retries on screen %s — forcing content_exhausted",
                    fingerprint[:8],
                )
                analysis.content_exhausted = True

        analysis_duration = time.time() - start
        ctx.metrics.record(operation="analysis", duration=analysis_duration)

        # Always record tokens from analysis, defaulting to 0 if not present.
        # The analysis.metrics dict may not contain token values if the provider
        # didn't populate them (e.g., missing usage_metadata), so we always attempt
        # to record with sensible defaults.
        prompt_tokens = int(analysis.metrics.get("prompt_tokens", 0))
        completion_tokens = int(analysis.metrics.get("completion_tokens", 0))
        cached_tokens = int(analysis.metrics.get("cached_tokens", 0))
        ctx.metrics.record_tokens(
            prompt=prompt_tokens,
            completion=completion_tokens,
            cached=cached_tokens,
        )

        ctx.emit(
            ExplorationEvent.LLM_CALL_COMPLETED,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            duration=analysis_duration,
        )

        # Register screen + persist VLM description in a single add_screen call
        await ctx.knowledge_graph.add_screen(
            state=screen_state, description=analysis.screen_description
        )

        # Rich screen translation: the LLM sees the existing description in
        # context and only outputs NEW observations.  Append if non-empty.
        rich_text = analysis.metadata.get("rich_description", "")
        if rich_text and rich_text.strip():
            await ctx.knowledge_graph.append_activity_description(
                activity=screen_state.activity,
                observation=rich_text,
            )
            logger.debug(
                "Appended observation for activity %s (screen %s)",
                screen_state.activity,
                fingerprint[:8],
            )

        # VLM signals all elements exhausted
        if analysis.content_exhausted:
            retries_so_far = ctx.exhaustion_retries.get(fingerprint, 0)
            depth = len(ctx.current_path)
            # Depth-floor veto: on a fresh shallow screen, give the VLM one
            # more chance to pick an element before we backtrack.  The retry
            # pass triggers the DEPTH FLOOR directive in the kg_context.
            if ctx.depth_floor_policy.should_veto(depth=depth, retries=retries_so_far):
                ctx.exhaustion_retries[fingerprint] = retries_so_far + 1
                logger.info(
                    "Depth-floor veto: screen %s exhausted at depth %d (< %d); "
                    "re-prompting with DEPTH FLOOR directive",
                    fingerprint[:8],
                    depth,
                    ctx.depth_floor_policy.minimum,
                )
                ctx.emit(
                    ExplorationEvent.PHASE_TRANSITION,
                    **{
                        "from": ctx.phase.value,
                        "to": ctx.phase.value,
                        "reason": "depth_floor_retry",
                    },
                )
                return {
                    **state,
                    "action": None,
                    "analysis": analysis,
                    "kg_context": kg_context,
                    "content_exhausted": False,
                    "screen_description": analysis.screen_description,
                    "analysis_duration": analysis_duration,
                    "bfs_phase": ctx.phase.value,
                }

            ctx.fully_scanned.add(fingerprint)
            prev_phase = ctx.phase.value
            ctx.phase = BFSPhase.BACKTRACK
            logger.info(
                "Screen %s fully scanned, backtracking (depth=%d)",
                fingerprint[:8],
                len(ctx.current_path),
            )
            ctx.emit(
                ExplorationEvent.PHASE_TRANSITION,
                **{"from": prev_phase, "to": ctx.phase.value, "reason": "content_exhausted"},
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
        # Successful action picked — reset exhaustion-retry counter for this
        # screen so a future stall isn't pre-empted by an old veto.
        ctx.exhaustion_retries.pop(fingerprint, None)

        action = analysis.action

        # HARDWARE BACK override (clear bounds for back actions)
        if action.action_type == ActionType.BACK:
            action = action.model_copy(update={"bounds": None})

        ctx.emit(
            ExplorationEvent.ACTION_PLANNED,
            action_type=action.action_type.value,
            target=action.natural_language_target or action.target,
            reasoning=action.rationale,
        )

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
        Execute navigation actions for BACKTRACK (BACK) or ADVANCE
        (recovery path replay) phases.  Synthesises a BACK action for
        BACKTRACK; consumes from ``ctx.pending_nav`` for ADVANCE.
        """

        if ctx.is_cancelled:
            return {
                **state,
                "is_complete": True,
                "completion_reason": "Workflow cancelled by user",
            }

        # Determine action
        if ctx.phase == BFSPhase.BACKTRACK:
            if not ctx.current_path:
                orphans = _find_orphaned_screens(ctx)
                if orphans:
                    for entry in orphans:
                        ctx.bfs_queue.append(entry)
                    ctx.phase = BFSPhase.ADVANCE
                    logger.info(
                        "BACKTRACK at root — skipping BACK, %d orphans queued for recovery",
                        len(orphans),
                    )
                    if ctx.bfs_queue:
                        entry = ctx.bfs_queue.popleft()
                        while entry.screen_hash in ctx.fully_scanned and ctx.bfs_queue:
                            entry = ctx.bfs_queue.popleft()
                        if entry.screen_hash not in ctx.fully_scanned:
                            ctx.pending_nav = _compute_navigation(
                                current_path=ctx.current_path,
                                target_path=entry.path_from_root,
                            )
                            ctx.scanning_hash = entry.screen_hash
                            ctx.current_path = list(entry.path_from_root)
                            if ctx.pending_nav:
                                action = ctx.pending_nav.pop(0)
                            else:
                                ctx.phase = BFSPhase.SCAN
                                return {**state, "bfs_phase": ctx.phase.value}
                        else:
                            return {
                                **state,
                                "is_complete": True,
                                "completion_reason": "DFS complete — all reachable screens scanned",
                            }
                    else:
                        return {
                            **state,
                            "is_complete": True,
                            "completion_reason": "DFS complete — all reachable screens scanned",
                        }
                else:
                    return {
                        **state,
                        "is_complete": True,
                        "completion_reason": "DFS complete — all reachable screens scanned",
                    }
            else:
                action = Action(
                    confidence=1.0,
                    target="back navigation",
                    action_type=ActionType.BACK,
                    rationale="DFS: backtracking from exhausted screen",
                )
                ctx.emit(ExplorationEvent.BACKTRACK, action="back", depth=len(ctx.current_path))
        elif ctx.pending_nav:
            action = ctx.pending_nav.pop(0)
        else:
            # ADVANCE with empty pending_nav — try to dequeue recovery targets
            if not ctx.bfs_queue:
                return {
                    **state,
                    "is_complete": True,
                    "completion_reason": "DFS exploration complete — all reachable screens scanned",
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
                    "completion_reason": "DFS exploration complete — all reachable screens scanned",
                }

            # Compute navigation to orphaned screen
            ctx.pending_nav = _compute_navigation(
                current_path=ctx.current_path,
                target_path=entry.path_from_root,
            )
            ctx.scanning_hash = entry.screen_hash
            ctx.current_path = list(entry.path_from_root)

            logger.info(
                "ADVANCE (recovery) to screen %s (depth=%d, nav_steps=%d)",
                entry.screen_hash[:8],
                entry.depth,
                len(ctx.pending_nav),
            )

            ctx.emit(
                ExplorationEvent.NAVIGATION_STARTED,
                target=entry.screen_hash[:8],
                steps=len(ctx.pending_nav),
                depth=entry.depth,
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

        is_paused = ctx.pause_event and not ctx.pause_event.is_set()
        if is_paused:
            reviewed_action, intent_changed, exit_session = prompt_human_review(
                action=action,
                step_number=ctx.agent_state.step_count + 1,
                screen_description=None,
                current_intent=ctx.agent_state.intent,
                update_intent=ctx.update_intent,
            )
            ctx.pause_event.set()

            if exit_session:
                ctx.agent_state.mark_complete("Session ended by user")
                return {
                    **state,
                    "is_complete": True,
                    "completion_reason": "Session ended by user",
                    "step_result": None,
                }

            if intent_changed:
                return {
                    **state,
                    "action": None,
                    "analysis": None,
                    "step_result": None,
                    "completion_reason": "Intent updated by user",
                }

            if reviewed_action is not None and reviewed_action != action:
                action = reviewed_action

        if ctx.is_cancelled:
            logger.info("navigate_node: cancelled after pause")
            return {
                **state,
                "step_result": None,
                "execution_duration": 0.0,
            }

        step = Step(
            action=action,
            screen_hash=pre_hash,
            step_number=ctx.agent_state.step_count,
        )

        step_start = time.time()

        # Tracing (fire-and-forget — non-blocking)
        if capture:
            coordinates = await get_action_coordinates(ctx.device, action)
            asyncio.create_task(_trace_exploration(ctx, action, capture.image, coordinates))

        action_result = await execute_device_action(device=ctx.device, action=action)
        execution_duration = time.time() - step_start
        ctx.metrics.record(operation="action", duration=execution_duration)

        ctx.emit(
            ExplorationEvent.ACTION_EXECUTED,
            success=action_result.success,
            action=action.action_type.value,
            error=action_result.error,
        )

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
            "analysis": None,  # Clear stale scan analysis so record_node doesn't display it
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

        is_paused = ctx.pause_event and not ctx.pause_event.is_set()
        if is_paused:
            analysis: Optional[AnalysisResult] = state.get("analysis")
            screen_description = analysis.screen_description if analysis else None
            reviewed_action, intent_changed, exit_session = prompt_human_review(
                action=action,
                step_number=ctx.agent_state.step_count + 1,
                screen_description=screen_description,
                current_intent=ctx.agent_state.intent,
                update_intent=ctx.update_intent,
            )
            ctx.pause_event.set()

            if exit_session:
                ctx.agent_state.mark_complete("Session ended by user")
                return {
                    **state,
                    "is_complete": True,
                    "completion_reason": "Session ended by user",
                    "step_result": None,
                }

            if intent_changed:
                return {
                    **state,
                    "action": None,
                    "analysis": None,
                    "step_result": None,
                    "completion_reason": "Intent updated by user",
                }

            if reviewed_action is not None and reviewed_action != action:
                action = reviewed_action

        if ctx.is_cancelled:
            logger.info("exploration execute_node: cancelled after pause")
            return {
                **state,
                "step_result": None,
                "execution_duration": 0.0,
            }

        pre_hash = screen_state.visual_hash if screen_state else "0"

        # Handle memory-only actions (no device interaction needed)
        if action.action_type == ActionType.SAVE_MEMORY:
            if action.memory_updates:
                for key, value in action.memory_updates.items():
                    await ctx.ledger.set(key=key, value=value)
            return {
                **state,
                "step_result": StepResult(
                    step=Step(
                        action=action, screen_hash=pre_hash, step_number=ctx.agent_state.step_count
                    ),
                    error=None,
                    pre_hash=pre_hash,
                    success=True,
                    duration=0,
                    post_hash=pre_hash,
                    screen_changed=False,
                ),
                "execution_duration": 0.0,
            }

        if action.action_type == ActionType.RETRIEVE_MEMORY:
            return {
                **state,
                "step_result": StepResult(
                    step=Step(
                        action=action, screen_hash=pre_hash, step_number=ctx.agent_state.step_count
                    ),
                    error=None,
                    pre_hash=pre_hash,
                    success=True,
                    duration=0,
                    post_hash=pre_hash,
                    screen_changed=False,
                ),
                "execution_duration": 0.0,
            }

        # UX rendering (after HITL so edits are reflected)
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

        # Tracing (fire-and-forget — non-blocking)
        try:
            coordinates = await get_action_coordinates(ctx.device, action)
            asyncio.create_task(_trace_exploration(ctx, action, capture.image, coordinates))

            action_result = await execute_device_action(device=ctx.device, action=action)
        except Exception as exc:
            logger.exception("Device action failed with exception")
            action_result = ActionResult(success=False, duration=0, error=str(exc))

        if action.memory_updates:
            for key, value in action.memory_updates.items():
                await ctx.ledger.set(key=key, value=value)

        execution_duration = time.time() - step_start
        ctx.metrics.record(operation="action", duration=execution_duration)

        ctx.emit(
            ExplorationEvent.ACTION_EXECUTED,
            success=action_result.success,
            action=action.action_type.value,
            error=action_result.error,
        )

        step_result = StepResult(
            step=step,
            error=action_result.error,
            pre_hash=pre_hash,
            success=action_result.success,
            duration=int(execution_duration * 1000),
            post_hash="0",  # Filled in by record_node after recapture
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
        history, audit, and determine DFS phase transitions.
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

        # ── Package scope enforcement ───────────────────────────────
        if ctx.target_package:
            pre_recovery_pkg = await ctx.device.get_current_package()
            needs_recovery = pre_recovery_pkg != ctx.target_package

            if needs_recovery:
                recovered = await ensure_target_package(
                    device=ctx.device,
                    target_package=ctx.target_package,
                )
                if not recovered:
                    logger.error(
                        "Could not recover to target package %s — terminating",
                        ctx.target_package,
                    )
                    ctx.agent_state.record_step(result=step_result)
                    if action:
                        ctx.history.save_step(
                            result=step_result,
                            absolute_center=None,
                            activity=screen_state.activity if screen_state else None,
                        )
                    return {
                        **state,
                        "is_complete": True,
                        "completion_reason": (
                            f"Left target package {ctx.target_package} and could not recover"
                        ),
                    }

                # Recovery succeeded but we don't know our position in the app.
                # Reset DFS navigation state so the next cycle re-orients.
                ctx.pending_nav.clear()
                ctx.current_path = []
                ctx.phase = BFSPhase.SCAN
                logger.info("DFS state reset after package recovery")

        # Re-capture post-state
        post_capture = await ctx.capture_tool.capture()
        post_state = ctx.capture_tool.compute_state(capture=post_capture)
        post_hash = ctx.knowledge_graph.resolve_hash(post_state.visual_hash)

        # Canonicalise pre_hash for consistent comparisons
        pre_hash = ctx.knowledge_graph.resolve_hash(pre_hash)

        # Update step_result with actual post_hash
        step_result = step_result.model_copy(
            update={
                "post_hash": post_hash,
                "screen_changed": pre_hash != post_hash,
            }
        )

        # Record in agent state (sync, in-memory — fast)
        ctx.agent_state.record_step(result=step_result)

        # Check if post_hash is new BEFORE adding it to the KG
        post_is_new = not ctx.knowledge_graph.has_screen(post_hash)

        # Start coordinate fetch early (overlaps with add_screen I/O)
        size_task: Optional[asyncio.Task[Any]] = None
        if action and action.bounds:
            size_task = asyncio.create_task(ctx.device.get_screen_size())

        # KG screen must complete first (transition + DFS logic depend on it)
        await ctx.knowledge_graph.add_screen(state=post_state)

        if step_result.screen_changed:
            kg_stats = ctx.knowledge_graph.get_stats()
            unique = int(kg_stats.get("unique_screens", 0))
            unexplored = int(kg_stats.get("unexplored", 0))
            coverage = ((unique - unexplored) / unique * 100) if unique > 0 else 0.0
            event_payload = {
                "activity": post_state.activity,
                "screen_hash": post_hash[:8],
                "unique_screens": unique,
                "coverage": coverage,
            }
            if post_is_new:
                ctx.emit(ExplorationEvent.SCREEN_DISCOVERED, **event_payload)
            else:
                ctx.emit(ExplorationEvent.SCREEN_REVISITED, **event_payload)

        # Run independent writes in parallel
        parallel_writes: List[Any] = []
        if action and pre_hash != "0":
            parallel_writes.append(
                ctx.knowledge_graph.record_transition(
                    source_hash=pre_hash,
                    action=action,
                    destination_hash=post_hash,
                )
            )
        if action:
            parallel_writes.append(
                ctx.memory.store_experience(
                    action=action,
                    success=step_result.success,
                    visual_hash=pre_hash,
                )
            )

        # Resolve coordinates then offload sync history I/O to thread
        center: Optional[List[int]] = None
        if size_task and action and action.bounds:
            size = await size_task
            converter = CoordinateConverter(screen_width=size[0], screen_height=size[1])
            cx, cy = converter.center_to_pixels(bounds=action.bounds)
            center = [cx, cy]

        activity = screen_state.activity if screen_state else None
        parallel_writes.append(
            asyncio.to_thread(
                ctx.history.save_step,
                result=step_result,
                absolute_center=center,
                activity=activity,
            )
        )

        await asyncio.gather(*parallel_writes, return_exceptions=True)

        # Audit (sync, in-memory append — negligible)
        if screen_state:
            ctx.audit_service.record_context(
                knowledge={},
                success=step_result.success,
                visual_hash=screen_state.visual_hash,
                step_number=ctx.agent_state.step_count,
                context=ctx.agent_state.build_context(),
                action_description=action.to_description() if action else "unknown",
            )

            analysis: Optional[AnalysisResult] = state.get("analysis")
            grounding_dur = state.get("grounding_duration", 0.0)
            analysis_dur = state.get("analysis_duration", 0.0)
            execution_dur = state.get("execution_duration", 0.0)
            total_dur = grounding_dur + analysis_dur + execution_dur

            ctx.audit_service.log_exploration_step(
                is_stuck=ctx.agent_state.is_stuck,
                step_count=ctx.agent_state.step_count,
                state=screen_state,
                is_new_screen=state.get("is_new_screen", False),
                result=ActionResult(success=step_result.success, duration=step_result.duration),
                total_duration=total_dur,
                analysis_duration=analysis_dur,
                execution_duration=execution_dur,
                grounding_duration=grounding_dur,
                analysis=analysis,
                executed_action=action,
                phase=ctx.phase.value,
                depth=len(ctx.current_path),
            )

        # Append to accumulated results
        results: List[StepResult] = list(state.get("step_results", []))
        results.append(step_result)

        # ── DFS phase transitions ─────────────────────────────────
        is_complete = False
        prev_phase = ctx.phase.value

        if ctx.phase == BFSPhase.SCAN:
            if pre_hash != post_hash:
                is_back_action = action and action.action_type == ActionType.BACK
                if is_back_action:
                    if ctx.current_path:
                        ctx.current_path = ctx.current_path[:-1]
                    new_path = list(ctx.current_path)
                else:
                    new_path = (
                        list(ctx.current_path) + [(pre_hash, action)]
                        if action
                        else list(ctx.current_path)
                    )
                    ctx.current_path = new_path

                if post_hash in ctx.fully_scanned:
                    # Already exhausted — backtrack immediately
                    ctx.phase = BFSPhase.BACKTRACK
                    logger.debug(
                        "Navigated to already-scanned screen %s, backtracking",
                        post_hash[:8],
                    )
                else:
                    # DFS: stay in SCAN on the new screen (go deeper)
                    ctx.phase = BFSPhase.SCAN
                    if post_is_new:
                        logger.info(
                            "DFS: discovered new screen %s at depth %d",
                            post_hash[:8],
                            len(new_path),
                        )
            # else: stayed on same screen — remain in SCAN

        elif ctx.phase == BFSPhase.BACKTRACK:
            # We pressed BACK.  Pop from the DFS path.
            if ctx.current_path:
                ctx.current_path = ctx.current_path[:-1]

            if post_hash not in ctx.fully_scanned:
                # Landed on a screen with untried elements — scan it
                ctx.phase = BFSPhase.SCAN
                ctx.scanning_hash = post_hash
                logger.debug("BACKTRACK landed on unexplored screen %s", post_hash[:8])
            elif ctx.current_path:
                # Still have depth — keep backtracking
                ctx.phase = BFSPhase.BACKTRACK
                logger.debug(
                    "BACKTRACK: screen %s fully scanned, continuing up",
                    post_hash[:8],
                )
            else:
                # At root level, everything on the DFS path is scanned.
                # Check KG for orphaned unexplored screens.
                orphans = _find_orphaned_screens(ctx)
                if orphans:
                    for entry in orphans:
                        ctx.bfs_queue.append(entry)
                    ctx.phase = BFSPhase.ADVANCE
                    logger.info(
                        "DFS tree exhausted, %d orphaned screens found for recovery",
                        len(orphans),
                    )
                else:
                    is_complete = True

        elif ctx.phase == BFSPhase.ADVANCE:
            # Recovery navigation — check if we arrived at the target
            if post_hash == ctx.scanning_hash:
                ctx.pending_nav.clear()
                ctx.phase = BFSPhase.SCAN
                logger.debug("Recovery navigation complete, scanning %s", post_hash[:8])
            elif not ctx.pending_nav:
                # Ran out of nav actions — start scanning wherever we landed
                ctx.phase = BFSPhase.SCAN
                ctx.scanning_hash = post_hash
                ctx.current_path = _path_to_screen(ctx, post_hash)
                logger.debug("Nav exhausted, scanning landed screen %s", post_hash[:8])

        # Check step limit (merge with DFS completion signal)
        is_complete = is_complete or ctx.agent_state.step_count >= ctx.max_steps

        if is_complete and ctx.agent_state.step_count >= ctx.max_steps:
            completion_reason = "Max steps reached"
        elif is_complete:
            completion_reason = "DFS complete — all screens scanned"
        else:
            completion_reason = None

        if prev_phase != ctx.phase.value:
            ctx.emit(
                ExplorationEvent.PHASE_TRANSITION,
                **{"from": prev_phase, "to": ctx.phase.value},
            )

        ctx.emit(
            ExplorationEvent.STEP_COMPLETED,
            step=ctx.agent_state.step_count,
            max_steps=ctx.max_steps,
            phase=ctx.phase.value,
        )

        return {
            **state,
            "step_result": step_result,
            "step_results": results,
            "step_number": ctx.agent_state.step_count,
            "bfs_phase": ctx.phase.value,
            "is_complete": is_complete,
            "completion_reason": completion_reason,
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
    """Router after dfs_route: dispatch by exploration phase."""

    def route(state: ExplorationGraphState) -> str:
        if ctx.is_cancelled:
            return "done"

        phase = state.get("bfs_phase", "scan")

        if phase == "scan":
            return "scan"
        elif phase == "backtrack":
            return "navigate"
        elif phase == "advance":
            # Check if recovery queue is empty
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
            # Screen fully scanned — loop back to dfs_route (now in BACKTRACK)
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

        elapsed = time.time() - ctx.start_time
        if elapsed >= ctx.timeout:
            logger.info("Exploration timeout reached (%.0fs >= %.0fs)", elapsed, ctx.timeout)
            ctx.audit_service.print_session_summary()
            return "done"

        return "ground"

    return route


# ── Private helpers ─────────────────────────────────────────────────────


def _path_to_screen(
    ctx: ExplorationNodeContext,
    screen_hash: Optional[str],
) -> List[Tuple[str, Action]]:
    """Best-effort path reconstruction for a screen we've already visited."""
    if not screen_hash or screen_hash == ctx.root_hash:
        return []
    # Check the recovery queue for any entry targeting this screen
    for entry in ctx.bfs_queue:
        if entry.screen_hash == screen_hash:
            return list(entry.path_from_root)
    # Fallback: current_path minus last hop
    return list(ctx.current_path[:-1])


def _find_orphaned_screens(
    ctx: ExplorationNodeContext,
) -> List[BFSQueueEntry]:
    """Find screens in the KG that are not in ``fully_scanned`` and have a
    known inbound transition so we can attempt to navigate to them.

    Used as a DFS recovery mechanism when BACKTRACK reaches root but the
    knowledge graph contains screens that were discovered (via transitions)
    but never fully scanned -- typically caused by BACK overshooting or
    navigating to an already-scanned screen that had unexplored neighbours.
    """

    kg = ctx.knowledge_graph
    orphans: List[BFSQueueEntry] = []

    for visual_hash in kg.nodes:
        if visual_hash in ctx.fully_scanned:
            continue
        if visual_hash == ctx.root_hash:
            continue

        result = kg.get_inbound_edge(visual_hash)
        if result is None:
            continue
        source_hash, edge = result

        try:
            resolved_action_type = ActionType(edge.action_type)
        except (ValueError, KeyError):
            resolved_action_type = ActionType.TAP

        action = Action(
            confidence=1.0,
            target=edge.action_target or "orphan recovery",
            action_type=resolved_action_type,
            rationale=f"DFS recovery: navigate to orphaned screen via {resolved_action_type.value}",
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
                rationale="DFS recovery: navigating to common ancestor",
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
    """Write a visual trace image in a background thread (fire-and-forget).

    The synchronous PIL work (decode, annotate, save) is offloaded to the
    default ``ThreadPoolExecutor`` so it never blocks the event loop.
    """

    if not coordinates:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = (
        f"explore__{ctx.agent_state.step_count + 1}__{action.action_type.value}__{timestamp}.png"
    )
    path = f"assets/traces/{filename}"
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: ImageAnnotator.trace(
                output_path=path,
                coords=coordinates,
                image_data=image_data,
                label=action.to_description(),
                action_type=action.action_type.value,
            ),
        )
    except Exception as exc:
        logger.debug("Trace write failed (non-critical): %s", exc)
