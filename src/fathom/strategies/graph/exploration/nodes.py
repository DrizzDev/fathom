from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, cast

from fathom.constants import ActionType
from fathom.constants.defect import DEFECT_DETECTED_EVENT
from fathom.constants.exploration import (
    EXPLORATION_PROGRESS_EVENT,
    MAX_ROUTES_WITHOUT_PROGRESS,
    MAX_SENSITIVE_ACTION_RETRIES,
    MAX_STEPS_WITHOUT_NEW_SCREEN,
    RECENT_ACTION_WINDOW,
    BFSPhase,
)
from fathom.constants.graph import NodeName
from fathom.constants.state import CommonStateKey as CKey
from fathom.constants.state import CompletionReason
from fathom.constants.state import ExplorationStateKey as EKey
from fathom.core.defect.inline import InlineDefectDetector
from fathom.core.defect.vision import VisionDefectDetector
from fathom.core.exceptions import DeviceError
from fathom.core.exploration.config import ExplorationPolicyConfig
from fathom.core.exploration.dedup import ActionKey, DedupPolicy
from fathom.core.exploration.depth import DepthFloorPolicy
from fathom.core.safety.guard import TraversalGuard
from fathom.core.services.exploration import ExplorationVisionService
from fathom.interfaces.defect import (
    DefectRepositoryPort,
    InlineDefectDetectorPort,
    ScreenDefectDetectorPort,
)
from fathom.schemas.actions import Action
from fathom.schemas.defect import Defect, ScreenSnapshot, StepSignals
from fathom.schemas.exploration import ActionOutcome
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.exploration.dfs import DfsNavigator, DfsState
from fathom.strategies.graph.exploration.state import (
    ExplorationGraphState,
    get_action,
    get_bfs_phase,
    get_capture,
    get_screen_state,
    get_step_result,
    get_step_results,
    is_complete,
    is_content_exhausted,
)
from fathom.utils.wait import stability_wait

logger = logging.getLogger(__name__)

_DFS_COMPLETE = "DFS complete - all reachable screens scanned"

# Hardware BACK presses attempted to climb back into the target package when the
# device drifts out of it (e.g. a share sheet or permission prompt) before the
# run is abandoned as out-of-scope.
PACKAGE_RECOVERY_BACK_LIMIT = 3


class ExplorationNodeProvider:
    """
    Provides the LangGraph nodes and routers for DFS application exploration.

    Owns the mutable DFS bookkeeping (:class:`DfsState`) for the lifetime of a
    run, the navigator that plans recovery paths over the knowledge graph, and
    the decision policies (depth-floor, dedup, sampling) that guard the scan.
    The bound node methods drive a depth-first walk; the router methods steer
    the conditional edges between them.
    """

    def __init__(
        self,
        context: GraphContext,
        *,
        vision: Optional[ExplorationVisionService] = None,
        policy: Optional[ExplorationPolicyConfig] = None,
        dfs: Optional[DfsState] = None,
        inline_detector: Optional[InlineDefectDetectorPort] = None,
        defects: Optional[DefectRepositoryPort] = None,
        screen_detector: Optional[ScreenDefectDetectorPort] = None,
        traversal_guard: Optional[TraversalGuard] = None,
    ) -> None:
        """
        Initialize with shared context, the vision service, and DFS policies.
        """

        self.__context = context
        self.__vision = vision or ExplorationVisionService(
            llm=context.llm,
            use_cache=context.configuration.llm.use_cache,
            guarded=not context.focus,
        )
        self.__policy = policy or ExplorationPolicyConfig()
        self.__dfs = dfs or DfsState()
        self.__navigator = DfsNavigator(dfs=self.__dfs, knowledge_graph=context.exploration_graph)
        self.__depth_floor = DepthFloorPolicy(config=self.__policy.depth)
        self.__dedup = DedupPolicy(dedup=self.__policy.dedup, sampling=self.__policy.sampling)
        self.__inline_detector = inline_detector or InlineDefectDetector()
        self.__defects = defects
        self.__screen_detector = screen_detector
        # None means guardrails are off; the factory injects a guard for real runs.
        self.__traversal_guard = traversal_guard
        self.__inspected_screens: Set[str] = set()

    # ── Nodes ──────────────────────────────────────────────────────────────

    async def ground(self, state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Capture the screen, compute its MLSIA state, and reset per-step fields.
        """

        ctx = self.__context

        if ctx.is_cancelled:
            return self.__complete(state, reason=CompletionReason.CANCELLED)

        interrupt_reason = await self.__check_interrupts()
        if interrupt_reason is not None:
            return self.__complete(state, reason=interrupt_reason)

        start_time = time.time()

        try:
            screen = await ctx.perception.perceive(
                session_id=ctx.workflow_id, step_number=ctx.agent_state.step_count
            )
            screen_state = ctx.perception.build_state(capture=screen)

            # Fail fast (and loud) when perception yields nothing usable rather
            # than letting a blank screen wedge the routing loop. The structured
            # log names exactly what came back so the failing capture call is
            # diagnosable from a single run.
            if not self.__is_usable_capture(capture=screen, screen_state=screen_state):
                logger.error(
                    "Exploration grounding produced no usable screen",
                    extra={
                        "image_bytes": len(screen.image) if screen and screen.image else 0,
                        "has_hierarchy": bool(screen and screen.xml_content),
                        "activity": screen.activity if screen else None,
                    },
                )
                return self.__complete(state, reason=CompletionReason.PERCEPTION_FAILED)

            screen = screen.model_copy(update={"state": screen_state})
            is_new = ctx.agent_state.update_screen(screen=screen_state)

            result = self.__mutable(state)
            result[CKey.CAPTURE] = screen
            result[CKey.SCREEN_STATE] = screen_state
            result[CKey.IS_NEW_SCREEN] = is_new
            result[CKey.GROUNDING_DURATION] = time.time() - start_time
            # Reset per-step fields so a stale action never leaks across steps.
            result[EKey.ACTION] = None
            result[CKey.ANALYSIS] = None
            result[CKey.STEP_RESULT] = None
            result[EKey.CONTENT_EXHAUSTED] = False
            return cast("ExplorationGraphState", result)

        except Exception:
            logger.exception("Exploration grounding failed")
            return self.__complete(state, reason=CompletionReason.PERCEPTION_FAILED)

    @staticmethod
    def __is_usable_capture(*, capture: Any, screen_state: Any) -> bool:
        """
        Whether perception returned a screen the scan can actually act on.

        Guards against a blank or partial capture (empty screenshot, missing
        computed state) silently wedging the DFS routing loop.
        """

        return bool(capture and capture.image and screen_state)

    async def bfs_route(self, state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Establish the root on the first step and publish the current DFS phase.
        """

        ctx = self.__context
        dfs = self.__dfs

        # No-progress watchdog: bfs_route is the routing hub every cycle passes
        # through, while record() resets the counter on each completed step. If
        # routing cycles this many times without a step, the phase machine is
        # wedged (e.g. a screen with no usable capture); end cleanly rather than
        # spin to the graph recursion limit.
        dfs.stalled_routes += 1
        if dfs.stalled_routes > MAX_ROUTES_WITHOUT_PROGRESS:
            return self.__complete(state, reason=CompletionReason.STUCK)

        screen_state = get_screen_state(state)
        fingerprint = (
            ctx.exploration_graph.resolve_hash(screen_state.visual_hash) if screen_state else None
        )

        if dfs.root_hash is None and fingerprint:
            # First step of the run: restore the exhaustion frontier from the
            # persisted graph so a relaunch resumes instead of re-treading
            # screens it already fully explored in an earlier run.
            dfs.fully_scanned |= ctx.exploration_graph.exhausted_hashes()
            dfs.root_hash = fingerprint
            dfs.scanning_hash = fingerprint
            # An entry screen already exhausted last run is not worth re-scanning;
            # backtrack straight into frontier recovery instead of SCAN.
            dfs.phase = BFSPhase.BACKTRACK if fingerprint in dfs.fully_scanned else BFSPhase.SCAN

        result = self.__mutable(state)
        result[EKey.BFS_PHASE] = dfs.phase.value
        return cast("ExplorationGraphState", result)

    async def scan(self, state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Pick the next untried interactive element via the exploration vision.

        Guards the proposal against already-tried actions and over-sampled
        categories, and applies the depth-floor veto before honouring a
        content-exhaustion signal.
        """

        ctx = self.__context
        dfs = self.__dfs

        if ctx.is_cancelled:
            return self.__complete(state, reason=CompletionReason.CANCELLED)

        capture = get_capture(state)
        screen_state = get_screen_state(state)
        if not capture or not screen_state:
            result = self.__mutable(state)
            result[EKey.ACTION] = None
            result[EKey.CONTENT_EXHAUSTED] = True
            result[CKey.ANALYSIS_DURATION] = 0.0
            return cast("ExplorationGraphState", result)

        fingerprint = ctx.exploration_graph.resolve_hash(screen_state.visual_hash)
        dfs.scanning_hash = fingerprint

        depth = dfs.depth
        retries = dfs.exhaustion_retries.get(fingerprint, 0)
        knowledge_context = ctx.exploration_graph.build_exploration_context(
            current_hash=fingerprint,
            depth=depth,
            parent_description=self.__parent_description(),
            fully_scanned_count=len(dfs.fully_scanned),
            fully_scanned=dfs.fully_scanned,
            recent_actions=self.__recent_actions(state=state),
            depth_floor_active=self.__depth_floor.is_active(depth=depth, retries=retries),
            min_dfs_depth=self.__depth_floor.minimum,
            focus=ctx.focus,
        )

        start = time.time()
        analysis = await self.__vision.scan(
            capture=capture, knowledge_context=knowledge_context, intent=ctx.intent
        )
        analysis = await self.__guard_against_repeats(
            analysis=analysis,
            capture=capture,
            knowledge_context=knowledge_context,
            fingerprint=fingerprint,
        )
        analysis = await self.__guard_against_sensitive(
            analysis=analysis,
            capture=capture,
            knowledge_context=knowledge_context,
            fingerprint=fingerprint,
        )
        analysis_duration = time.time() - start
        ctx.metrics.record(operation="analysis", duration=analysis_duration)

        # Register the screen and persist the VLM description in one call, then
        # append any NEW rich observations to the activity description.
        await ctx.exploration_graph.add_screen(
            state=screen_state, description=analysis.screen_description
        )
        rich_text = analysis.metadata.get("rich_description", "")
        if rich_text and rich_text.strip():
            await ctx.exploration_graph.update_rich_description(
                visual_hash=fingerprint, rich_description=rich_text
            )

        # On a focused run, record how this screen relates to the focus so the
        # frontier ordering (F2) and resumed runs stay focus-aware.
        if ctx.focus and analysis.focus_relevance is not None:
            await ctx.exploration_graph.record_relevance(
                visual_hash=fingerprint, relevance=analysis.focus_relevance
            )

        # Record the screen's functional category (home, list, detail, ...) so the
        # per-screen documentation groups fingerprints into one logical screen.
        if analysis.category is not None:
            await ctx.exploration_graph.record_category(
                visual_hash=fingerprint, category=analysis.category
            )

        # Inspect each freshly-registered screen once for UI/content defects.
        await self.__inspect_screen(
            fingerprint=fingerprint, screen_state=screen_state, capture=capture
        )

        if analysis.content_exhausted:
            return await self.__handle_exhaustion(
                state=state,
                analysis=analysis,
                fingerprint=fingerprint,
                analysis_duration=analysis_duration,
            )

        # A usable action was picked: clear the exhaustion-retry counter so a
        # future stall on this screen is not pre-empted by a stale veto.
        dfs.exhaustion_retries.pop(fingerprint, None)
        action = analysis.action
        if action.action_type == ActionType.BACK:
            action = action.model_copy(update={"bounds": None})

        result = self.__mutable(state)
        result[EKey.ACTION] = action
        result[CKey.ANALYSIS] = analysis
        result[EKey.CONTENT_EXHAUSTED] = False
        result[CKey.ANALYSIS_DURATION] = analysis_duration
        return cast("ExplorationGraphState", result)

    async def execute(self, state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Execute the scan-recommended action via the ActionExecutor.
        """

        ctx = self.__context

        if ctx.is_cancelled:
            return self.__complete(state, reason=CompletionReason.CANCELLED)

        action = get_action(state)
        capture = get_capture(state)
        if not action or not capture:
            result = self.__mutable(state)
            result[CKey.STEP_RESULT] = None
            result[CKey.EXECUTION_DURATION] = 0.0
            return cast("ExplorationGraphState", result)

        screen_state = get_screen_state(state)
        step_result, duration = await self.__execute_action(
            action=action, capture=capture, screen_state=screen_state
        )

        result = self.__mutable(state)
        result[CKey.STEP_RESULT] = step_result
        result[CKey.EXECUTION_DURATION] = duration
        return cast("ExplorationGraphState", result)

    async def navigate(self, state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Drive BACKTRACK (hardware BACK) or ADVANCE (recovery replay) navigation.
        """

        ctx = self.__context
        dfs = self.__dfs

        if ctx.is_cancelled:
            return self.__complete(state, reason=CompletionReason.CANCELLED)

        outcome = self.__next_navigation_action()
        if outcome.is_complete:
            return self.__complete(state, reason=outcome.reason or _DFS_COMPLETE)
        if outcome.action is None:
            # Arrived at the recovery target with nothing to replay -> scan it.
            result = self.__mutable(state)
            result[EKey.BFS_PHASE] = dfs.phase.value
            return cast("ExplorationGraphState", result)

        capture = get_capture(state)
        if not capture:
            dfs.phase = BFSPhase.SCAN
            result = self.__mutable(state)
            result[EKey.BFS_PHASE] = dfs.phase.value
            return cast("ExplorationGraphState", result)

        screen_state = get_screen_state(state)
        step_result, duration = await self.__execute_action(
            action=outcome.action, capture=capture, screen_state=screen_state
        )

        result = self.__mutable(state)
        result[EKey.ACTION] = outcome.action
        result[CKey.ANALYSIS] = None
        result[CKey.STEP_RESULT] = step_result
        result[CKey.EXECUTION_DURATION] = duration
        return cast("ExplorationGraphState", result)

    async def record(self, state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Persist the step outcome, record the transition, and advance the DFS phase.
        """

        ctx = self.__context
        dfs = self.__dfs

        if ctx.is_cancelled:
            return self.__complete(state, reason=CompletionReason.CANCELLED)

        step_result = get_step_result(state)
        action = get_action(state)
        screen_state = get_screen_state(state)
        if not step_result:
            return state

        # A step actually completed: clear the no-progress watchdog counter.
        dfs.stalled_routes = 0

        # Step descriptors shared by transition recording and defect detection.
        activity = (
            screen_state.activity
            if isinstance(screen_state, ScreenState) and screen_state.activity
            else None
        )
        action_target = (action.natural_language_target or action.target) if action else None
        expects_transition = bool(
            action and action.expected_outcome and action.expected_outcome.implies_transition
        )

        # Keep the walk inside the target package before recording the step. An
        # unrecoverable exit strands the user, so flag it before ending the run.
        scope_reason = await self.__enforce_package_scope()
        if scope_reason is not None:
            ctx.agent_state.record_step(result=step_result)
            await self.__inspect_step(
                signals=StepSignals(
                    screen=step_result.pre_hash,
                    activity=activity,
                    action_target=action_target,
                    expects_transition=expects_transition,
                    screen_changed=False,
                    left_package=True,
                )
            )
            return self.__complete(state, reason=scope_reason)

        # Re-capture the post-action screen to resolve the destination hash.
        post_capture = await ctx.perception.perceive(
            session_id=ctx.workflow_id, step_number=ctx.agent_state.step_count
        )
        post_state = ctx.perception.build_state(capture=post_capture)
        post_hash = ctx.exploration_graph.resolve_hash(post_state.visual_hash)
        pre_hash = ctx.exploration_graph.resolve_hash(step_result.pre_hash)

        step_result = step_result.model_copy(
            update={"post_hash": post_hash, "screen_changed": pre_hash != post_hash}
        )
        ctx.agent_state.record_step(result=step_result)

        post_is_new = not ctx.exploration_graph.has_screen(visual_hash=post_hash)
        await ctx.exploration_graph.add_screen(state=post_state)

        await self.__persist_transition(
            action=action,
            pre_hash=pre_hash,
            post_hash=post_hash,
            success=step_result.success,
        )

        # Inline defect pass: a dead tap (a predicted change that never happened)
        # or a blank post-capture is evidence the user-facing step misbehaved.
        await self.__inspect_step(
            signals=StepSignals(
                screen=pre_hash,
                activity=activity,
                action_target=action_target,
                expects_transition=expects_transition,
                screen_changed=step_result.screen_changed,
                usable_capture=self.__is_usable_capture(
                    capture=post_capture, screen_state=post_state
                ),
            )
        )

        ctx.history.enqueue_save_step(
            result=step_result, intent="exploration", package_name=activity
        )

        results = list(get_step_results(state))
        results.append(step_result)

        # Plateau guard: end cleanly once exploration stops surfacing new screens
        # rather than spending the remaining step budget re-treading known ground.
        # Complements the routing watchdog, which instead fires when steps stop.
        if post_is_new:
            dfs.steps_since_new_screen = 0
        else:
            dfs.steps_since_new_screen += 1
        plateaued = dfs.steps_since_new_screen > MAX_STEPS_WITHOUT_NEW_SCREEN

        natural_complete = self.__advance_phase(
            action=action, pre_hash=pre_hash, post_hash=post_hash, post_is_new=post_is_new
        )
        max_steps_reached = ctx.agent_state.step_count >= ctx.max_steps
        is_complete = natural_complete or plateaued or max_steps_reached

        result = self.__mutable(state)
        result[CKey.STEP_RESULT] = step_result
        result[EKey.STEP_RESULTS] = results
        result[CKey.STEP_NUMBER] = ctx.agent_state.step_count
        result[EKey.BFS_PHASE] = dfs.phase.value
        result[CKey.IS_COMPLETE] = is_complete
        if max_steps_reached:
            result[CKey.COMPLETION_REASON] = CompletionReason.MAX_STEPS
        elif natural_complete:
            result[CKey.COMPLETION_REASON] = _DFS_COMPLETE
        elif plateaued:
            result[CKey.COMPLETION_REASON] = CompletionReason.COVERAGE_PLATEAU

        await self.__emit_progress(action=action)
        return cast("ExplorationGraphState", result)

    # ── Routers ────────────────────────────────────────────────────────────

    def after_ground(self, state: ExplorationGraphState) -> NodeName:
        """
        Proceed to phase routing, or end when cancelled or capture failed.
        """

        if self.__context.is_cancelled or is_complete(state) or get_capture(state) is None:
            return NodeName.END
        return NodeName.BFS_ROUTE

    def after_bfs_route(self, state: ExplorationGraphState) -> NodeName:
        """
        Dispatch to scan or navigate by the published DFS phase.
        """

        dfs = self.__dfs
        if self.__context.is_cancelled or is_complete(state):
            return NodeName.END

        phase = get_bfs_phase(state, BFSPhase.SCAN.value)
        if phase == BFSPhase.SCAN.value:
            return NodeName.SCAN
        if phase == BFSPhase.BACKTRACK.value:
            return NodeName.NAVIGATE
        if phase == BFSPhase.ADVANCE.value:
            if not dfs.bfs_queue and not dfs.pending_nav:
                return NodeName.END
            return NodeName.NAVIGATE
        return NodeName.END

    def after_scan(self, state: ExplorationGraphState) -> NodeName:
        """
        Execute the chosen action, or loop back to routing when exhausted.
        """

        if self.__context.is_cancelled or is_complete(state):
            return NodeName.END
        if is_content_exhausted(state) or get_action(state) is None:
            return NodeName.BFS_ROUTE
        return NodeName.EXECUTE

    def after_record(self, state: ExplorationGraphState) -> NodeName:
        """
        Loop back to grounding, or end when complete or unable to continue.
        """

        ctx = self.__context
        if ctx.is_cancelled or is_complete(state):
            return NodeName.END
        if not ctx.agent_state.can_continue:
            return NodeName.END
        return NodeName.GROUND

    def node_callables(self) -> Dict[str, Callable[..., Any]]:
        """
        Maps node names to this provider's bound async node methods.
        """

        return {
            NodeName.GROUND: self.ground,
            NodeName.BFS_ROUTE: self.bfs_route,
            NodeName.SCAN: self.scan,
            NodeName.EXECUTE: self.execute,
            NodeName.NAVIGATE: self.navigate,
            NodeName.RECORD: self.record,
        }

    # ── Scan helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def __recent_actions(*, state: ExplorationGraphState) -> List[ActionOutcome]:
        """
        Project the last few executed steps into outcomes the scan can react to.
        """

        return [
            ActionOutcome.from_step_result(result=result)
            for result in get_step_results(state)[-RECENT_ACTION_WINDOW:]
        ]

    def __parent_description(self) -> Optional[str]:
        """
        Description of the parent screen on the current DFS path, if known.
        """

        dfs = self.__dfs
        if not dfs.current_path:
            return None
        parent_hash = dfs.current_path[-1][0]
        parent_node = self.__context.exploration_graph.get_screen(visual_hash=parent_hash)
        return parent_node.description if parent_node else None

    async def __guard_against_repeats(
        self,
        *,
        analysis: AnalysisResult,
        capture: Any,
        knowledge_context: str,
        fingerprint: str,
    ) -> AnalysisResult:
        """
        Re-prompt until the model picks a novel action or the retry budget runs out.

        Rejects actions already tried on this screen and categories sampled past
        their per-screen cap, forcing content-exhaustion if the budget is spent.
        """

        ctx = self.__context
        dedup = self.__dedup
        tried_keys = {
            ActionKey(
                kind=tried.action_type.lower(),
                label=(tried.coord_bucket or tried.target).lower(),
            )
            for tried in ctx.exploration_graph.get_tried_actions(visual_hash=fingerprint)
        }

        for retry in range(dedup.retries):
            if analysis.content_exhausted or dedup.is_repeatable(analysis.action):
                break

            failures = self.__sampling_rejection(action=analysis.action, fingerprint=fingerprint)
            if failures is not None:
                analysis = await self.__vision.scan(
                    capture=capture,
                    knowledge_context=knowledge_context,
                    intent=ctx.intent,
                    failures=failures,
                )
                continue

            if dedup.is_novel(action=analysis.action, tried=tried_keys):
                break

            action_key = dedup.key_for(analysis.action)
            logger.warning(
                "Dedup guard: repeated %s on screen %s (retry %d/%d)",
                action_key,
                fingerprint[:8],
                retry + 1,
                dedup.retries,
            )
            label = (
                analysis.action.natural_language_target
                or analysis.action.target
                or action_key.label
            )
            analysis = await self.__vision.scan(
                capture=capture,
                knowledge_context=knowledge_context,
                intent=ctx.intent,
                failures=[
                    f'You already tried {action_key.kind} "{label}" - '
                    "pick a DIFFERENT untried element."
                ],
            )
        else:
            if (
                not analysis.content_exhausted
                and not dedup.is_repeatable(analysis.action)
                and not dedup.is_novel(action=analysis.action, tried=tried_keys)
            ):
                logger.warning(
                    "Dedup guard exhausted retries on screen %s - forcing content_exhausted",
                    fingerprint[:8],
                )
                analysis.content_exhausted = True

        return analysis

    async def __guard_against_sensitive(
        self,
        *,
        analysis: AnalysisResult,
        capture: Any,
        knowledge_context: str,
        fingerprint: str,
    ) -> AnalysisResult:
        """
        Re-prompt away from actions that enter sensitive areas (payment, auth,
        destructive), exhausting the screen when the model keeps choosing one so
        the crawl describes it but backtracks instead of acting in.
        """

        guard = self.__traversal_guard
        if guard is None:
            return analysis

        for _ in range(MAX_SENSITIVE_ACTION_RETRIES):
            if analysis.content_exhausted:
                return analysis

            verdict = guard.inspect_action(
                target=analysis.action.natural_language_target or analysis.action.target or "",
                rationale=analysis.action.rationale or "",
            )
            if verdict.allowed or verdict.reason is None:
                return analysis

            logger.info(
                "Traversal guard vetoed a %s action on screen %s",
                verdict.category.value if verdict.category else "sensitive",
                fingerprint[:8],
            )
            analysis = await self.__vision.scan(
                capture=capture,
                knowledge_context=knowledge_context,
                intent=self.__context.intent,
                failures=[verdict.reason],
            )

        blocked = not guard.inspect_action(
            target=analysis.action.natural_language_target or analysis.action.target or "",
            rationale=analysis.action.rationale or "",
        ).allowed
        if blocked and not analysis.content_exhausted:
            analysis.content_exhausted = True
        return analysis

    def __sampling_rejection(self, *, action: Action, fingerprint: str) -> Optional[List[str]]:
        """
        Returns rejection feedback when the action's category is over-sampled.
        """

        category = action.element_category
        limit = self.__dedup.limit_for(category)
        if category is None or limit is None:
            return None

        sampled = self.__context.exploration_graph.count_category_taps(
            visual_hash=fingerprint, category=category
        )
        if not self.__dedup.is_over_sampled(category=category, sampled=sampled):
            return None

        logger.warning(
            "Sampling guard: %s sampled %d/%d on screen %s - rejecting",
            category,
            sampled,
            limit,
            fingerprint[:8],
        )
        return [
            f"You have already sampled {sampled} {category} elements on this screen "
            f"(limit {limit}). Per LIST SAMPLING, the rest are effectively tried - pick a "
            "DIFFERENT category (P1 global_navigation, P2 primary_action, P5 secondary_control) "
            "or press BACK."
        ]

    async def __handle_exhaustion(
        self,
        *,
        state: ExplorationGraphState,
        analysis: AnalysisResult,
        fingerprint: str,
        analysis_duration: float,
    ) -> ExplorationGraphState:
        """
        Apply the depth-floor veto or commit to backtracking from this screen.
        """

        dfs = self.__dfs
        retries = dfs.exhaustion_retries.get(fingerprint, 0)
        depth = dfs.depth

        if self.__depth_floor.should_veto(depth=depth, retries=retries):
            dfs.exhaustion_retries[fingerprint] = retries + 1
            logger.info(
                "Depth-floor veto: screen %s exhausted at depth %d (< %d); re-prompting",
                fingerprint[:8],
                depth,
                self.__depth_floor.minimum,
            )
            exhausted = False
        else:
            dfs.fully_scanned.add(fingerprint)
            await self.__context.exploration_graph.mark_exhausted(visual_hash=fingerprint)
            dfs.phase = BFSPhase.BACKTRACK
            logger.info("Screen %s fully scanned, backtracking (depth=%d)", fingerprint[:8], depth)
            exhausted = True

        result = self.__mutable(state)
        result[EKey.ACTION] = None
        result[CKey.ANALYSIS] = analysis
        result[EKey.CONTENT_EXHAUSTED] = exhausted
        result[CKey.ANALYSIS_DURATION] = analysis_duration
        result[EKey.BFS_PHASE] = dfs.phase.value
        return cast("ExplorationGraphState", result)

    # ── Execution + persistence helpers ──────────────────────────────────────

    async def __execute_action(
        self, *, action: Action, capture: Any, screen_state: Optional[ScreenState]
    ) -> tuple[StepResult, float]:
        """
        Run one action through the ActionExecutor and build its StepResult.
        """

        ctx = self.__context
        pre_hash = screen_state.visual_hash if screen_state else "0"
        step = Step(action=action, screen_hash=pre_hash, step_number=ctx.agent_state.step_count)

        start_time = time.time()
        execution_result = await ctx.action_executor.act(
            step=step,
            pre_capture=capture,
            package_name=ctx.package_name,
            session_id=ctx.workflow_id,
        )
        # Settle the screen before record re-captures the post-action state.
        await stability_wait(ctx.configuration)
        duration = time.time() - start_time
        ctx.metrics.record(operation="action", duration=duration)

        step_result = StepResult(
            step=step,
            error=execution_result.error,
            pre_hash=pre_hash,
            success=execution_result.success,
            duration=int(duration * 1000),
            post_hash="0",  # Filled in by record after re-capture.
            screen_changed=True,
        )
        return step_result, duration

    async def __persist_transition(
        self, *, action: Optional[Action], pre_hash: str, post_hash: str, success: bool
    ) -> None:
        """
        Record the screen transition and the action experience in parallel.
        """

        if action is None:
            return

        ctx = self.__context
        writes: List[Any] = []
        if pre_hash != "0":
            writes.append(
                ctx.exploration_graph.record_transition(
                    source_hash=pre_hash, action=action, destination_hash=post_hash
                )
            )
        writes.append(
            ctx.memory.store_experience(visual_hash=pre_hash, action=action, success=success)
        )
        await asyncio.gather(*writes, return_exceptions=True)

    async def __inspect_step(self, *, signals: StepSignals) -> None:
        """
        Detect inline defects for one step, persist them, and surface them live.
        """

        for defect in self.__inline_detector.inspect_step(signals=signals):
            await self.__record_defect(defect=defect)

    async def __inspect_screen(self, *, fingerprint: str, screen_state: Any, capture: Any) -> None:
        """
        Run the screen-level defect detector once per unique screen, best-effort.
        """

        if self.__screen_detector is None or fingerprint in self.__inspected_screens:
            return
        self.__inspected_screens.add(fingerprint)

        image = getattr(capture, "image", None)
        if not image:
            return

        activity = screen_state.activity if isinstance(screen_state, ScreenState) else None
        snapshot = ScreenSnapshot(screen=fingerprint, activity=activity, screenshot=image)
        try:
            defects = await self.__screen_detector.inspect_screen(snapshot=snapshot)
        except Exception:
            # Defect inspection is best-effort enrichment; never let it break the crawl.
            logger.warning("Screen defect inspection failed for %s", fingerprint[:8], exc_info=True)
            return

        for defect in defects:
            await self.__record_defect(defect=defect)

    async def __record_defect(self, *, defect: Defect) -> None:
        """
        Persist one defect to the shared repository and surface it live.
        """

        ctx = self.__context
        if self.__defects is not None:
            await self.__defects.record(session=ctx.workflow_id, defect=defect)
        await ctx.telemetry.info(
            DEFECT_DETECTED_EVENT,
            signal=defect.signal.value,
            kind=defect.kind.value,
            severity=defect.severity.value,
            screen=defect.evidence.screen,
            summary=defect.summary,
        )

    async def __emit_progress(self, *, action: Optional[Action]) -> None:
        """
        Publish a per-step progress snapshot for live observers (e.g. the TUI).
        """

        ctx = self.__context
        dfs = self.__dfs
        screens = ctx.exploration_graph.node_count
        coverage = round(len(dfs.fully_scanned) / screens * 100.0, 1) if screens else 0.0
        await ctx.telemetry.info(
            EXPLORATION_PROGRESS_EVENT,
            step=ctx.agent_state.step_count,
            max_steps=ctx.max_steps,
            phase=dfs.phase.value,
            unique_screens=screens,
            coverage=coverage,
            action=self.__action_summary(action=action),
        )

    @staticmethod
    def __action_summary(*, action: Optional[Action]) -> str:
        """
        Short human-readable label for the action that drove a step.
        """

        if action is None:
            return "navigation"
        label = action.natural_language_target or action.target or ""
        return f"{action.action_type.value} {label}".strip()

    # ── DFS transition logic ─────────────────────────────────────────────────

    def __advance_phase(
        self, *, action: Optional[Action], pre_hash: str, post_hash: str, post_is_new: bool
    ) -> bool:
        """
        Drive the SCAN/BACKTRACK/ADVANCE transition after an action lands.

        Returns whether the DFS has exhausted all reachable screens.
        """

        dfs = self.__dfs

        if dfs.phase == BFSPhase.SCAN:
            self.__advance_from_scan(action=action, pre_hash=pre_hash, post_hash=post_hash)
            return False
        if dfs.phase == BFSPhase.BACKTRACK:
            return self.__advance_from_backtrack(post_hash=post_hash)
        if dfs.phase == BFSPhase.ADVANCE:
            self.__advance_from_recovery(post_hash=post_hash)
            return False
        return False

    def __advance_from_scan(
        self, *, action: Optional[Action], pre_hash: str, post_hash: str
    ) -> None:
        """
        After a scan action: descend onto the new screen or backtrack if revisited.
        """

        dfs = self.__dfs
        if pre_hash == post_hash:
            return  # Stayed on the same screen; remain in SCAN.

        if action is not None and action.action_type == ActionType.BACK:
            if dfs.current_path:
                dfs.current_path = dfs.current_path[:-1]
        elif action is not None:
            dfs.current_path = [*dfs.current_path, (pre_hash, action)]

        if post_hash in dfs.fully_scanned:
            dfs.phase = BFSPhase.BACKTRACK
        else:
            dfs.phase = BFSPhase.SCAN

    def __advance_from_backtrack(self, *, post_hash: str) -> bool:
        """
        After a BACK press: scan the landed screen, keep climbing, or recover.
        """

        dfs = self.__dfs
        if dfs.current_path:
            dfs.current_path = dfs.current_path[:-1]

        if post_hash not in dfs.fully_scanned:
            dfs.phase = BFSPhase.SCAN
            dfs.scanning_hash = post_hash
            return False
        if dfs.current_path:
            dfs.phase = BFSPhase.BACKTRACK
            return False

        orphans = self.__navigator.find_orphaned_screens()
        if orphans:
            dfs.bfs_queue.extend(orphans)
            dfs.phase = BFSPhase.ADVANCE
            return False
        return True

    def __advance_from_recovery(self, *, post_hash: str) -> None:
        """
        After a recovery hop: scan on arrival or when the replay runs out.
        """

        dfs = self.__dfs
        if post_hash == dfs.scanning_hash:
            dfs.pending_nav.clear()
            dfs.phase = BFSPhase.SCAN
        elif not dfs.pending_nav:
            dfs.phase = BFSPhase.SCAN
            dfs.scanning_hash = post_hash
            dfs.current_path = self.__navigator.path_to_screen(screen_hash=post_hash)

    def __next_navigation_action(self) -> _NavigationOutcome:
        """
        Resolve the next BACKTRACK/ADVANCE action, dequeuing recovery targets.
        """

        dfs = self.__dfs

        if dfs.phase == BFSPhase.BACKTRACK:
            if dfs.current_path:
                return _NavigationOutcome(
                    action=Action(
                        action_type=ActionType.BACK,
                        confidence=1.0,
                        target="back navigation",
                        rationale="DFS: backtracking from exhausted screen",
                    )
                )
            # At the root with an exhausted path: switch to recovery.
            orphans = self.__navigator.find_orphaned_screens()
            if not orphans:
                return _NavigationOutcome(is_complete=True, reason=_DFS_COMPLETE)
            dfs.bfs_queue.extend(orphans)
            dfs.phase = BFSPhase.ADVANCE
            return self.__begin_recovery()

        if dfs.pending_nav:
            return _NavigationOutcome(action=dfs.pending_nav.pop(0))

        return self.__begin_recovery()

    def __begin_recovery(self) -> _NavigationOutcome:
        """
        Dequeue the next unscanned recovery target and plan navigation to it.
        """

        dfs = self.__dfs
        entry = self.__next_recovery_entry()
        if entry is None:
            return _NavigationOutcome(is_complete=True, reason=_DFS_COMPLETE)

        dfs.pending_nav = self.__navigator.compute_navigation(
            current_path=dfs.current_path, target_path=entry.path_from_root
        )
        dfs.scanning_hash = entry.screen_hash
        dfs.current_path = list(entry.path_from_root)

        if not dfs.pending_nav:
            dfs.phase = BFSPhase.SCAN
            return _NavigationOutcome()
        return _NavigationOutcome(action=dfs.pending_nav.pop(0))

    def __next_recovery_entry(self) -> Optional[Any]:
        """
        Pop the next recovery-queue entry that has not been fully scanned.
        """

        dfs = self.__dfs
        while dfs.bfs_queue:
            entry = dfs.bfs_queue.popleft()
            if entry.screen_hash not in dfs.fully_scanned:
                return entry
        return None

    # ── Interrupt + scope helpers ────────────────────────────────────────────

    async def __check_interrupts(self) -> Optional[str]:
        """
        Honour external pause/cancel signals before grounding the next step.

        Returns a completion reason when the run was cancelled, else None.
        """

        ctx = self.__context
        if await ctx.hitl.check_signal() == "CANCELLED":
            return CompletionReason.CANCELLED
        if await ctx.hitl.is_pause_requested():
            logger.info("[HITL] exploration %s paused; awaiting resume", ctx.workflow_id)
            await ctx.hitl.wait_for_resume()
        return None

    async def __enforce_package_scope(self) -> Optional[str]:
        """
        Keep exploration within the target package, recovering transient drift.

        Presses BACK up to ``PACKAGE_RECOVERY_BACK_LIMIT`` times to dismiss
        out-of-app overlays (share sheets, permission prompts). On recovery the
        DFS navigation state is reset so the next step re-orients; if the device
        stays outside the package, returns a terminal completion reason.
        """

        ctx = self.__context
        target = ctx.package_name
        if not target or target == "unknown":
            return None

        current = await self.__current_package()
        if current is None or current == target:
            return None

        for _ in range(PACKAGE_RECOVERY_BACK_LIMIT):
            await ctx.device.back()
            await stability_wait(ctx.configuration)
            if await self.__current_package() == target:
                dfs = self.__dfs
                dfs.pending_nav.clear()
                dfs.current_path = []
                dfs.phase = BFSPhase.SCAN
                logger.info("Recovered to target package %s; DFS navigation reset", target)
                return None

        logger.error("Left target package %s and could not recover", target)
        return f"Left target package {target} and could not recover"

    async def __current_package(self) -> Optional[str]:
        """
        Returns the foreground package, or None when the device cannot report it.

        A transient unparseable focus (mid-launch animation, lock screen, empty
        focus) must not abort exploration, so an indeterminate read is treated as
        a package that cannot be confirmed rather than a scope violation.
        """

        try:
            return await self.__context.device.get_current_package()
        except DeviceError:
            logger.debug("Foreground package indeterminate; skipping scope check this step")
            return None

    # ── State helpers ────────────────────────────────────────────────────────

    @staticmethod
    def __mutable(state: ExplorationGraphState) -> Dict[str, Any]:
        """
        Returns a shallow mutable copy of the graph state for in-place updates.
        """

        return cast("Dict[str, Any]", dict(state))

    def __complete(self, state: ExplorationGraphState, *, reason: str) -> ExplorationGraphState:
        """
        Returns a terminal state carrying the completion reason.
        """

        result = self.__mutable(state)
        result[CKey.IS_COMPLETE] = True
        result[CKey.COMPLETION_REASON] = reason
        return cast("ExplorationGraphState", result)


class _NavigationOutcome:
    """
    The next navigation step: an action to run, a no-op, or DFS completion.
    """

    __slots__ = ("action", "is_complete", "reason")

    def __init__(
        self,
        *,
        action: Optional[Action] = None,
        is_complete: bool = False,
        reason: Optional[str] = None,
    ) -> None:
        self.action = action
        self.is_complete = is_complete
        self.reason = reason


class ExplorationGraphFactory:
    """
    Factory for the exploration node provider.
    """

    @staticmethod
    def build(context: GraphContext) -> ExplorationNodeProvider:
        """
        Builds the provider that supplies the exploration nodes and routers.
        """

        return ExplorationNodeProvider(
            context=context,
            defects=context.defect_repository,
            screen_detector=VisionDefectDetector(
                llm=context.llm, use_cache=context.configuration.llm.use_cache
            ),
            # Guardrails apply only to broad-coverage runs; a focused run may be a
            # deliberate request to exercise a sensitive flow, so it is not guarded.
            traversal_guard=TraversalGuard() if not context.focus else None,
        )
