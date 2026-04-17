from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union, cast

from fathom.constants import ActionType, FathomEvent
from fathom.constants.execution import (
    GCC_BRANCHING_THRESHOLD,
    GROUNDING_FAILURE_MESSAGE,
    LAUNCHER_PACKAGES,
    PLANNER_RETRY_ESCALATION_THRESHOLD,
    RECORDING_FAILURE_MESSAGE,
    REDECOMPOSE_VERIFY_FAILURE_THRESHOLD,
    VISUAL_HASH_LENGTH,
)
from fathom.constants.graph import NodeName
from fathom.constants.screen import ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD, ZERO_HASH
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.exceptions import FathomError
from fathom.core.prompts.verification import (
    SUBGOAL_VERIFICATION_SYSTEM,
    VERIFICATION_SYSTEM,
    build_intent_verification_user_prompt,
    build_subgoal_verification_user_prompt,
)
from fathom.core.services.comparator import ScreenComparator
from fathom.core.services.hitl import HITLService
from fathom.schemas.hierarchy import HierarchyProcessingResult
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.screens import (
    PostActionScreenComparison,
    ScreenCapture,
    ScreenDiff,
    ScreenHashBundle,
    ScreenState,
)
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.ui import LabeledElement
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.state import IntentGraphState
from fathom.utils.parsing import strip_code_fences
from fathom.utils.wait import stability_wait

logger = logging.getLogger(__name__)


class IntentNodeProvider:
    """
    Provides LangGraph nodes for intent execution.
    Encapsulates dependencies and shared private logic.
    """

    def __init__(
        self,
        *,
        context: GraphContext,
        screen_comparator: ScreenComparator,
    ) -> None:
        """
        Initialize provider with shared context.
        """

        self.__context = context
        self.__screen_comparator = screen_comparator

    def __build_screen_state(
        self,
        *,
        capture: ScreenCapture,
        visual_hash: str,
        xml_hash: Optional[str] = None,
        interaction_hash: Optional[str] = None,
    ) -> ScreenState:
        """
        Build a normalized `ScreenState` from the available capture signals.
        """

        return ScreenState(
            activity=capture.activity,
            timestamp=capture.timestamp,
            activity_hash=hashlib.md5(
                capture.activity.encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()[:VISUAL_HASH_LENGTH],
            visual_hash=visual_hash,
            xml_hash=xml_hash,
            interaction_hash=interaction_hash,
        )

    def __resolve_capture_hashes(
        self,
        *,
        capture: ScreenCapture,
        elements: Optional[List[LabeledElement]] = None,
    ) -> ScreenHashBundle:
        """
        Compute the visual, XML, and interaction hashes for one capture.
        """

        return ScreenHashBundle(
            visual_hash=self.__context.perception.compute_visual_hash(capture=capture),
            xml_hash=self.__context.perception.compute_xml_hash(capture=capture),
            interaction_hash=self.__context.perception.compute_interaction_hash(elements=elements),
        )

    def __build_post_action_elements(
        self,
        *,
        capture: ScreenCapture,
    ) -> List[LabeledElement]:
        """
        Extract post-action interactive elements when XML is available.
        """

        if not self.__context.use_xml or not capture.xml_content:
            return []

        return self.__context.hierarchy.extract_elements(
            xml=capture.xml_content,
            screen=capture,
            action_type=ActionType.TAP,
        )

    async def __capture_post_action_screen(
        self,
        *,
        before_capture: ScreenCapture,
        before_state: Optional[ScreenState],
    ) -> PostActionScreenComparison:
        """
        Capture the post-action screen and compare it to the pre-action state.
        """

        capture_start = time.time()
        post_capture = await self.__context.perception_port.capture()
        capture_duration = time.time() - capture_start

        logger.info(
            "[NODE: EXECUTE] Post-action capture completed in %.2fs (image=%d bytes)",
            capture_duration,
            len(post_capture.image) if post_capture.image else 0,
        )

        if not post_capture.image:
            return PostActionScreenComparison()

        elements_start = time.time()
        post_elements = self.__build_post_action_elements(capture=post_capture)
        logger.info(
            "[NODE: EXECUTE] Post-action elements extracted in %.3fs (count=%d)",
            time.time() - elements_start,
            len(post_elements),
        )

        hash_start = time.time()
        post_hashes = self.__resolve_capture_hashes(
            capture=post_capture,
            elements=post_elements,
        )
        logger.info(
            "[NODE: EXECUTE] Post-action hashes computed in %.3fs", time.time() - hash_start
        )

        after_state = self.__build_screen_state(
            capture=post_capture,
            xml_hash=post_hashes.xml_hash,
            visual_hash=post_hashes.visual_hash,
            interaction_hash=post_hashes.interaction_hash,
        )

        diff_start = time.time()
        screen_diff = await asyncio.to_thread(
            self.__screen_comparator.compare,
            after=post_capture,
            before=before_capture,
            after_state=after_state,
            before_state=before_state,
        )
        logger.info(
            "[NODE: EXECUTE] Screen diff completed in %.2fs (off event loop)",
            time.time() - diff_start,
        )

        return PostActionScreenComparison(
            screen_diff=screen_diff,
            post_visual_hash=post_hashes.visual_hash,
        )

    async def __is_cancelled(self) -> bool:
        """
        Consolidated check for execution cancellation.
        """

        if self.__context.is_cancelled:
            return True

        signal = await self.__context.hitl.check_signal()
        return signal == "CANCELLED"

    async def __handle_hitl(self, current_step: int) -> None:
        """
        Orchestrates Human-In-The-Loop interruptions.

        After resume, drains any injected context from the signal queue
        and injects it as user guidance so the next LLM call sees it.
        """

        _ = current_step

        if (
            isinstance(self.__context.hitl, HITLService)
            and await self.__context.hitl.is_pause_requested()
        ):
            logger.info(
                f"[HITL] Workflow {self.__context.workflow_id} is paused. "
                "Waiting for resume/context."
            )
            await self.__context.hitl.wait_for_resume()

            # Drain injected context after resume — same logic as executor.__handle_interrupt
            consumed = 0
            while await self.__context.hitl.has_injected_context():
                context = await self.__context.hitl.peek_next_context()
                if not context:
                    break

                consumed += 1
                logger.info("[HITL] Processing injected context %d: '%s...'", consumed, context)

                await self.__context.context_manager.inject_user_guidance(
                    guidance=context,
                    step=self.__context.agent_state.step_count,
                )
                await self.__context.hitl.consume_context()

            if consumed > 0:
                logger.info("[HITL] Processed %d user contexts after resume", consumed)

    def __restore_agent_state_from_graph(self, *, state: IntentGraphState) -> None:
        """
        Restore agent_state from graph checkpoint if present.
        This is intended for graph resume / checkpoint recovery.

        IMPORTANT: Do not call this at the start of every node. Replacing the live
        AgentState object repeatedly can discard in-flight updates made by earlier
        nodes within the same graph run. Restore once at loop entry (e.g. ANALYZE),
        then rely on the live AgentState across EXECUTE/RECORD/VERIFY, while
        persisting back to graph state at node boundaries.
        """

        checkpoint = state.get(IntentStateKey.AGENT_STATE_CHECKPOINT)
        current_index_value = state.get(IntentStateKey.CURRENT_SUB_GOAL_INDEX, 0)
        current_index = (
            int(current_index_value) if isinstance(current_index_value, (int, str)) else 0
        )

        if checkpoint and isinstance(checkpoint, dict):
            logger.debug(
                f"[SYNC] Restoring agent_state from graph checkpoint: "
                f"step_count={checkpoint.get('step_count')}, "
                f"sub_goal_index={checkpoint.get('current_sub_goal_index')}"
            )
            # Restore from checkpoint (this handles sub-goals and index)
            from fathom.core.agent.state import AgentState

            restored = AgentState.from_checkpoint(checkpoint)
            # Replace context's agent_state with restored version using public setter
            self.__context.set_agent_state(restored)
        elif current_index > 0:
            logger.debug(f"[SYNC] Updating sub-goal index from graph: {current_index}")
            # Partial update: just index (if checkpoint not available)
            if self.__context.agent_state.sub_goal_list and current_index < len(
                self.__context.agent_state.sub_goal_list
            ):
                self.__context.agent_state.set_current_sub_goal_index(current_index)

    async def __advance_after_verify_passed(
        self,
        *,
        is_subgoal_verify: bool,
        current_sub_goal: Optional[SubGoal],
        reason: str,
    ) -> IntentGraphState:
        """
        Handle the ``is_truly_complete`` branch of ``verify``.

        Extracted so the sub-goal transition cues (SUB_GOAL_COMPLETED →
        SUB_GOAL_STARTED) can be unit-tested without standing up the
        full verification machinery. Three outcomes:

        - sub-goal verified with more remaining: emit COMPLETED +
          STARTED, return a retry result;
        - sub-goal verified with none remaining: emit COMPLETED, mark
          the intent complete, return the terminal result;
        - full-intent verify (no sub-goals): mark complete, return the
          terminal result with the LLM's rationale as the reason.
        """

        agent_state = self.__context.agent_state

        if not (is_subgoal_verify and current_sub_goal is not None):
            # Full intent verified — exit.
            agent_state.mark_complete(reason=reason)
            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: reason,
                },
            )
            self.__persist_agent_state_to_graph(result=result)
            return result

        # Sub-goal verified — advance to next sub-goal (or complete intent).
        has_more = agent_state.mark_current_sub_goal_complete(
            completion_signal=SubGoalCompletionSignal(
                evidence=f"Verified by screenshot: {reason}",
                llm_signaled=True,
                rationale_verified=False,
                action_executed=True,
                screen_verified=True,
            ),
        )
        agent_state.reset_completion()

        # Clear stale verification feedback from previous sub-goals
        # so it doesn't pollute the next sub-goal's context.
        self.__context.context_manager.clear_user_guidance()

        total_sub_goals = len(agent_state.sub_goal_list)

        # Cue: the just-finished sub-goal.
        await self.__context.telemetry.info(
            "Sub-goal completed",
            type=FathomEvent.SUB_GOAL_COMPLETED,
            index=current_sub_goal.index,
            total=total_sub_goals,
            description=current_sub_goal.description,
        )

        if has_more:
            next_sg = agent_state.get_current_sub_goal()
            logger.info(
                f"[NODE: VERIFY] ✓ Sub-goal {current_sub_goal.index} VERIFIED. "
                f"Advancing to sub-goal {next_sg.index if next_sg else '(none)'}: "
                f"'{next_sg.description if next_sg else ''}'"
            )

            # Cue: the newly-active sub-goal.
            if next_sg is not None:
                await self.__context.telemetry.info(
                    "Sub-goal started",
                    type=FathomEvent.SUB_GOAL_STARTED,
                    index=next_sg.index,
                    total=total_sub_goals,
                    description=next_sg.description,
                )
            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: False,
                    IntentStateKey.SHOULD_RETRY: True,
                },
            )
            self.__persist_agent_state_to_graph(result=result)
            return result

        # Last sub-goal verified — mark intent complete.
        logger.info("[NODE: VERIFY] All sub-goals verified. Marking intent complete.")
        agent_state.mark_complete(
            reason="All sub-goals completed and verified sequentially"
        )
        result = cast(
            "IntentGraphState",
            {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: (
                    "All sub-goals completed and verified sequentially"
                ),
            },
        )
        self.__persist_agent_state_to_graph(result=result)
        return result

    def __persist_agent_state_to_graph(
        self,
        *,
        result: Union[IntentGraphState, Dict[str, Any]],
    ) -> None:
        """
        Serialize agent_state updates back to graph state for checkpoint persistence.
        This ensures sub-goal progress survives loop recovery and graph restarts.
        """

        checkpoint = self.__context.agent_state.to_checkpoint()
        current_index = self.__context.agent_state.current_sub_goal_index

        logger.debug(
            f"[SYNC] Persisting agent_state to graph: "
            f"step_count={checkpoint.get('step_count')}, "
            f"sub_goal_index={current_index}"
        )

        # Cast to dict to allow enum key access
        result_dict = cast("Dict[str, Any]", result)
        result_dict[IntentStateKey.AGENT_STATE_CHECKPOINT.value] = checkpoint
        result_dict[IntentStateKey.CURRENT_SUB_GOAL_INDEX.value] = current_index

    def __should_skip_launcher_persistence(
        self, *, execution_activity: str, observed_activity: str
    ) -> bool:
        """
        Skip persistence only when the step both starts and ends on the launcher.
        """

        execution_package = execution_activity.split("/")[0]
        observed_package = observed_activity.split("/")[0]

        if execution_package not in LAUNCHER_PACKAGES:
            return False

        return observed_package in LAUNCHER_PACKAGES or observed_package == "unknown"

    async def __publish_generated_script(self, *, script_data: str, step_number: int) -> None:
        """
        Publish generated script telemetry after background history persistence completes.
        """

        await self.__context.telemetry.info(
            script_data,
            type=FathomEvent.SCRIPT_GENERATED,
            step=step_number + 1,
        )

    def __enqueue_history_persistence(
        self,
        *,
        step_result: StepResult,
        current_activity: Optional[str],
        execution_activity: Optional[str] = None,
    ) -> None:
        """
        Queue ordered history persistence for the completed step.
        """

        async def __publish(script_data: str) -> None:
            await self.__publish_generated_script(
                script_data=script_data,
                step_number=step_result.step.step_number,
            )

        self.__context.history.enqueue_save_step(
            result=step_result,
            intent=self.__context.intent,
            package_name=current_activity,
            execution_activity=execution_activity,
            on_complete=__publish,
        )

    async def __replan_remaining_sub_goals(
        self,
        capture: ScreenCapture,
        *,
        failure_reason: Optional[str] = None,
        suggested_next_action: Optional[str] = None,
    ) -> Optional[IntentGraphState]:
        """Re-decompose the remaining intent with context from the failure.

        Called once per failure burst, from the VERIFY node after
        ``REDECOMPOSE_VERIFY_FAILURE_THRESHOLD`` rejections on the same
        sub-goal. Receives the verifier's rejection reason and suggested
        next action so the decomposer can propose a different approach.

        The decomposer receives:
          1. The remaining (unfinished) sub-goals joined as the new intent.
          2. The stuck sub-goal's description separately.
          3. The failure reason + verifier-suggested action when available.
          4. A short tail of recent action lines so the planner can see
             which approaches have already been tried.
          5. The current screenshot for visual grounding.
        """

        from fathom.core.prompts.trace import format_trace_action_line
        from fathom.core.services.decomposer import IntentDecomposer

        agent_state = self.__context.agent_state
        stuck_sub_goal = agent_state.get_current_sub_goal()
        stuck_description = stuck_sub_goal.description if stuck_sub_goal else None

        # Build a new intent from only the remaining (unfinished) sub-goals
        # so the decomposer never re-plans completed steps.
        remaining_descriptions = [
            sg.description for sg in agent_state.sub_goals if not sg.is_complete()
        ]
        if not remaining_descriptions:
            return None
        remaining_intent = ". ".join(remaining_descriptions)

        # Collect a short tail of recent actions so the replan can steer
        # away from whatever the agent just tried.
        recent_actions: tuple[str, ...] = ()
        try:
            full_context = self.__context.context_manager.get_full_context()
            trace = full_context.get("trace", [])
            if trace:
                recent_actions = tuple(
                    format_trace_action_line(entry) for entry in list(trace)[-10:]
                )
        except Exception as exc:  # defensive — trace access must not break replan
            logger.debug("[REPLAN] Failed to collect recent_actions: %s", exc)

        # When the caller did not supply a reason, synthesize one from the
        # sub-goal action counter so the decomposer still sees *why*.
        resolved_failure_reason = failure_reason
        if not resolved_failure_reason:
            resolved_failure_reason = (
                f"Agent executed {agent_state.sub_goal_action_count} actions on "
                f"this sub-goal without advancing (loop detected)."
            )

        try:
            decomposer = IntentDecomposer.with_configuration(
                llm=self.__context.llm,
                configuration=self.__context.configuration.llm,
            )
            new_sub_goals = await decomposer.decompose(
                intent=remaining_intent,
                screenshot=capture.image if capture.image else None,
                stuck_sub_goal=stuck_description,
                failure_reason=resolved_failure_reason,
                suggested_next_action=suggested_next_action,
                recent_actions=recent_actions,
            )

            logger.info(
                f"[NODE: VERIFY] Replanned: {len(new_sub_goals)} new sub-goals "
                f"replacing {len(remaining_descriptions)} unfinished sub-goals"
            )

            agent_state.replace_remaining_sub_goals(new_sub_goals)
            agent_state.reset_completion()
            self.__context.context_manager.clear_user_guidance()

            result_state = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: False,
                    IntentStateKey.SHOULD_RETRY: True,
                },
            )
            self.__persist_agent_state_to_graph(result=result_state)
            return result_state

        except Exception as exception:
            logger.warning(f"[NODE: VERIFY] Replanning failed: {exception}")

        return None

    async def __maybe_escalate_on_planner_retries(
        self,
        *,
        plan: PlanResult,
        capture: ScreenCapture,
    ) -> Optional[IntentGraphState]:
        """Escalate to the decomposer replan when the stuck-retry counter
        hits its threshold.

        Called from the ANALYZE node after ``StepPlanner.plan_step``
        returns. Fires only when the planner signaled
        ``ACTION_BLOCKED`` AND ``sub_goal_planner_retries`` has reached
        ``PLANNER_RETRY_ESCALATION_THRESHOLD``. Otherwise returns
        ``None`` so the caller falls through to normal retry routing.

        Parallels the VERIFY-side escalation at
        ``REDECOMPOSE_VERIFY_FAILURE_THRESHOLD`` but triggers on the
        no-progress failure mode (screen never changes) rather than
        the wrong-progress failure mode.
        """

        if plan.rationale != CompletionReason.ACTION_BLOCKED.value:
            return None

        retries = self.__context.agent_state.sub_goal_planner_retries
        if retries < PLANNER_RETRY_ESCALATION_THRESHOLD:
            return None

        logger.info(
            "[NODE: ANALYZE] Planner retries hit escalation threshold "
            "(%d >= %d); invoking decomposer replan.",
            retries,
            PLANNER_RETRY_ESCALATION_THRESHOLD,
        )
        return await self.__replan_remaining_sub_goals(
            capture=capture,
            failure_reason=(
                f"Planner rejection-history retry fired {retries} times "
                f"on this sub-goal without unsticking the action loop. "
                f"The cheap local retry exhausted its budget."
            ),
        )

    async def ground(self, state: IntentGraphState) -> IntentGraphState:
        """
        Capture the screen and update state.

        ERROR BOUNDARY: All exceptions are caught and converted to terminal states
        to ensure graph execution completes gracefully even on device failures.
        """

        logger.info("[NODE: GROUND] Starting grounding node")
        logger.info(f"[NODE: GROUND] Current step count: {self.__context.agent_state.step_count}")
        logger.info(
            f"[NODE: GROUND] Incoming state has planned_step: {state.get(IntentStateKey.PLANNED_STEP) is not None}"
        )

        # ERROR BOUNDARY: Wrap entire node in try/except
        try:
            if await self.__is_cancelled():
                logger.warning("[NODE: GROUND] Execution cancelled")
                self.__context.agent_state.mark_complete(reason=CompletionReason.CANCELLED.value)

                return cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
                    },
                )

            # Check max steps BEFORE planning to avoid planning actions we can't execute
            if self.__context.agent_state.step_count >= self.__context.max_steps:
                logger.warning(
                    f"[NODE: GROUND] Max steps ({self.__context.max_steps}) reached. "
                    f"Current step count: {self.__context.agent_state.step_count}"
                )
                self.__context.agent_state.mark_complete(reason=CompletionReason.MAX_STEPS.value)

                return cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: CompletionReason.MAX_STEPS.value,
                    },
                )

            current_step_num = self.__context.agent_state.step_count + 1

            await self.__context.telemetry.info(
                f"Grounding step {current_step_num}...",
                type="STEP_STARTED",
                step=current_step_num,
            )

            start_time = time.time()

            # 1. Capture State (Screenshot + optional hierarchy)
            screen = await self.__context.perception.perceive(session_id=self.__context.workflow_id)

            if not screen.image:
                await self.__context.telemetry.error(
                    "Ground: Empty screenshot captured",
                    step=self.__context.agent_state.step_count + 1,
                )
                logger.error("[NODE: GROUND] Empty screenshot captured")
                self.__context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)
                return cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.CAPTURE: None,
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                    },
                )

            # 2. Capture Dimensions (Independent hardware metadata)
            width = screen.width
            height = screen.height
            logger.info(f"Device dimension is {height=}x{width=}")

            # Validate dimensions
            if width <= 0 or height <= 0:
                await self.__context.telemetry.error(
                    f"Ground: Invalid dimensions {width}x{height}",
                    step=self.__context.agent_state.step_count + 1,
                )
                logger.error(f"[NODE: GROUND] Invalid dimensions {width}x{height}")
                self.__context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)
                return cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.CAPTURE: None,
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                    },
                )

            activity = screen.activity

            raw_screen = screen
            # XML Dump if enabled
            xml = raw_screen.xml_content

            elements = None

            logger.debug(
                f"[DEBUG: GROUND] Config use_xml={self.__context.use_xml}, xml_content present={xml is not None}"
            )

            if self.__context.use_xml and xml:
                hierarchy_dump_duration = raw_screen.metadata.get("hierarchy_dump_duration")
                if isinstance(hierarchy_dump_duration, (int, float)):
                    self.__context.metrics.record(
                        operation="hierarchy_dump",
                        duration=float(hierarchy_dump_duration),
                    )

                process_start = time.time()
                hierarchy_result = await self.__context.hierarchy.process_xml_and_screen(
                    xml=xml,
                    session_id=self.__context.workflow_id,
                    package_name=activity,
                    path_manager=self.__context.path_manager,
                    action_type=ActionType.TAP,
                    screen=raw_screen,
                )
                self.__context.metrics.record(
                    operation="hierarchy_processing", duration=time.time() - process_start
                )

                elements = hierarchy_result.label_map
                if hierarchy_result.annotated_capture is not None:
                    screen = hierarchy_result.annotated_capture
            else:
                hierarchy_result = HierarchyProcessingResult()

            capture_hashes = self.__resolve_capture_hashes(
                capture=raw_screen,
                elements=hierarchy_result.labeled_elements,
            )

            screen_state = self.__build_screen_state(
                capture=raw_screen,
                visual_hash=capture_hashes.visual_hash,
                xml_hash=capture_hashes.xml_hash,
                interaction_hash=capture_hashes.interaction_hash,
            )
            screen = screen.model_copy(update={"state": screen_state})

            is_new_screen = self.__context.agent_state.update_screen(screen=screen_state)

            duration = time.time() - start_time
            self.__context.metrics.record(operation="screenshot", duration=duration)

            logger.info(
                f"[NODE: GROUND] Screen captured: hash={capture_hashes.visual_hash}, activity={activity}, is_new={is_new_screen}, elements={len(elements) if elements else 0}"
            )
            logger.info(f"[NODE: GROUND] Grounding completed in {duration:.2f}s")
            logger.info("[NODE: GROUND] -> Transitioning to ANALYZE")

            # Reset per-step fields
            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.ANALYSIS: None,
                    CommonStateKey.CAPTURE: screen,
                    IntentStateKey.XML_CONTENT: xml,
                    CommonStateKey.STEP_RESULT: None,
                    IntentStateKey.ELEMENTS: elements,
                    IntentStateKey.PLANNED_STEP: None,
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.SCREEN_STATE: screen_state,
                    CommonStateKey.GROUNDING_DURATION: duration,
                    CommonStateKey.IS_NEW_SCREEN: is_new_screen,
                },
            )

            # Persist sub-goal state to graph for checkpoint recovery
            self.__persist_agent_state_to_graph(result=result)

            return result

        except Exception as exception:
            logger.exception(f"[NODE: GROUND] Grounding failed: {exception}")
            display_error = (
                exception.display(fallback=GROUNDING_FAILURE_MESSAGE)
                if isinstance(exception, FathomError)
                else GROUNDING_FAILURE_MESSAGE
            )
            await self.__context.telemetry.error(display_error)
            self.__context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)

            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.CAPTURE: None,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                },
            )

            # Persist failure state
            self.__persist_agent_state_to_graph(result=result)

            return result

    async def analyze(self, state: IntentGraphState) -> IntentGraphState:
        """
        Plan the next step using the Agent Planner.

        ERROR BOUNDARY: Wraps planning in try/except to handle LLM/network failures gracefully.
        """

        logger.info("[NODE: ANALYZE] Starting analysis node")

        # Restore agent_state from graph checkpoint if available
        self.__restore_agent_state_from_graph(state=state)

        if await self.__is_cancelled():
            logger.warning("[NODE: ANALYZE] Execution cancelled")
            self.__context.agent_state.mark_complete(reason=CompletionReason.CANCELLED.value)

            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
                },
            )
            self.__persist_agent_state_to_graph(result=result)
            return result

        # Use type guard to satisfy MyPy
        screen_capture = state.get(CommonStateKey.CAPTURE)

        if not screen_capture or not isinstance(screen_capture, ScreenCapture):
            logger.error("[NODE: ANALYZE] No valid screen capture found, setting should_retry=True")
            return cast("IntentGraphState", {IntentStateKey.SHOULD_RETRY: True})

        capture: ScreenCapture = screen_capture

        # ERROR BOUNDARY: Wrap planning logic
        try:
            # Check injected context
            current_step = self.__context.agent_state.step_count
            state_injected = state.get(IntentStateKey.INJECTED_CONTEXT)
            guidance_snapshot = self.__context.context_manager.get_user_guidance()
            logger.debug(
                f"[H3] Analysis Context | Step: {current_step} | "
                f"Active Guidance: {len(guidance_snapshot)} items | "
                f"State Injected: {state_injected is not None}"
            )

            start_time = time.time()
            raw_elements = state.get(IntentStateKey.ELEMENTS)

            elements: Optional[Dict[str, Any]] = None
            if isinstance(raw_elements, dict):
                elements = raw_elements

            # Use dimensions from the screenshot itself (PNG IHDR) so
            # coordinates stay consistent with what the LLM sees, even
            # after the device rotates mid-session.
            width, height = capture.width, capture.height

            # Determine interactive mode & config for planner
            is_interactive = self.__context.signal.supports_interruption()
            prompt_if_stuck = self.__context.configuration.intent.prompt_user_if_stuck

            # HITL: Check for pause request or context injection before planning
            await self.__handle_hitl(current_step=current_step)

            logger.info(f"[NODE: ANALYZE] Calling planner for step {current_step + 1}")
            plan = await self.__context.planner.plan_step(
                capture=capture,
                elements=elements,
                screen_width=width,
                screen_height=height,
                use_xml=self.__context.use_xml,
                interactive_mode=is_interactive,
                prompt_if_stuck=prompt_if_stuck,
                state=self.__context.agent_state,
                reasoner=self.__context.reasoner,
                context_manager=self.__context.context_manager,
            )

            duration = time.time() - start_time
            self.__context.metrics.record(operation="analysis", duration=duration)

            if plan.metrics:
                self.__context.metrics.record_tokens(
                    prompt=int(plan.metrics.get("prompt_tokens", 0)),
                    cached=int(plan.metrics.get("cached_tokens", 0)),
                    completion=int(plan.metrics.get("completion_tokens", 0)),
                )

            # Log plan details
            if plan.step:
                logger.info(
                    f"[NODE: ANALYZE] Plan created: action={plan.step.action.action_type.value}, "
                    f"confidence={plan.step.action.confidence:.2f}, "
                    f"target={plan.step.action.target}"
                )

                # Emit structured telemetry for streaming UI
                await self.__context.telemetry.info(
                    plan.rationale or "No reasoning",
                    type=FathomEvent.REASONING,
                    step=current_step + 1,
                    reasoning=plan.rationale,
                    rationale=plan.step.action.rationale if plan.step else None,
                )
                await self.__context.telemetry.info(
                    plan.step.action.to_description(),
                    type=FathomEvent.PLANNED_ACTION,
                    step=current_step + 1,
                )
            else:
                logger.warning(
                    f"[NODE: ANALYZE] No step planned: is_complete={plan.is_complete}, "
                    f"should_retry={plan.should_retry}, rationale={plan.rationale}"
                )

            # Handle ACTION_BLOCKED signal from planner: inject guidance into
            # the context manager so the next GROUND→ANALYZE cycle sees it as
            # a <SYSTEM_OVERRIDE>. This keeps guidance mutation in the graph node
            # rather than the planner (SRP).
            blocked_action = (plan.metadata or {}).get("blocked_action")
            if plan.rationale == CompletionReason.ACTION_BLOCKED.value and blocked_action:
                await self.__context.context_manager.inject_user_guidance(
                    guidance=(
                        f"ACTION BLOCKED: '{blocked_action}' has been repeated 3+ times on this "
                        "screen without progress and is now BLOCKED. You MUST use a completely "
                        "different action or interaction path to achieve the same goal. "
                        "Consider alternative UI elements, different navigation paths, or "
                        "scrolling to reveal new elements."
                    ),
                    step=current_step,
                )

            # Replanning escalation: if the cheap rejection-history retry has
            # fired PLANNER_RETRY_ESCALATION_THRESHOLD times on this sub-goal
            # without breaking the action loop, the local retry budget is
            # exhausted. Escalate to the expensive decomposer replan so the
            # remaining sub-goal tail gets rewritten from the stuck screen.
            # Mirror of the VERIFY-side REDECOMPOSE_VERIFY_FAILURE_THRESHOLD
            # escalation, but triggered on no-progress instead of wrong-progress.
            replanned = await self.__maybe_escalate_on_planner_retries(
                plan=plan,
                capture=capture,
            )
            if replanned is not None:
                return replanned
            # Replan returned None (no escalation or decomposer failure). Fall
            # through to normal retry routing so the graph keeps running.

            logger.info(
                f"[NODE: ANALYZE] Analysis completed in {duration:.2f}s: "
                f"is_complete={plan.is_complete}, should_retry={plan.should_retry}, "
                f"has_step={plan.step is not None}"
            )

            completion_reason = (
                plan.rationale if plan.is_complete else state.get(CommonStateKey.COMPLETION_REASON)
            )
            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.PLAN: plan,
                    IntentStateKey.ELEMENTS: elements,
                    IntentStateKey.INJECTED_CONTEXT: None,
                    IntentStateKey.PLANNED_STEP: plan.step,
                    CommonStateKey.ANALYSIS_DURATION: duration,
                    CommonStateKey.IS_COMPLETE: plan.is_complete,
                    IntentStateKey.SHOULD_RETRY: plan.should_retry,
                    CommonStateKey.COMPLETION_REASON: completion_reason,
                },
            )

            # Log what will happen next based on routing logic
            if plan.is_complete:
                logger.info("[NODE: ANALYZE] -> Will route to VERIFY (is_complete=True)")

            elif plan.should_retry:
                logger.info("[NODE: ANALYZE] -> Will route to GROUND (should_retry=True)")

            elif not plan.step:
                logger.info("[NODE: ANALYZE] -> Will route to GROUND (no planned_step)")

            else:
                logger.info("[NODE: ANALYZE] -> Will route to EXECUTE")

            # Persist sub-goal state to graph for checkpoint recovery
            self.__persist_agent_state_to_graph(result=result)

            return result

        except Exception as exception:
            logger.exception(f"[NODE: ANALYZE] Analysis failed: {exception}")
            await self.__context.telemetry.error(
                f"Analysis failed: {exception}",
                step=self.__context.agent_state.step_count + 1,
            )
            # Return retry state to attempt recovery
            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.SHOULD_RETRY: True,
                    CommonStateKey.ANALYSIS_DURATION: 0.0,
                    IntentStateKey.INJECTED_CONTEXT: None,
                },
            )
            self.__persist_agent_state_to_graph(result=result)
            return result

    async def execute(self, state: IntentGraphState) -> IntentGraphState:
        """
        Execute the planned action via ActionExecutor.

        ERROR BOUNDARY: Wraps execution in try/except to handle device/action failures gracefully.
        """

        logger.info("[NODE: EXECUTE] Starting execution node")

        if await self.__is_cancelled():
            logger.warning("[NODE: EXECUTE] Execution cancelled")
            self.__context.agent_state.mark_complete(reason=CompletionReason.CANCELLED.value)

            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
                },
            )
            self.__persist_agent_state_to_graph(result=result)
            return result

        # Type guards for planned_step and capture
        screen_capture = state.get(CommonStateKey.CAPTURE)
        current_step = state.get(IntentStateKey.PLANNED_STEP)

        if not isinstance(current_step, Step) or not isinstance(screen_capture, ScreenCapture):
            logger.error(
                f"[NODE: EXECUTE] Invalid state: has_step={isinstance(current_step, Step)}, "
                f"has_capture={isinstance(screen_capture, ScreenCapture)}"
            )
            return {}

        step: Step = current_step
        capture: ScreenCapture = screen_capture
        # Retrieve elements for Ground Truth resolution
        raw_elements = state.get(IntentStateKey.ELEMENTS)

        elements: Optional[Dict[str, Any]] = None
        if isinstance(raw_elements, dict):
            elements = raw_elements

        logger.info(
            f"[NODE: EXECUTE] Executing action: type={step.action.action_type.value}, "
            f"target={step.action.target}, confidence={step.action.confidence:.2f}"
        )

        start_time = time.time()

        # Resolve References & Snap to Label
        resolved_action = await self.__context.resolution.resolve(
            action=step.action, elements=elements
        )
        step = step.model_copy(update={"action": resolved_action})

        # Use a stable session package for trace/screenshot history paths.
        # Dynamic foreground activity changes (launcher -> target app) fragment artifacts
        # across multiple directories and makes traces appear to stop after step 0.
        current_screen = state.get(CommonStateKey.SCREEN_STATE)
        package_name = self.__context.package_name or "unknown"
        if package_name == "unknown" and isinstance(current_screen, ScreenState):
            package_name = current_screen.activity or "unknown"

        # Process memory updates (Side-effects from tool calls)
        if step.action.memory_updates:
            logger.info(f"[NODE: EXECUTE] Processing memory updates: {step.action.memory_updates}")
            for key, value in step.action.memory_updates.items():
                await self.__context.memory.set(key=key, value=str(value))

        if step.action.action_type == ActionType.ASK_USER:
            logger.info("[NODE: EXECUTE] Intercepting ASK_USER action for native HITL")
            question = step.action.text or "I need human assistance to proceed."
            current_step = self.__context.agent_state.step_count

            user_response = await self.__context.hitl.ask(
                prompt=question,
                step=current_step + 1,
            )

            await self.__context.context_manager.inject_user_guidance(
                guidance=user_response, step=current_step
            )

            # Atomic update of budget and loop detector for HITL
            self.__context.agent_state.record_hitl_intervention()
            logger.info(msg="[NODE: EXECUTE] HITL intervention recorded. Loop history reset.")

            from fathom.schemas.results import ExecutionResult

            execution_result = ExecutionResult(
                success=True, duration=int((time.time() - start_time) * 1000)
            )
        else:
            # Delegate to ActionExecutor
            logger.info(
                f"[NODE: EXECUTE] Calling action executor for {step.action.action_type.value}"
            )
            execution_result = await self.__context.action_executor.act(
                step=step,
                pre_capture=capture,
                package_name=package_name,
                session_id=self.__context.workflow_id,
            )

        logger.info(
            f"[NODE: EXECUTE] Action executed: success={execution_result.success}, "
            f"duration={execution_result.duration}ms, error={execution_result.error}"
        )

        # Wait for screen stability after action with a hard upper bound.
        await stability_wait(self.__context.configuration)

        # Capture post-execution screen to compute post_hash
        current_screen_state = state.get(CommonStateKey.SCREEN_STATE)
        pre_hash = (
            current_screen_state.visual_hash
            if isinstance(current_screen_state, ScreenState)
            else ZERO_HASH
        )

        screen_diff: Optional[ScreenDiff] = None
        post_activity = package_name

        try:
            current_state = (
                current_screen_state if isinstance(current_screen_state, ScreenState) else None
            )
            post_action_comparison = await self.__capture_post_action_screen(
                before_capture=capture,
                before_state=current_state,
            )
            post_hash = post_action_comparison.post_visual_hash
            screen_diff = post_action_comparison.screen_diff

            # Capture post-action package for RECORD node (avoids extra device call)
            try:
                post_activity = await self.__context.device.get_current_package() or package_name
            except Exception:
                post_activity = package_name

            if post_hash is None:
                post_hash = pre_hash

        except Exception as exception:
            await self.__context.telemetry.warning(
                f"Execute: Failed to capture post-screen: {exception}"
            )
            post_hash = pre_hash  # Explicitly prevent 'screen changed' on error

        duration = time.time() - start_time
        self.__context.metrics.record(operation="action", duration=duration)

        if screen_diff is not None:
            screen_changed = screen_diff.action_had_effect
            ssim_str = (
                f"{screen_diff.ssim_score:.4f}" if screen_diff.ssim_score is not None else "N/A"
            )
            pixel_diff_str = (
                f"{screen_diff.content_pixel_diff_ratio:.4f}"
                if screen_diff.content_pixel_diff_ratio is not None
                else "N/A"
            )
            logger.info(
                f"[NODE: EXECUTE] ScreenDiff: phash={screen_diff.phash_distance}, "
                f"ssim={ssim_str}, content_diff={pixel_diff_str}, "
                f"regions={len(screen_diff.changed_regions)}, "
                f"scroll={screen_diff.scroll_translation}, "
                f"action_had_effect={screen_diff.action_had_effect}"
            )
        else:
            screen_changed = (
                ScreenState.hamming_distance(
                    left_hash=pre_hash,
                    right_hash=post_hash,
                )
                > ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD
            )

        logger.info(
            f"[NODE: EXECUTE] Execution completed in {duration:.2f}s: "
            f"pre_hash={pre_hash[:8]}, post_hash={post_hash[:8]}, screen_changed={screen_changed}"
        )
        logger.info("[NODE: EXECUTE] -> Transitioning to RECORD")

        # Extract observation from plan
        execution_plan = state.get(IntentStateKey.PLAN)

        if isinstance(execution_plan, PlanResult):
            observation = execution_plan.metadata.get("observation")
        else:
            observation = None

        step_result = StepResult(
            step=step,
            pre_hash=pre_hash,
            post_hash=post_hash,
            observation=observation,
            duration=int(duration * 1000),
            error=execution_result.error,
            screen_changed=screen_changed,
            success=execution_result.success,
            generalized_target=step.action.script_target,
            is_positional=(step.action.target_type == "positional"),
        )

        result_dict: Dict[Any, Any] = {
            CommonStateKey.STEP_RESULT: step_result,
            CommonStateKey.EXECUTION_DURATION: duration,
            IntentStateKey.ELEMENTS: state.get(IntentStateKey.ELEMENTS),
            IntentStateKey.POST_ACTIVITY: post_activity,
        }

        # If ASK_USER was executed, always clear state to force re-planning
        # ASK_USER is triggered when agent is stuck/uncertain, so fresh start is always needed
        if step.action.action_type == ActionType.ASK_USER:
            logger.info("[NODE: EXECUTE] Clearing graph state after ASK_USER for fresh analysis")
            result_dict[IntentStateKey.PLAN] = None
            result_dict[CommonStateKey.IS_COMPLETE] = False
            result_dict[IntentStateKey.PLANNED_STEP] = None
            result_dict[IntentStateKey.SHOULD_RETRY] = True
            result_dict[CommonStateKey.COMPLETION_REASON] = None

        # Persist sub-goal state to graph for checkpoint recovery
        self.__persist_agent_state_to_graph(result=result_dict)

        return result_dict  # type: ignore[return-value]

    async def record(self, state: IntentGraphState) -> IntentGraphState:
        """
        Record the execution result.

        ERROR BOUNDARY: Wraps recording in try/except to handle storage/telemetry failures gracefully.
        """

        logger.info("[NODE: RECORD] Starting record node")

        if await self.__is_cancelled():
            logger.warning("[NODE: RECORD] Execution cancelled")
            self.__context.agent_state.mark_complete(reason=CompletionReason.CANCELLED.value)

            return cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
                },
            )

        result = state.get(CommonStateKey.STEP_RESULT)
        if not isinstance(result, StepResult):
            logger.error("[NODE: RECORD] No valid step result found")
            return {}

        step_result: StepResult = result

        # ERROR BOUNDARY: Wrap recording logic
        try:
            # Record in agent state (internal bookkeeping, always done)
            self.__context.agent_state.record_step(result=step_result)

            # Accumulate step results in graph state so MemorySaver checkpoints them.
            existing_step_results = cast(
                "List[StepResult]", state.get(IntentStateKey.STEP_RESULTS) or []
            )
            accumulated_step_results = existing_step_results + [step_result]

            # Use post-action activity captured in EXECUTE node (avoids extra device call)
            post_activity_raw = state.get(IntentStateKey.POST_ACTIVITY)
            current_activity = str(post_activity_raw) if post_activity_raw else "unknown"
            execution_activity = "unknown"
            screen_state_value = state.get(CommonStateKey.SCREEN_STATE)
            if isinstance(screen_state_value, ScreenState):
                execution_activity = screen_state_value.activity or "unknown"

            logger.info(
                f"[NODE: RECORD] Recording step: success={step_result.success}, "
                f"screen_changed={step_result.screen_changed}, duration={step_result.duration}ms, "
                f"execution_package={execution_activity}, observed_package={current_activity}"
            )

            # LAUNCHER BLOCKING: Never persist actions taken on launcher apps
            execution_package_base = execution_activity.split("/")[0]
            observed_package_base = current_activity.split("/")[0]
            is_on_launcher = self.__should_skip_launcher_persistence(
                execution_activity=execution_activity,
                observed_activity=current_activity,
            )

            if is_on_launcher:
                logger.warning(
                    f"[NODE: RECORD] Skipping persistence: on launcher app. "
                    f"Launcher={execution_package_base}, "
                    f"Observed={observed_package_base}, "
                    f"step_num={step_result.step.step_number}, action_type={step_result.step.action.action_type.value}"
                )
                await self.__context.telemetry.warning(
                    f"Step {step_result.step.step_number} not persisted (on launcher)",
                    execution_package=execution_activity,
                    observed_package=current_activity,
                    step_number=step_result.step.step_number + 1,
                    action_type=step_result.step.action.action_type.value,
                )
            else:
                self.__enqueue_history_persistence(
                    step_result=step_result,
                    current_activity=current_activity,
                    execution_activity=execution_activity,
                )
                logger.debug(
                    f"[NODE: RECORD] Recording step to history. Observed={current_activity}"
                )

            # Emit enriched telemetry for the UI to render full step details
            record = step_result.to_record(activity=current_activity)

            analysis_duration_raw = state.get(CommonStateKey.ANALYSIS_DURATION) or 0.0
            grounding_duration_raw = state.get(CommonStateKey.GROUNDING_DURATION) or 0.0
            execution_duration_raw = state.get(CommonStateKey.EXECUTION_DURATION) or 0.0

            analysis_duration = (
                float(analysis_duration_raw)
                if isinstance(analysis_duration_raw, (int, float, str))
                else 0.0
            )
            grounding_duration = (
                float(grounding_duration_raw)
                if isinstance(grounding_duration_raw, (int, float, str))
                else 0.0
            )
            execution_duration = (
                float(execution_duration_raw)
                if isinstance(execution_duration_raw, (int, float, str))
                else 0.0
            )

            total_duration = int(
                (grounding_duration + analysis_duration + execution_duration) * 1000
            )
            plan_metrics: Dict[str, Any] = {}
            plan_raw = state.get(IntentStateKey.PLAN)
            if isinstance(plan_raw, PlanResult):
                plan_metrics = dict(plan_raw.metrics or {})

            await self.__context.telemetry.info(
                f"Step {step_result.step.step_number} completed",
                type=FathomEvent.STEP_COMPLETED,
                success=record.success,
                duration=total_duration,
                rationale=record.rationale,
                observation=record.observation,
                action_type=record.action_type,
                step=step_result.step.step_number + 1,
                action_description=record.action_description,
                target=record.natural_language_target or record.target,
                analysis_llm_ms=float(plan_metrics.get("llm_analysis_ms", 0.0) or 0.0),
                analysis_parse_ms=float(plan_metrics.get("parse_ms", 0.0) or 0.0),
                analysis_payload_ms=float(plan_metrics.get("payload_ms", 0.0) or 0.0),
                analysis_manifest_ms=float(plan_metrics.get("manifest_ms", 0.0) or 0.0),
                analysis_tool_scope_ms=float(plan_metrics.get("tool_scope_ms", 0.0) or 0.0),
                analysis_total_ms=float(plan_metrics.get("analyze_ms", 0.0) or 0.0),
            )

            # Console audit logging for rich step visualization
            current_screen = state.get(CommonStateKey.SCREEN_STATE)
            is_new_screen_state = state.get(CommonStateKey.IS_NEW_SCREEN)
            execution_plan = state.get(IntentStateKey.PLAN)

            if (
                isinstance(execution_plan, PlanResult)
                and isinstance(current_screen, ScreenState)
                and isinstance(is_new_screen_state, bool)
            ):
                self.__context.auditor.log_step(
                    plan=execution_plan,
                    state=current_screen,
                    hierarchy_duration=0.0,
                    is_new_screen=is_new_screen_state,
                    result=step_result.to_record(),
                    analysis_duration=analysis_duration,
                    execution_duration=execution_duration,
                    grounding_duration=grounding_duration,
                    is_stuck=self.__context.agent_state.is_stuck,
                    step_count=self.__context.agent_state.step_count,
                    total_duration=grounding_duration + analysis_duration + execution_duration,
                )

            # SCRIPT_GENERATED is emitted only when the run completes (intent strategy),
            # not on every step, to avoid sending stale script content to the client.

            await self.__context.memory.store_experience(
                success=step_result.success,
                action=step_result.step.action,
                visual_hash=step_result.pre_hash,
            )

            logger.debug(
                f"[H3] Committing to trace | thought={step_result.step.action.rationale[:50]}..."
            )

            observation = f"Screen: {step_result.pre_hash[:8]}"

            await self.__context.context_manager.commit(
                observation=observation,
                action=step_result.step.action,
                thought=step_result.step.action.rationale,
            )

            full_context = self.__context.context_manager.get_full_context()
            active_count = full_context.get("active_count", 0)

            if active_count >= GCC_BRANCHING_THRESHOLD:
                logger.info(f"[NODE: RECORD] Triggering GCC branch: active_count={active_count}")
                await self.__context.context_manager.branch()

            execution_plan = state.get(IntentStateKey.PLAN)

            if isinstance(execution_plan, PlanResult) and execution_plan.is_complete:
                logger.info("[NODE: RECORD] Plan indicates completion. This is the final step.")
                self.__context.agent_state.mark_complete(
                    reason=execution_plan.rationale or "Completed"
                )

                result = cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: execution_plan.rationale,
                        IntentStateKey.STEP_RESULTS: accumulated_step_results,
                    },
                )
                self.__persist_agent_state_to_graph(result=result)
                return result

            # ── Sub-goal completion check (post-execution) ──
            # Evaluated here — after the action has executed and been recorded —
            # so we never advance a sub-goal on an action that didn't run.
            subgoal_result = await self.__evaluate_subgoal_completion(
                plan=execution_plan,
                step_result=step_result,
                accumulated_step_results=accumulated_step_results,
            )
            if subgoal_result is not None:
                self.__persist_agent_state_to_graph(result=subgoal_result)
                return subgoal_result

            logger.info(
                f"[NODE: RECORD] Step {self.__context.agent_state.step_count} recorded successfully"
            )
            logger.info("[NODE: RECORD] -> Will route to GROUND for next step")

            result = cast(
                "IntentGraphState",
                {IntentStateKey.STEP_RESULTS: accumulated_step_results},
            )
            self.__persist_agent_state_to_graph(result=result)
            return result

        except Exception as exception:
            logger.exception(f"[NODE: RECORD] Recording failed: {exception}")
            display_error = (
                exception.display(fallback=RECORDING_FAILURE_MESSAGE)
                if isinstance(exception, FathomError)
                else RECORDING_FAILURE_MESSAGE
            )
            await self.__context.telemetry.error(
                display_error,
                step=self.__context.agent_state.step_count,
            )
            existing_step_results = cast(
                "List[StepResult]", state.get(IntentStateKey.STEP_RESULTS) or []
            )
            result = cast("IntentGraphState", {IntentStateKey.STEP_RESULTS: existing_step_results})
            self.__persist_agent_state_to_graph(result=result)
            return result

    async def __evaluate_subgoal_completion(
        self,
        *,
        plan: Any,
        step_result: StepResult,
        accumulated_step_results: List[StepResult],
    ) -> Optional[IntentGraphState]:
        """
        Evaluate sub-goal completion after action execution.

        Returns an IntentGraphState if the sub-goal completed (or all sub-goals
        finished), otherwise None to continue normal flow.
        """

        agent_state = self.__context.agent_state
        current_sub_goal = agent_state.get_current_sub_goal()
        if not current_sub_goal or not agent_state.has_sub_goals():
            return None

        # Extract the analysis from the plan metadata (set by planner).
        # After LangGraph checkpoint serialization the AnalysisResult may
        # be deserialized as a plain dict — reconstruct it when needed.
        raw_analysis = None
        if isinstance(plan, PlanResult) and plan.metadata:
            raw_analysis = plan.metadata.get("_analysis")

        if raw_analysis is None:
            # Cheap "action-loop" recovery is handled by the planner via
            # is_action_repeating_on_screen + rejection history. The
            # expensive redecompose path is driven exclusively by the
            # verification-failure counter (see VERIFY node below).
            return None

        analysis = (
            raw_analysis
            if isinstance(raw_analysis, AnalysisResult)
            else AnalysisResult.model_validate(raw_analysis)
        )

        # Validation-type steps don't require a screen change — observing
        # the screen IS the goal.  Detected from the step's event_type
        # (set by the LLM tool schema) OR from sub-goal description keywords.
        is_validation_step = step_result.step.event_type == "validation" or any(
            keyword in current_sub_goal.description.lower()
            for keyword in ["validate", "verify", "confirm", "check if", "check that"]
        )

        sub_goal_signal = self.__context.reasoner.analyze_subgoal_completion(
            analysis=analysis,
            sub_goal_description=current_sub_goal.description,
            screen_description=step_result.observation or step_result.step.action.target or "",
            screen_changed=step_result.screen_changed or is_validation_step,
            pre_screen_hash=step_result.pre_hash,
            post_screen_hash=step_result.post_hash,
        )

        # Two-signal policy: llm_signaled + effective_action (action_executed
        # AND screen_verified).  Rationale verification is disabled — the
        # VERIFY node with its screenshot check is the final completion gate.
        required_threshold = 2
        signal_count = sub_goal_signal.count_signals()
        current_idx, total = agent_state.get_sub_goal_progress()

        logger.info(
            f"[NODE: RECORD] Sub-goal completion check: '{current_sub_goal.description[:50]}...' | "
            f"signals={signal_count}/{required_threshold} | "
            f"llm={sub_goal_signal.llm_signaled} | "
            f"rationale={sub_goal_signal.rationale_verified} | "
            f"action={sub_goal_signal.action_executed} | "
            f"screen_verified={sub_goal_signal.screen_verified} | "
            f"evidence: {sub_goal_signal.evidence}"
        )

        if not sub_goal_signal.meets_threshold(required_signals=required_threshold):
            logger.warning(
                f"[NODE: RECORD] Sub-goal {current_sub_goal.index} NOT completing yet: "
                f"{signal_count}/{required_threshold} signals | "
                f"Progress: [{current_idx + 1}/{total}] | "
                f"Type: {'validation' if is_validation_step else 'action'} | "
                f"Actions on sub-goal: {agent_state.sub_goal_action_count}"
            )

            # Cheap "action-loop" recovery is the planner's 3-repeated-
            # action detector; expensive redecompose is driven by verify
            # failures in the VERIFY node. Neither fires from here.
            return None

        # Route to VERIFY for per-sub-goal screenshot verification.
        # Sub-goal advancement is deferred — VERIFY will capture a screenshot
        # and confirm the sub-goal is actually done before advancing.
        logger.info(
            f"[NODE: RECORD] Sub-goal {current_sub_goal.index} signals passed "
            f"({signal_count}/{required_threshold}). Routing to VERIFY for "
            f"screenshot confirmation before advancing."
        )
        agent_state.mark_complete(
            reason=f"Sub-goal '{current_sub_goal.description[:50]}' pending verification"
        )
        return cast(
            "IntentGraphState",
            {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: (
                    f"Sub-goal '{current_sub_goal.description[:50]}' pending verification"
                ),
                IntentStateKey.STEP_RESULTS: accumulated_step_results,
            },
        )

    async def __advance_after_verified(
        self,
        *,
        current_sub_goal: Optional[SubGoal],
        is_subgoal_verify: bool,
        reason: str,
    ) -> IntentGraphState:
        """
        Apply state transitions after the verifier confirms completion.

        When a sub-goal was under verification, mark it complete, emit the
        ``SUB_GOAL_COMPLETED`` cue, and either emit ``SUB_GOAL_STARTED`` for
        the next sub-goal (returning a retry state) or mark the whole intent
        complete (returning a terminal state).  When the full intent was
        under verification, mark the intent complete with ``reason``.
        """

        agent_state = self.__context.agent_state

        if is_subgoal_verify and current_sub_goal is not None:
            has_more = agent_state.mark_current_sub_goal_complete(
                completion_signal=SubGoalCompletionSignal(
                    evidence=f"Verified by screenshot: {reason}",
                    llm_signaled=True,
                    rationale_verified=False,
                    action_executed=True,
                    screen_verified=True,
                ),
            )
            agent_state.reset_completion()

            # Clear stale verification feedback from previous sub-goals
            # so it doesn't pollute the next sub-goal's context.
            self.__context.context_manager.clear_user_guidance()

            total_sub_goals = len(agent_state.sub_goal_list)

            # Cue: the just-finished sub-goal.
            await self.__context.telemetry.info(
                "Sub-goal completed",
                type=FathomEvent.SUB_GOAL_COMPLETED,
                index=current_sub_goal.index,
                total=total_sub_goals,
                description=current_sub_goal.description,
            )

            if has_more:
                next_sg = agent_state.get_current_sub_goal()
                logger.info(
                    f"[NODE: VERIFY] ✓ Sub-goal {current_sub_goal.index} VERIFIED. "
                    f"Advancing to sub-goal {next_sg.index if next_sg else '(none)'}: "
                    f"'{next_sg.description if next_sg else ''}'"
                )

                # Cue: the newly-active sub-goal.
                if next_sg is not None:
                    await self.__context.telemetry.info(
                        "Sub-goal started",
                        type=FathomEvent.SUB_GOAL_STARTED,
                        index=next_sg.index,
                        total=total_sub_goals,
                        description=next_sg.description,
                    )
                result = cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.IS_COMPLETE: False,
                        IntentStateKey.SHOULD_RETRY: True,
                    },
                )
                self.__persist_agent_state_to_graph(result=result)
                return result

            # Last sub-goal verified — mark intent complete.
            logger.info("[NODE: VERIFY] All sub-goals verified. Marking intent complete.")
            agent_state.mark_complete(
                reason="All sub-goals completed and verified sequentially"
            )
            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: (
                        "All sub-goals completed and verified sequentially"
                    ),
                },
            )
            self.__persist_agent_state_to_graph(result=result)
            return result

        # Full intent verified — exit.
        agent_state.mark_complete(reason=reason)
        result = cast(
            "IntentGraphState",
            {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: reason,
            },
        )
        self.__persist_agent_state_to_graph(result=result)
        return result

    async def verify(self, state: IntentGraphState) -> IntentGraphState:
        """
        Explicitly verify if the intent is truly complete by capturing the screen and asking the LLM.
        If verification fails, it adds negative feedback and routes back to the main loop.
        """

        logger.info("[NODE: VERIFY] Starting verification phase")

        if await self.__is_cancelled():
            logger.warning("[NODE: VERIFY] Execution cancelled")
            self.__context.agent_state.mark_complete(reason=CompletionReason.CANCELLED.value)

            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
                },
            )
            self.__persist_agent_state_to_graph(result=result)
            return result

        start_time = time.time()

        # 1. Capture the latest screen state
        try:
            capture = await self.__context.perception_port.capture()
            image_bytes = capture.image
            if not image_bytes:
                logger.warning("[NODE: VERIFY] Failed to capture screen for verification")
                self.__context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)

                return cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                    },
                )
        except Exception as exception:
            logger.error(f"[NODE: VERIFY] Screen capture failed: {exception}")
            self.__context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)

            return cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                },
            )

        # 2. Construct binary validation prompt.
        # When sub-goals are active and the current sub-goal is pending
        # verification, verify the sub-goal — not the full intent.
        agent_state = self.__context.agent_state
        current_sub_goal = agent_state.get_current_sub_goal()
        is_subgoal_verify = (
            current_sub_goal is not None
            and agent_state.has_sub_goals()
            and not agent_state.all_sub_goals_complete()
        )

        if is_subgoal_verify and current_sub_goal is not None:
            verify_target = current_sub_goal.description
            logger.info(
                f"[NODE: VERIFY] Verifying sub-goal {current_sub_goal.index}: "
                f"'{verify_target[:60]}...'"
            )
        else:
            verify_target = self.__context.intent

        user_guidance_lines = [
            guidance.content for guidance in self.__context.context_manager.get_user_guidance()
        ]

        if is_subgoal_verify:
            system_prompt = SUBGOAL_VERIFICATION_SYSTEM
            recent_trace = self.__context.context_manager.get_full_context().get("trace", [])
            user_prompt = build_subgoal_verification_user_prompt(
                intent=verify_target,
                user_guidance=user_guidance_lines,
                recent_trace=recent_trace,
            )
        else:
            system_prompt = VERIFICATION_SYSTEM
            user_prompt = build_intent_verification_user_prompt(
                intent=verify_target,
                user_guidance=user_guidance_lines,
            )

        # 3. Ask the LLM with escalating thinking on retries
        is_truly_complete = False
        reason = "Verification failed without specific reason."
        next_action = ""
        thinking_levels = ["low", "medium", "high"]
        max_verify_retries = len(thinking_levels)

        for attempt in range(max_verify_retries):
            try:
                llm_result = await self.__context.llm.generate(
                    use_cache=False,
                    system_instruction=system_prompt,
                    prompt=[user_prompt, image_bytes],
                    thinking_level=thinking_levels[attempt],
                )

                text = strip_code_fences(llm_result.content)
                data = json.loads(text)
                is_truly_complete = bool(data.get("is_complete", False))
                reason = str(data.get("reason", "Verification failed without specific reason."))
                next_action = str(data.get("next_action", "")).strip()
                break

            except Exception as exception:
                logger.warning(
                    f"[NODE: VERIFY] Verification attempt {attempt + 1}/{max_verify_retries} "
                    f"failed (thinking={thinking_levels[attempt]}): {exception}"
                )
                if attempt == max_verify_retries - 1:
                    logger.error(f"[NODE: VERIFY] All verification attempts exhausted: {exception}")
                    reason = f"Verification failed due to error: {exception}"

        duration = time.time() - start_time
        logger.info(
            f"[NODE: VERIFY] Verification finished in {duration:.2f}s: "
            f"is_complete={is_truly_complete}, reason={reason}"
            f"{f', next_action={next_action}' if next_action else ''}"
        )

        if is_truly_complete:
            return await self.__advance_after_verify_passed(
                is_subgoal_verify=is_subgoal_verify,
                current_sub_goal=current_sub_goal,
                reason=reason,
            )
        else:
            agent_state.record_verify_failure()
            failure_count = agent_state.sub_goal_verify_failures

            # On the 3rd rejection for the same sub-goal, check if the
            # agent has actually progressed beyond this sub-goal's scope
            # by examining the execution trace.  If actions suggest the
            # sub-goal was completed and the agent moved on, force-advance.
            if (
                is_subgoal_verify
                and current_sub_goal is not None
                and failure_count >= REDECOMPOSE_VERIFY_FAILURE_THRESHOLD
            ):
                logger.info(
                    f"[NODE: VERIFY] {failure_count} rejections for sub-goal "
                    f"{current_sub_goal.index}. Redecomposing remaining intent with "
                    "current screen context."
                )
                replanned = await self.__replan_remaining_sub_goals(
                    capture=capture,
                    failure_reason=reason,
                    suggested_next_action=next_action or None,
                )
                if replanned is not None:
                    return replanned

            # Standard rejection: suggest next action and route back.
            feedback_parts = [f"Verification FAILED: {reason}."]
            if next_action:
                feedback_parts.append(f"SUGGESTED NEXT ACTION: {next_action}")
            else:
                feedback_parts.append("Take a concrete UI action to make progress toward the goal.")
            feedback = " ".join(feedback_parts)
            logger.warning(f"[NODE: VERIFY] {feedback}")

            # Reset the is_complete flag
            self.__context.agent_state.reset_completion()

            # Clear stale guidance before injecting new feedback so only
            # the latest rejection is visible — prevents accumulation of
            # old failure messages from prior sub-goals or retries.
            self.__context.context_manager.clear_user_guidance()

            # Inject into ContextManager so the LLM sees it next iteration
            await self.__context.context_manager.inject_user_guidance(
                guidance=feedback, step=self.__context.agent_state.step_count
            )

            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: False,
                    IntentStateKey.SHOULD_RETRY: True,
                    IntentStateKey.INJECTED_CONTEXT: feedback,
                },
            )
            self.__persist_agent_state_to_graph(result=result)
            return result


class IntentGraphFactory:
    """
    Factory for building the Intent Node functions.
    Wraps node logic to prevent standalone functions and cleaner imports.
    """

    @staticmethod
    def build(*, context: GraphContext) -> Dict[str, Callable[..., Any]]:
        """
        Builds the node functions for the intent graph.
        """

        provider = IntentNodeProvider(
            context=context,
            screen_comparator=context.comparator,
        )

        return {
            NodeName.GROUND: provider.ground,
            NodeName.ANALYZE: provider.analyze,
            NodeName.EXECUTE: provider.execute,
            NodeName.RECORD: provider.record,
            NodeName.VERIFY: provider.verify,
        }
