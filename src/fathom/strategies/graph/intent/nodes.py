from __future__ import annotations

# mypy: disable-error-code="misc"
import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union, cast

from fathom.adapters.signal.noop import NoopSignal
from fathom.constants import ActionType, FathomEvent
from fathom.constants.execution import (
    LAUNCHER_PACKAGES,
    MAX_STABILITY_WAIT_MS,
    VISUAL_HASH_LENGTH,
)
from fathom.constants.graph import NodeName
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.prompts.templates import VERIFICATION_SYSTEM, VERIFICATION_USER_TEMPLATE
from fathom.core.services.hitl import HITLService
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.state import IntentGraphState

logger = logging.getLogger(__name__)


class IntentNodeProvider:
    """
    Provides LangGraph nodes for intent execution.
    Encapsulates dependencies and shared private logic.
    """

    def __init__(self, context: GraphContext) -> None:
        """
        Initialize provider with shared context.
        """

        self.__context = context

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

    def __restore_agent_state_from_graph(self, state: IntentGraphState) -> None:
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

    def __persist_agent_state_to_graph(
        self, result: Union[IntentGraphState, Dict[str, Any]]
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

    async def ground(self, state: IntentGraphState) -> IntentGraphState:
        """
        Capture the screen and update state.

        ERROR BOUNDARY: All exceptions are caught and converted to terminal states
        to ensure graph execution completes gracefully even on device failures.
        """

        logger.info("=" * 80)
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

            # 1. Capture State (Screenshot + XML)
            screenshot_bytes, xml_content = await self.__context.device.get_snapshot()

            if not screenshot_bytes or len(screenshot_bytes) == 0:
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
            width, height = await self.__context.device.get_dimensions()
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

            # Get current package
            try:
                activity = await self.__context.device.get_current_package()
            except Exception as exception:
                await self.__context.telemetry.warning(
                    f"Ground: Failed to get current package: {exception}",
                    step=self.__context.agent_state.step_count + 1,
                )
                activity = "unknown"

            # Persist capture in background to avoid blocking
            asyncio.create_task(
                self.__context.storage.save(
                    data=screenshot_bytes,
                    metadata={
                        "type": "screenshots",
                        "category": "screenshot",
                        "timestamp": time.time(),
                        "package_name": activity,
                        "activity_name": activity,
                        "session_id": self.__context.workflow_id,
                        "filename": f"{int(time.time() * 1000)}__{activity}.png",
                    },
                )
            )
            storage_id = "pending_background_upload"

            screen = ScreenCapture(
                width=width,
                height=height,
                activity=activity,
                image=screenshot_bytes,
                timestamp=int(time.time() * 1000),
                metadata={"storage_id": storage_id},
            )

            # XML Dump if enabled
            xml: Optional[str] = None

            if isinstance(xml_content, bytes):
                xml = xml_content.decode("utf-8", errors="ignore")

            elif isinstance(xml_content, str):
                xml = xml_content

            elements = None

            logger.debug(
                f"[DEBUG: GROUND] Config use_xml={self.__context.use_xml}, xml_content present={xml is not None}"
            )

            if self.__context.use_xml and xml:
                dump_start = time.time()
                self.__context.metrics.record(
                    operation="hierarchy_dump", duration=time.time() - dump_start
                )

                process_start = time.time()
                (
                    annotated_screen,
                    elements,
                ) = await self.__context.hierarchy.process_xml_and_screen(
                    xml=xml,
                    screen=screen,
                    package_name=activity,
                    action_type=ActionType.TAP,
                    session_id=self.__context.workflow_id,
                    path_manager=self.__context.path_manager,
                )
                self.__context.metrics.record(
                    operation="hierarchy_processing", duration=time.time() - process_start
                )

                if annotated_screen:
                    screen = annotated_screen

            # Update Agent State with robust multi-layer hashing
            xml_hash = self.__context.perception.compute_xml_hash(capture=screen)
            visual_hash = self.__context.perception.compute_visual_hash(capture=screen)
            interaction_hash = self.__context.perception.compute_interaction_hash(elements=elements)

            screen_state = ScreenState(
                xml_hash=xml_hash,
                visual_hash=visual_hash,
                activity=screen.activity,
                timestamp=screen.timestamp,
                interaction_hash=interaction_hash,
                activity_hash=hashlib.md5(
                    screen.activity.encode(), usedforsecurity=False
                ).hexdigest()[:VISUAL_HASH_LENGTH],
                structural_hash="0" * VISUAL_HASH_LENGTH,
            )

            is_new_screen = self.__context.agent_state.update_screen(screen=screen_state)

            duration = time.time() - start_time
            self.__context.metrics.record(operation="screenshot", duration=duration)

            logger.info(
                f"[NODE: GROUND] Screen captured: hash={visual_hash}, activity={activity}, is_new={is_new_screen}, elements={len(elements) if elements else 0}"
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
            self.__persist_agent_state_to_graph(result)

            return result

        except Exception as exception:
            await self.__context.telemetry.error(f"Grounding failed: {exception}")
            logger.exception(f"[NODE: GROUND] Grounding failed: {exception}")
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
            self.__persist_agent_state_to_graph(result)

            return result

    async def analyze(self, state: IntentGraphState) -> IntentGraphState:
        """
        Plan the next step using the Agent Planner.

        ERROR BOUNDARY: Wraps planning in try/except to handle LLM/network failures gracefully.
        """

        logger.info("=" * 80)
        logger.info("[NODE: ANALYZE] Starting analysis node")

        # Restore agent_state from graph checkpoint if available
        self.__restore_agent_state_from_graph(state)

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
            self.__persist_agent_state_to_graph(result)
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

            # Get Device Dimensions for Accurate Normalization (Strict)
            width, height = await self.__context.device.get_dimensions()

            # Determine interactive mode & config for planner
            is_interactive = not isinstance(self.__context.signal, NoopSignal)
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
                    plan.reason or "No reasoning",
                    type=FathomEvent.REASONING,
                    step=current_step + 1,
                    reasoning=plan.reason,
                    rationale=plan.step.action.rationale if plan.step else None,
                )
                await self.__context.telemetry.info(
                    plan.step.action.to_description(),
                    type=FathomEvent.PLANNED_ACTION,
                    step=current_step + 1,
                )

                self.__context.ux.render_fallback(
                    reasoning=plan.reason or "No reasoning",
                    action=plan.step.action.to_description(),
                    step_number=self.__context.agent_state.step_count + 1,
                )
            else:
                logger.warning(
                    f"[NODE: ANALYZE] No step planned: is_complete={plan.is_complete}, "
                    f"should_retry={plan.should_retry}, reason={plan.reason}"
                )

            logger.info(
                f"[NODE: ANALYZE] Analysis completed in {duration:.2f}s: "
                f"is_complete={plan.is_complete}, should_retry={plan.should_retry}, "
                f"has_step={plan.step is not None}"
            )

            completion_reason = (
                plan.reason if plan.is_complete else state.get(CommonStateKey.COMPLETION_REASON)
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
            self.__persist_agent_state_to_graph(result)

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
            self.__persist_agent_state_to_graph(result)
            return result

    async def execute(self, state: IntentGraphState) -> IntentGraphState:
        """
        Execute the planned action via ActionExecutor.

        ERROR BOUNDARY: Wraps execution in try/except to handle device/action failures gracefully.
        """

        logger.info("=" * 80)
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
            self.__persist_agent_state_to_graph(result)
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
        requested_wait_s = float(self.__context.configuration.engine.stability_wait)
        requested_wait_ms = requested_wait_s * 1000.0
        applied_wait_ms = min(requested_wait_ms, MAX_STABILITY_WAIT_MS)
        stability_wait_s = applied_wait_ms / 1000.0
        logger.debug(
            "[WAIT] source=stability_wait requested=%.3fs applied=%.3fs",
            requested_wait_s,
            stability_wait_s,
        )
        await asyncio.sleep(delay=stability_wait_s)

        # Capture post-execution screen to compute post_hash
        current_screen_state = state.get(CommonStateKey.SCREEN_STATE)
        pre_hash = (
            current_screen_state.visual_hash
            if isinstance(current_screen_state, ScreenState)
            else "0"
        )

        try:
            post_screenshot = await self.__context.device.capture_screen()

            if post_screenshot:
                # Construct a temporary ScreenCapture for hashing, inheriting metadata from pre_capture
                temp_capture = ScreenCapture(
                    image=post_screenshot,
                    width=capture.width,
                    height=capture.height,
                    activity=package_name,
                    timestamp=int(time.time() * 1000),
                )
                post_hash = self.__context.perception.compute_visual_hash(capture=temp_capture)
            else:
                post_hash = pre_hash

        except Exception as exception:
            await self.__context.telemetry.warning(
                f"Execute: Failed to capture post-screen: {exception}"
            )
            post_hash = pre_hash  # Explicitly prevent 'screen changed' on error

        duration = time.time() - start_time
        self.__context.metrics.record(operation="action", duration=duration)

        screen_changed = pre_hash != post_hash

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
        self.__persist_agent_state_to_graph(result_dict)

        return result_dict  # type: ignore[return-value]

    async def record(self, state: IntentGraphState) -> IntentGraphState:
        """
        Record the execution result.

        ERROR BOUNDARY: Wraps recording in try/except to handle storage/telemetry failures gracefully.
        """

        logger.info("=" * 80)
        logger.info("[NODE: RECORD] Starting record node")

        if await self.__is_cancelled():
            logger.warning("[NODE: RECORD] Execution cancelled")
            self.__context.agent_state.mark_complete(reason=CompletionReason.CANCELLED.value)

            return {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
            }

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

            current_activity = "unknown"
            try:
                current_activity = await self.__context.device.get_current_package()
            except Exception:
                current_activity = "unknown"

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
            is_on_launcher = execution_package_base in LAUNCHER_PACKAGES

            if is_on_launcher:
                logger.warning(
                    f"[NODE: RECORD] Skipping persistence: on launcher app. "
                    f"Launcher={execution_package_base}, "
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
                logger.debug(
                    f"[NODE: RECORD] Recording step to history. Observed={current_activity}"
                )
                await self.__context.history.save_step(
                    result=step_result, intent=self.__context.intent, activity=current_activity
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

            analysis_result = state.get(CommonStateKey.ANALYSIS)
            analysis: Optional[AnalysisResult] = None
            if isinstance(analysis_result, AnalysisResult):
                analysis = analysis_result

            observation = f"Screen: {step_result.pre_hash[:8]}"
            if analysis and analysis.screen_description:
                observation += f" | Content: {analysis.screen_description[:100]}..."

            await self.__context.context_manager.commit(
                observation=observation,
                action=step_result.step.action,
                thought=step_result.step.action.rationale,
            )

            full_context = self.__context.context_manager.get_full_context()
            active_count = full_context.get("active_count", 0)

            BRANCHING_THRESHOLD = 15
            if active_count >= BRANCHING_THRESHOLD:
                logger.info(f"[NODE: RECORD] Triggering GCC branch: active_count={active_count}")
                await self.__context.context_manager.branch()

            execution_plan = state.get(IntentStateKey.PLAN)
            current_screen = state.get(CommonStateKey.SCREEN_STATE)
            is_new_screen = state.get(CommonStateKey.IS_NEW_SCREEN)

            if (
                isinstance(execution_plan, PlanResult)
                and isinstance(current_screen, ScreenState)
                and isinstance(is_new_screen, bool)
            ):
                self.__context.auditor.log_step(
                    plan=execution_plan,
                    state=current_screen,
                    hierarchy_duration=0.0,
                    is_new_screen=is_new_screen,
                    result=step_result.to_record(),
                    analysis_duration=analysis_duration,
                    execution_duration=execution_duration,
                    grounding_duration=grounding_duration,
                    is_stuck=self.__context.agent_state.is_stuck,
                    step_count=self.__context.agent_state.step_count,
                    total_duration=grounding_duration + analysis_duration + execution_duration,
                )

            if isinstance(execution_plan, PlanResult) and execution_plan.is_complete:
                logger.info("[NODE: RECORD] Plan indicates completion. This is the final step.")
                self.__context.agent_state.mark_complete(
                    reason=execution_plan.reason or "Completed"
                )

                result = cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: execution_plan.reason,
                        IntentStateKey.STEP_RESULTS: accumulated_step_results,
                    },
                )
                self.__persist_agent_state_to_graph(result)
                return result

            logger.info(
                f"[NODE: RECORD] Step {self.__context.agent_state.step_count} recorded successfully"
            )
            logger.info("[NODE: RECORD] -> Will route to GROUND for next step")

            result = cast(
                "IntentGraphState",
                {IntentStateKey.STEP_RESULTS: accumulated_step_results},
            )
            self.__persist_agent_state_to_graph(result)
            return result

        except Exception as exception:
            logger.exception(f"[NODE: RECORD] Recording failed: {exception}")
            await self.__context.telemetry.error(
                f"Recording failed: {exception}",
                step=self.__context.agent_state.step_count,
            )
            existing_step_results = cast(
                "List[StepResult]", state.get(IntentStateKey.STEP_RESULTS) or []
            )
            result = cast("IntentGraphState", {IntentStateKey.STEP_RESULTS: existing_step_results})
            self.__persist_agent_state_to_graph(result)
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
            self.__persist_agent_state_to_graph(result)
            return result

        start_time = time.time()

        # 1. Capture the latest screen state
        try:
            image_bytes = await self.__context.device.capture_screen()
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

        # 2. Construct binary validation prompt
        intent = self.__context.intent
        system_prompt = VERIFICATION_SYSTEM

        guidance_section = ""
        if user_guidance := self.__context.context_manager.get_user_guidance():
            guidance_text = "\n".join([f"- {guidance.content}" for guidance in user_guidance])
            guidance_section = f"\nUser Guidance:\n{guidance_text}\n"

        user_prompt = VERIFICATION_USER_TEMPLATE.format(
            intent=intent, guidance_section=guidance_section
        )

        # 3. Ask the LLM
        try:
            llm_result = await self.__context.llm.generate(
                use_cache=False,
                system_instruction=system_prompt,
                prompt=[user_prompt, image_bytes],
            )

            text = llm_result.content.strip()

            if text.startswith("```json"):
                text = text[7:-3]

            elif text.startswith("```"):
                text = text[3:-3]

            data = json.loads(text)
            is_truly_complete = bool(data.get("is_complete", False))
            reason = str(data.get("reason", "Verification failed without specific reason."))

        except Exception as exception:
            logger.error(f"[NODE: VERIFY] LLM verification failed: {exception}")
            is_truly_complete = False
            reason = f"Verification failed due to error: {exception}"

        duration = time.time() - start_time
        logger.info(
            f"[NODE: VERIFY] Verification finished in {duration:.2f}s: is_complete={is_truly_complete}, reason={reason}"
        )

        if is_truly_complete:
            self.__context.agent_state.mark_complete(reason=reason)
            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: reason,
                },
            )
            self.__persist_agent_state_to_graph(result)
            return result
        else:
            # Inject negative feedback to force the agent to continue
            feedback = f"Verification failed: {reason}"
            logger.warning(f"[NODE: VERIFY] {feedback}")

            # Reset the is_complete flag
            self.__context.agent_state.reset_completion()

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
            self.__persist_agent_state_to_graph(result)
            return result


class IntentGraphFactory:
    """
    Factory for building the Intent Node functions.
    Wraps node logic to prevent standalone functions and cleaner imports.
    """

    @staticmethod
    def build(context: GraphContext) -> Dict[str, Callable[..., Any]]:
        """
        Builds the node functions for the intent graph.
        """

        provider = IntentNodeProvider(context=context)

        return {
            NodeName.GROUND: provider.ground,
            NodeName.ANALYZE: provider.analyze,
            NodeName.EXECUTE: provider.execute,
            NodeName.RECORD: provider.record,
            NodeName.VERIFY: provider.verify,
        }
