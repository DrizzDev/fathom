from __future__ import annotations

# mypy: disable-error-code="misc"
import asyncio
import hashlib
import logging
import time
from typing import Any, Callable, Dict, Optional

from fathom.adapters.signal.noop import NoopSignal
from fathom.constants import ActionType
from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.constants.graph import NodeName
from fathom.constants.state import CommonStateKey, IntentStateKey
from fathom.schemas.results import ActionResult, AnalysisResult, PlanResult
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

    async def ground(self, state: IntentGraphState) -> IntentGraphState:
        """
        Capture the screen and update state.
        """

        logger.info("=" * 80)
        logger.info("[NODE: GROUND] Starting grounding node")
        logger.info(f"[NODE: GROUND] Current step count: {self.__context.agent_state.step_count}")
        logger.info(
            f"[NODE: GROUND] Incoming state has planned_step: {state.get(IntentStateKey.PLANNED_STEP) is not None}"
        )

        if self.__context.is_cancelled:
            logger.warning("[NODE: GROUND] Execution cancelled")
            return {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: "Cancelled",
            }

        start_time = time.time()

        try:
            # Parallel Snapshot (Screenshot + XML) via DevicePort
            screenshot_bytes, xml_content = await self.__context.device.get_snapshot()

            if not screenshot_bytes or len(screenshot_bytes) == 0:
                await self.__context.telemetry.error("Ground: Empty screenshot captured")
                logger.error("[NODE: GROUND] Empty screenshot captured")
                return {
                    CommonStateKey.CAPTURE: None,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: "Empty screenshot captured",
                }

            width, height = await self.__context.device.get_dimensions()

            # Validate dimensions
            if width <= 0 or height <= 0:
                await self.__context.telemetry.error(f"Ground: Invalid dimensions {width}x{height}")
                logger.error(f"[NODE: GROUND] Invalid dimensions {width}x{height}")
                return {
                    CommonStateKey.CAPTURE: None,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: f"Invalid dimensions: {width}x{height}",
                }

            # Get current package
            try:
                activity = await self.__context.device.get_current_package()
            except Exception as exception:
                await self.__context.telemetry.warning(
                    f"Ground: Failed to get current package: {exception}"
                )
                activity = "unknown"

            # Persist capture
            storage_id = await self.__context.storage.save(
                data=screenshot_bytes,
                metadata={
                    "type": "screenshots",
                    "timestamp": time.time(),
                    "package_name": activity,
                    "activity_name": activity,
                    "session_id": self.__context.workflow_id,
                },
            )

            screen = ScreenCapture(
                width=width,
                height=height,
                activity=activity,
                image=screenshot_bytes,
                timestamp=int(time.time() * 1000),
                metadata={"storage_id": storage_id},
            )

            # XML Dump if enabled
            xml_content_str = xml_content if xml_content else None
            elements = None

            if self.__context.use_xml and xml_content_str:
                dump_start = time.time()
                self.__context.metrics.record(
                    operation="hierarchy_dump", duration=time.time() - dump_start
                )

                process_start = time.time()
                (
                    annotated_screen,
                    elements,
                ) = await self.__context.hierarchy.process_xml_and_screen(
                    screen=screen,
                    xml=xml_content_str,
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

            # Update Agent State
            visual_hash = hashlib.sha256(screen.image).hexdigest()[:VISUAL_HASH_LENGTH]

            screen_state = ScreenState(
                visual_hash=visual_hash,
                activity=screen.activity,
                timestamp=screen.timestamp,
                activity_hash=hashlib.md5(
                    screen.activity.encode(), usedforsecurity=False
                ).hexdigest()[:VISUAL_HASH_LENGTH],
                structural_hash="0" * VISUAL_HASH_LENGTH,
            )

            is_new_screen = self.__context.agent_state.update_screen(screen=screen_state)

            duration = time.time() - start_time
            self.__context.metrics.record(operation="screenshot", duration=duration)

            logger.info(
                f"[NODE: GROUND] Screen captured: hash={visual_hash}, activity={activity}, is_new={is_new_screen}"
            )
            logger.info(f"[NODE: GROUND] Grounding completed in {duration:.2f}s")
            logger.info("[NODE: GROUND] -> Transitioning to ANALYZE")

            # Reset per-step fields
            return {
                CommonStateKey.ANALYSIS: None,
                CommonStateKey.CAPTURE: screen,
                CommonStateKey.STEP_RESULT: None,
                IntentStateKey.ELEMENTS: elements,
                IntentStateKey.PLANNED_STEP: None,
                IntentStateKey.SHOULD_RETRY: False,
                CommonStateKey.SCREEN_STATE: screen_state,
                CommonStateKey.GROUNDING_DURATION: duration,
                IntentStateKey.XML_CONTENT: xml_content_str,
                CommonStateKey.IS_NEW_SCREEN: is_new_screen,
            }

        except Exception as exception:
            await self.__context.telemetry.error(f"Grounding failed: {exception}")
            logger.exception(f"[NODE: GROUND] Grounding failed: {exception}")
            return {
                CommonStateKey.CAPTURE: None,
                CommonStateKey.COMPLETION_REASON: f"Grounding failed: {exception}",
                CommonStateKey.IS_COMPLETE: True,
            }

    async def analyze(self, state: IntentGraphState) -> IntentGraphState:
        """
        Plan the next step using the Agent Planner.
        """

        logger.info("=" * 80)
        logger.info("[NODE: ANALYZE] Starting analysis node")

        if self.__context.is_cancelled:
            logger.warning("[NODE: ANALYZE] Execution cancelled")
            return {CommonStateKey.IS_COMPLETE: True}

        # Use type guard to satisfy MyPy
        screen_capture = state.get(CommonStateKey.CAPTURE)

        if not screen_capture or not isinstance(screen_capture, ScreenCapture):
            logger.error("[NODE: ANALYZE] No valid screen capture found, setting should_retry=True")
            return {IntentStateKey.SHOULD_RETRY: True}

        capture: ScreenCapture = screen_capture

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

        # Determine interactive mode & config
        is_interactive = not isinstance(self.__context.signal, NoopSignal)
        prompt_if_stuck = self.__context.configuration.intent.prompt_user_if_stuck

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
            await self.__context.telemetry.info(plan.reason or "No reasoning", type="REASONING")
            await self.__context.telemetry.info(
                plan.step.action.to_description(), type="PLANNED_ACTION"
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

        result = {
            IntentStateKey.PLAN: plan,
            IntentStateKey.PLANNED_STEP: plan.step,
            CommonStateKey.IS_COMPLETE: plan.is_complete,
            CommonStateKey.COMPLETION_REASON: plan.reason
            if plan.is_complete
            else state.get(CommonStateKey.COMPLETION_REASON),
            IntentStateKey.SHOULD_RETRY: plan.should_retry,
            CommonStateKey.ANALYSIS_DURATION: duration,
            IntentStateKey.INJECTED_CONTEXT: None,
        }

        # Log what will happen next based on routing logic
        if plan.is_complete:
            logger.info("[NODE: ANALYZE] -> Will route to END (is_complete=True)")

        elif plan.should_retry:
            logger.info("[NODE: ANALYZE] -> Will route to GROUND (should_retry=True)")

        elif not plan.step:
            logger.info("[NODE: ANALYZE] -> Will route to GROUND (no planned_step)")

        else:
            logger.info("[NODE: ANALYZE] -> Will route to EXECUTE")

        return result  # type: ignore[return-value]

    async def execute(self, state: IntentGraphState) -> IntentGraphState:
        """
        Execute the planned action via ActionExecutor.
        """

        logger.info("=" * 80)
        logger.info("[NODE: EXECUTE] Starting execution node")

        if self.__context.is_cancelled:
            logger.warning("[NODE: EXECUTE] Execution cancelled")
            return {CommonStateKey.IS_COMPLETE: True}

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

        # Determine package name from state for tracing
        current_screen = state.get(CommonStateKey.SCREEN_STATE)

        if isinstance(current_screen, ScreenState) and current_screen.activity:
            package_name = current_screen.activity
        else:
            package_name = "unknown"

        # Process memory updates (Side-effects from tool calls)
        if step.action.memory_updates:
            logger.info(f"[NODE: EXECUTE] Processing memory updates: {step.action.memory_updates}")
            for key, value in step.action.memory_updates.items():
                await self.__context.memory.set(key=key, value=str(value))

        if step.action.action_type == ActionType.ASK_USER:
            logger.info("[NODE: EXECUTE] Intercepting ASK_USER action for native HITL")
            question = step.action.text or "I need human assistance to proceed."
            user_response = await self.__context.signal.ask(prompt=question)

            # Inject guidance to context manager
            await self.__context.context_manager.inject_user_guidance(
                guidance=user_response, step=self.__context.agent_state.step_count
            )

            # Clear stuck state so execution can resume with new guidance
            self.__context.agent_state.reset_stuck_state()

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

        # Wait for screen stability after action
        await asyncio.sleep(delay=self.__context.configuration.engine.stability_wait)

        # Capture post-execution screen to compute post_hash
        try:
            post_screenshot = await self.__context.device.capture_screen()
            post_hash = (
                hashlib.sha256(post_screenshot).hexdigest()[:VISUAL_HASH_LENGTH]
                if post_screenshot
                else "0"
            )
        except Exception as exception:
            await self.__context.telemetry.warning(
                f"Execute: Failed to capture post-screen: {exception}"
            )
            post_hash = "0"

        duration = time.time() - start_time
        self.__context.metrics.record(operation="action", duration=duration)

        current_screen_state = state.get(CommonStateKey.SCREEN_STATE)
        pre_hash = (
            current_screen_state.visual_hash
            if isinstance(current_screen_state, ScreenState)
            else "0"
        )

        screen_changed = pre_hash != post_hash

        logger.info(
            f"[NODE: EXECUTE] Execution completed in {duration:.2f}s: "
            f"pre_hash={pre_hash[:8]}, post_hash={post_hash[:8]}, screen_changed={screen_changed}"
        )
        logger.info("[NODE: EXECUTE] -> Transitioning to RECORD")

        step_result = StepResult(
            step=step,
            pre_hash=pre_hash,
            post_hash=post_hash,
            duration=int(duration * 1000),
            error=execution_result.error,
            screen_changed=screen_changed,
            success=execution_result.success,
        )

        return {
            CommonStateKey.STEP_RESULT: step_result,
            CommonStateKey.EXECUTION_DURATION: duration,
        }

    async def record(self, state: IntentGraphState) -> IntentGraphState:
        """
        Record the execution result.
        """

        logger.info("=" * 80)
        logger.info("[NODE: RECORD] Starting record node")

        if self.__context.is_cancelled:
            logger.warning("[NODE: RECORD] Execution cancelled")
            return {CommonStateKey.IS_COMPLETE: True}

        result = state.get(CommonStateKey.STEP_RESULT)
        if not isinstance(result, StepResult):
            logger.error("[NODE: RECORD] No valid step result found")
            return {}

        step_result: StepResult = result

        logger.info(
            f"[NODE: RECORD] Recording step: success={step_result.success}, "
            f"screen_changed={step_result.screen_changed}, duration={step_result.duration}ms"
        )

        self.__context.agent_state.record_step(result=step_result)
        self.__context.history.save_step(result=step_result, intent=self.__context.intent)

        await self.__context.memory.store_experience(
            success=step_result.success,
            action=step_result.step.action,
            visual_hash=step_result.pre_hash,
        )

        # Commit cycle to ContextManager (GCC Trace)
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

        # GCC Branching - Semantic compression for long-running workflows
        # Threshold: 15 steps balances context freshness with compression benefits
        # For 100-150 step workflows, this creates ~7-10 milestones
        full_context = self.__context.context_manager.get_full_context()
        active_count = full_context.get("active_count", 0)

        BRANCHING_THRESHOLD = 15
        if active_count >= BRANCHING_THRESHOLD:
            logger.info(f"[NODE: RECORD] Triggering GCC branch: active_count={active_count}")
            await self.__context.context_manager.branch()

        # Audit logging
        execution_plan = state.get(IntentStateKey.PLAN)
        current_screen = state.get(CommonStateKey.SCREEN_STATE)
        is_new_screen = state.get(CommonStateKey.IS_NEW_SCREEN)

        if (
            isinstance(execution_plan, PlanResult)
            and isinstance(current_screen, ScreenState)
            and isinstance(is_new_screen, bool)
        ):
            # Explicit float conversion for metrics
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

            self.__context.auditor.log_step(
                plan=execution_plan,
                state=current_screen,
                hierarchy_duration=0.0,
                is_new_screen=is_new_screen,
                analysis_duration=analysis_duration,
                execution_duration=execution_duration,
                grounding_duration=grounding_duration,
                is_stuck=self.__context.agent_state.is_stuck,
                step_count=self.__context.agent_state.step_count,
                total_duration=grounding_duration + analysis_duration + execution_duration,
                result=ActionResult(success=step_result.success, duration=step_result.duration),
            )

        # Check max steps
        if self.__context.agent_state.step_count >= self.__context.max_steps:
            self.__context.agent_state.mark_complete(reason="Max steps reached")
            logger.info(f"[NODE: RECORD] Max steps reached ({self.__context.max_steps})")
            logger.info("[NODE: RECORD] -> Will route to END")
            return {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: "Max steps reached",
            }

        logger.info(
            f"[NODE: RECORD] Step {self.__context.agent_state.step_count} recorded successfully"
        )
        logger.info("[NODE: RECORD] -> Will route to GROUND for next step")
        return {}


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
        }
