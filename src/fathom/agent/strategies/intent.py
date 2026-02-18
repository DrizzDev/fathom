from __future__ import annotations

import asyncio
import time
from datetime import datetime
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console

from fathom.agent.planner import StepPlanner
from fathom.agent.reasoner import Reasoner
from fathom.agent.state import AgentState
from fathom.agent.strategies.base import ExecutionStrategy
from fathom.constants import ActionType, StrategyStatus
from fathom.infrastructure.memory.ledger import Ledger
from fathom.interfaces import ILedger, IMemoryProvider
from fathom.prompts.preprocessor import PromptPreprocessor
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.results import ActionResult, PlanResult, StrategyResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.services.audit import AuditService
from fathom.services.hierarchy import HierarchyService
from fathom.services.history import HistoryService
from fathom.services.resolution import ReferenceResolutionService
from fathom.services.ux import UXService
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision.processing.annotator import ImageAnnotator
from fathom.utils.coordinates import CoordinateConverter

console = Console()
logger = getLogger(name=__name__)


class IntentStrategy(ExecutionStrategy):
    """
    Strategy for executing a specific intent.
    """

    def __init__(
        self,
        intent: str,
        planner: StepPlanner,
        device: DeviceTool,
        capture: CaptureTool,
        memory: IMemoryProvider,
        *,
        max_steps: int = 20,
        use_xml: bool = False,
        step_timeout: float = 15.0,
        workflow_id: str = "default",
        package_name: str = "",
    ) -> None:
        self.__intent = intent
        self.__planner = planner

        self.__device = device
        self.__capture = capture

        self.__memory = memory
        self.__ledger: ILedger = Ledger()

        self.__use_xml = use_xml
        self.__max_steps = max_steps

        self.__reasoner = Reasoner(intent=intent)
        self.__state = AgentState(intent=intent, max_steps=max_steps)

        self.__ux_service = UXService()
        self.__audit_service = AuditService()

        self.__hierarchy = HierarchyService(device=device)
        self.__history = HistoryService(
            workflow_id=workflow_id, intent=intent, package_name=package_name
        )
        self.__resolution = ReferenceResolutionService(ledger=self.__ledger)

        self.__start_time = time.time()
        self.__metrics = ExecutionMetrics()

    @property
    def metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Return reporting metrics.
        """

        return self.__metrics.to_report_dict()

    @property
    def name(self) -> str:
        """
        Strategy name.
        """

        return "intent"

    @property
    def state(self) -> AgentState:
        """
        Agent state.
        """

        return self.__state

    async def execute_step(self) -> StrategyResult:
        """
        Executes a single step
        """

        step_start = time.time()

        # 1. GROUNDING
        screen, xml, grounding_duration = await self.__perform_grounding()
        self.__metrics.record(operation="screenshot", duration=grounding_duration)

        if not screen:
            return StrategyResult(status=StrategyStatus.ERROR, message="Capture failed")

        state, is_new_screen = self.__update_state(screen=screen)

        # 2. HIERARCHY
        planning_screen, elements, hierarchy_duration = await self.__process_hierarchy(
            screen=screen, xml=xml
        )
        if hierarchy_duration > 0:
            self.__metrics.record(operation="hierarchy_processing", duration=hierarchy_duration)

        if planning_screen is None:
            return StrategyResult(status=StrategyStatus.CONTINUE, message="Loading...")

        # 3. ANALYSIS
        plan, knowledge, analysis_duration = await self.__perform_analysis(
            state=state, planning_screen=planning_screen, elements=elements
        )
        self.__metrics.record(operation="analysis", duration=analysis_duration)

        # Track token usage from the LLM call
        # Track token usage from the LLM call
        if plan.metrics:
            self.__metrics.record_tokens(
                prompt=int(plan.metrics.get("prompt_tokens", 0)),
                completion=int(plan.metrics.get("completion_tokens", 0)),
                cached=int(plan.metrics.get("cached_tokens", 0)),
            )

        # RETRY: Invalid action
        if getattr(plan, "is_valid_action", True) is False:
            reason = getattr(plan, "validation_reasoning", "Invalid action")
            logger.warning(f"Invalid action: {reason}")
            self.__state.set_last_error(reason)
            return StrategyResult(
                status=StrategyStatus.CONTINUE, message=f"Invalid action: {reason}"
            )

        step = plan.step

        if not step:
            if plan.should_retry:
                return StrategyResult(status=StrategyStatus.CONTINUE, message=plan.reason)

            # If no step but complete, we are done
            if plan.is_complete:
                self.__audit_service.print_session_summary()
                return StrategyResult(status=StrategyStatus.COMPLETE, message=plan.reason)

            self.__audit_service.print_session_summary()
            return StrategyResult(status=StrategyStatus.ERROR, message=plan.reason)

        # HARDWARE BACK: Always use keycode for back actions
        if step.action.action_type == ActionType.BACK:
            # Override to use hardware key capability (bounds=None)
            # This ensures we use the device back button instead of clicking a UI element
            action = step.action.model_copy(update={"bounds": None})
            step = step.model_copy(update={"action": action})

        # RESOLUTION: Resolve dynamic references ($memory, $env)
        step = await self.__resolve_references(step=step)

        if self.__use_xml and step.action.label_id:
            step = await self.__resolve_coordinates(step=step)

        # 4. EXECUTION
        self.__render_ux(plan=plan, step=step, duration=analysis_duration)

        result, execution_duration, coordinates = await self.__execute_action_step(
            step=step, screen=screen
        )
        self.__metrics.record(operation="action", duration=execution_duration)

        # 5. RECORDING & AUDIT
        step_result = self.__record_result(
            step=step,
            state=state,
            result=result,
            step_start=step_start,
            coordinates=coordinates,
        )

        self.__audit_service.record_context(
            knowledge=knowledge,
            success=result.success,
            visual_hash=state.visual_hash,
            step_number=self.__state.step_count,
            context=self.__state.build_context(),
            action_description=step.action.to_description(),
        )

        self.__audit_service.log_step(
            plan=plan,
            state=state,
            result=result,
            is_new_screen=is_new_screen,
            is_stuck=self.__state.is_stuck,
            step_count=self.__state.step_count,
            analysis_duration=analysis_duration,
            grounding_duration=grounding_duration,
            hierarchy_duration=hierarchy_duration,
            execution_duration=execution_duration,
            total_duration=time.time() - step_start,
        )

        return StrategyResult(step_result=step_result, status=StrategyStatus.CONTINUE, message="OK")

    async def __perform_grounding(self) -> Tuple[Optional[ScreenCapture], Optional[str], float]:
        """
        Captures screen and hierarchy.
        """

        start = time.time()

        if self.__use_xml:
            screen_task = self.__capture.capture_stable(timeout=2000)
            xml_task = self.__device.dump_hierarchy()
            screen, xml = await asyncio.gather(screen_task, xml_task)
        else:
            xml = None
            screen = await self.__capture.capture_stable(timeout=2000)

        return screen, xml, time.time() - start

    def __update_state(self, screen: ScreenCapture) -> Tuple[ScreenState, bool]:
        """
        Updates agent state.
        """

        state = self.__capture.compute_state(capture=screen)

        screen = screen.model_copy(update={"state": state})
        is_new = self.__state.update_screen(screen=state)

        return state, is_new

    async def __process_hierarchy(
        self, screen: ScreenCapture, xml: Optional[str]
    ) -> Tuple[Optional[ScreenCapture], Dict[str, Any], float]:
        """
        Processes XML hierarchy.
        """

        if not (self.__use_xml and xml):
            logger.debug("XML Grounding disabled or XML missing.")
            return screen, {}, 0.0

        start = time.time()
        xml_size_kb = len(xml.encode("utf-8")) / 1024
        logger.info(f"Hierarchy processing started. XML Size: {xml_size_kb:.2f} KB")

        if xml_size_kb < 0.2:  # Threshold for very likely invalid/empty XML
            logger.warning("XML seems too small to be valid, waiting for UI stability...")
            await asyncio.sleep(1.0)
            return screen, {}, time.time() - start

        try:
            annotated, mapping = await self.__hierarchy.process_xml_and_screen(
                screen=screen, xml=xml, action_type=ActionType.TAP
            )

            duration = time.time() - start
            if annotated and annotated.image != screen.image:
                logger.info(f"Successfully annotated screen hierarchy in {duration:.2f}s")
                return annotated, mapping, duration

            logger.warning("Hierarchy processed but no annotation generated.")
            return screen, mapping, duration

        except Exception as exception:
            logger.exception(f"Error during hierarchy processing: {exception}")
            return screen, {}, time.time() - start

    async def __perform_analysis(
        self,
        state: ScreenState,
        elements: Dict[str, Any],
        planning_screen: ScreenCapture,
    ) -> Tuple[PlanResult, Dict[str, Any], float]:
        """
        Retrieves knowledge and plans next step.
        """

        entries = await self.__ledger.get_all()
        knowledge = await self.__memory.retrieve_knowledge(visual_hash=state.visual_hash)

        knowledge["memory_store"] = entries

        start = time.time()

        # Enhanced Context
        smart_context = self.__state.get_smart_context()

        # Prompt Preprocessing Hints
        hints = PromptPreprocessor.extract_hints(
            intent=self.__state.intent, current_activity=state.activity or ""
        )
        hint_str = PromptPreprocessor.build_context_prefix(hints)

        full_context = smart_context
        if hint_str:
            full_context = f"{hint_str}\n{smart_context}"

        plan = await self.__planner.plan_step(
            state=self.__state,
            use_xml=self.__use_xml,
            capture=planning_screen,
            reasoner=self.__reasoner,
            elements=elements if elements else None,
            additional_context=full_context,
        )
        return plan, knowledge, time.time() - start

    def __render_ux(self, plan: PlanResult, step: Step, duration: float) -> None:
        """
        Renders UX based on plan type.
        """

        if plan.metadata.get("tool_name"):
            self.__ux_service.render_tool_call(
                duration=duration,
                args=plan.metadata["tool_args"],
                tool_name=plan.metadata["tool_name"],
            )
        else:
            self.__ux_service.render_fallback(
                reasoning=step.action.rationale,
                action=step.action.to_description(),
                step_number=self.__state.step_count + 1,
            )

    async def __execute_action_step(
        self, step: Step, screen: ScreenCapture
    ) -> Tuple[ActionResult, float, Optional[List[int]]]:
        """
        Executes the action.
        """

        start = time.time()

        # Handle memory actions
        if step.action.action_type == ActionType.SAVE_MEMORY:
            if step.action.memory_updates:
                for key, value in step.action.memory_updates.items():
                    await self.__ledger.set(key=key, value=value)

            return ActionResult(success=True, duration=0), time.time() - start, None

        if step.action.action_type == ActionType.RETRIEVE_MEMORY:
            return ActionResult(success=True, duration=0), time.time() - start, None

        # Physical Action
        coordinates = await self.__get_action_coordinates(action=step.action)
        await self.__trace_background(
            action=step.action, image_data=screen.image, coordinates=coordinates
        )

        result = await self.__execute(action=step.action)

        if step.action.memory_updates:
            for key, value in step.action.memory_updates.items():
                await self.__ledger.set(key=key, value=value)

        # Convert to list for storage
        center = None

        if coordinates:
            coords_list = list(coordinates)
            if len(coords_list) == 2:
                center = coords_list
            elif len(coords_list) == 4:
                center = [
                    (coords_list[0] + coords_list[2]) // 2,
                    (coords_list[1] + coords_list[3]) // 2,
                ]

        return result, time.time() - start, center

    def __record_result(
        self,
        step: Step,
        step_start: float,
        state: ScreenState,
        result: ActionResult,
        coordinates: Optional[List[int]] = None,
    ) -> StepResult:
        """
        Records the step result.
        """

        step_result = StepResult(
            step=step,
            post_hash="0",
            screen_changed=True,
            success=result.success,
            pre_hash=state.visual_hash,
            duration=int((time.time() - step_start) * 1000),
        )
        self.__state.record_step(result=step_result)

        asyncio.create_task(
            coro=self.__memory.store_experience(
                action=step.action,
                success=result.success,
                visual_hash=step_result.pre_hash,
            )
        )
        self.__history.save_step(
            result=step_result, absolute_center=coordinates, activity=state.activity
        )
        return step_result

    async def should_continue(self) -> bool:
        """
        Check stop conditions.
        """

        return not self.__state.is_complete and self.__state.can_continue

    def get_progress(self) -> Dict[str, object]:
        """
        Return workflow progress.
        """

        return {
            "intent": self.__intent,
            "step_count": self.__state.step_count,
            "is_complete": self.__state.is_complete,
        }

    async def __resolve_references(self, step: Step) -> Step:
        """
        Resolves dynamic references in the action.
        """
        resolved_action = await self.__resolution.resolve(action=step.action)
        return step.model_copy(update={"action": resolved_action})

    async def __resolve_coordinates(self, step: Step) -> Step:
        """
        Resolves label to physical coordinates.
        """

        label_id = step.action.label_id
        mapping = self.__hierarchy.label_map

        if label_id and label_id in mapping:
            element = mapping[label_id]
            size = await self.__device.get_screen_size()

            if size[0] > 0 and size[1] > 0:
                normalized_x = int((element["center_x"] / size[0]) * 1000)
                normalized_y = int((element["center_y"] / size[1]) * 1000)

                # Resolve natural name from element text/description
                element_name = element.get("text") or element.get("content-desc")

                updates: Dict[str, Any] = {
                    "bounds": Bounds(
                        width=100,
                        height=100,
                        x=normalized_x - 50,
                        y=normalized_y - 50,
                    )
                }

                if element_name and str(element_name).strip():
                    updates["target"] = str(element_name).strip()
                    updates["natural_language_target"] = str(element_name).strip()

                action = step.action.model_copy(update=updates)
                return step.model_copy(update={"action": action})

        return step

    async def __get_action_coordinates(self, action: Action) -> Tuple[int, ...]:
        """
        Converts bounds to pixels.
        """

        size = await self.__device.get_screen_size()
        converter = CoordinateConverter(screen_width=size[0], screen_height=size[1])

        if action.action_type in (
            ActionType.TAP,
            ActionType.TYPE,
            ActionType.LONG_PRESS,
        ):
            if action.bounds:
                return converter.center_to_pixels(bounds=action.bounds)

            return (size[0] // 2, size[1] // 2)

        if action.action_type in (
            ActionType.SWIPE,
            ActionType.SCROLL,
            ActionType.SWIPE_UP,
            ActionType.SWIPE_DOWN,
            ActionType.SWIPE_LEFT,
            ActionType.SWIPE_RIGHT,
        ):
            if action.bounds:
                direction = "up"
                if "_" in action.action_type.value:
                    direction = action.action_type.value.split("_")[1]
                return converter.swipe_coordinates(bounds=action.bounds, direction=direction)

            return (size[0] // 2, size[1] * 3 // 4, size[0] // 2, size[1] // 4)

        return ()

    async def __trace_background(
        self,
        action: Action,
        image_data: bytes,
        coordinates: Tuple[int, ...],
    ) -> None:
        """
        Saves annotated trace.
        """

        if not coordinates:
            logger.debug("No coordinates for tracing, skipping.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = (
            f"step__{self.__state.step_count + 1}__{action.action_type.value}__{timestamp}.png"
        )
        path = f"assets/traces/{filename}"

        logger.info(f"Saving trace image: {path}")
        result = ImageAnnotator.trace(
            output_path=path,
            coords=coordinates,
            image_data=image_data,
            label=action.to_description(),
            action_type=action.action_type.value,
        )
        if not result:
            logger.warning(f"Failed to save trace image to {path}")

    async def __execute(self, action: Action) -> ActionResult:
        """
        Dispatches action to device.
        """

        size = await self.__device.get_screen_size()
        converter = CoordinateConverter(screen_width=size[0], screen_height=size[1])

        if action.action_type == ActionType.TAP:
            coordinates = (
                converter.center_to_pixels(bounds=action.bounds)
                if action.bounds
                else (size[0] // 2, size[1] // 2)
            )
            return await self.__device.tap(x=coordinates[0], y=coordinates[1])

        if action.action_type == ActionType.TYPE:
            if not action.bounds:
                return ActionResult(
                    success=False,
                    duration=0,
                    error="Type action requires bounds for focus tap guard",
                )
            x, y = converter.center_to_pixels(bounds=action.bounds)
            focus_result = await self.__device.tap(x=x, y=y)
            if not focus_result.success:
                return ActionResult(
                    success=False,
                    duration=0,
                    error=f"Focus tap failed before typing: {focus_result.error or 'unknown error'}",
                )
            return await self.__device.type_text(text=action.text or "")

        if action.action_type in (
            ActionType.SWIPE_LEFT,
            ActionType.SWIPE_RIGHT,
            ActionType.SWIPE_UP,
            ActionType.SWIPE_DOWN,
        ):
            direction = action.action_type.value.split("_")[1]
            coords = converter.swipe_coordinates(
                bounds=action.bounds or Bounds(x=200, y=200, width=600, height=600),
                direction=direction,
            )
            return await self.__device.swipe(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

        if action.action_type == ActionType.WAIT:
            duration = action.wait_duration or 1000

            await asyncio.sleep(delay=duration / 1000.0)
            return ActionResult(success=True, duration=duration)

        if action.action_type == ActionType.BACK:
            return await self.__device.back()

        if action.action_type == ActionType.HOME:
            return await self.__device.home()

        return await self.__device.execute(request=action.model_dump())
