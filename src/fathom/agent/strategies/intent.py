from __future__ import annotations

import asyncio
import time
from logging import getLogger
from typing import Dict, Optional

from fathom.agent.planner import StepPlanner
from fathom.agent.reasoner import Reasoner
from fathom.agent.state import AgentState
from fathom.agent.strategies.base import ExecutionStrategy
from fathom.constants import ActionType, StrategyStatus
from fathom.schemas.actions import Action, BoundingBox
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.results import ActionResult, StrategyResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step, StepResult
from fathom.services.hierarchy import HierarchyService
from fathom.services.history import HistoryService
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool

logger = getLogger(__name__)


class IntentStrategy(ExecutionStrategy):
    """
    Strategy for executing a specific intent.
    Strictly follows Fathom coding standards and delegates to services.
    """

    def __init__(
        self,
        intent: str,
        planner: StepPlanner,
        device: DeviceTool,
        capture: CaptureTool,
        *,
        max_steps: int = 20,
        step_timeout: float = 15.0,
        use_xml: bool = False,
        workflow_id: str = "default",
    ) -> None:
        self.__intent = intent
        self.__planner = planner
        self.__device = device
        self.__capture = capture
        self.__max_steps = max_steps
        self.__step_timeout = step_timeout
        self.__use_xml = use_xml

        self.__state = AgentState(intent, max_steps=max_steps)
        self.__reasoner = Reasoner(intent)
        self.__hierarchy = HierarchyService(device)
        self.__history = HistoryService(workflow_id)

        self.__start_time = time.time()
        self.__metrics = ExecutionMetrics()

    @property
    def name(self) -> str:
        """
        The name of this strategy.
        """
        return "intent"

    @property
    def metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Returns execution timing metrics formatted for reporting.
        """
        return self.__metrics.to_report_dict()

    @property
    def state(self) -> AgentState:
        """
        Returns the current internal agent state.
        """
        return self.__state

    async def execute_step(self) -> StrategyResult:
        """
        Executes a single step toward the goal.
        """
        step_start_timestamp = time.time()

        # 1. Capture screen
        capture_start_timestamp = time.time()
        screen_capture = await self.__capture_with_timeout()
        self.__metrics.record("screenshot", time.time() - capture_start_timestamp)

        if screen_capture is None:
            return StrategyResult(status=StrategyStatus.ERROR, message="Screen capture failed")

        self.__state.update_screen(self.__capture.compute_state(screen_capture))

        # 2. UI Grounding (XML and Labeling)
        screen_for_planning = screen_capture
        if self.__use_xml:
            (
                annotated_capture,
                dump_duration,
                processing_duration,
            ) = await self.__hierarchy.process_screen(screen_capture)
            self.__metrics.record("hierarchy_dump", dump_duration)
            self.__metrics.record("hierarchy_processing", processing_duration)
            screen_for_planning = annotated_capture or screen_capture

        # 3. Decision Making
        analysis_start_timestamp = time.time()
        plan_result = await self.__planner.plan_step(
            self.__state, self.__reasoner, screen_for_planning, use_xml=self.__use_xml
        )
        self.__metrics.record("analysis", time.time() - analysis_start_timestamp)

        if plan_result.is_complete:
            return StrategyResult(status=StrategyStatus.COMPLETE, message=plan_result.reason)

        if not plan_result.step:
            return StrategyResult(status=StrategyStatus.ERROR, message=plan_result.reason)

        # 4. Action Resolution (Labels to Bbox)
        step = plan_result.step
        if self.__use_xml and step.action.label_id:
            step = await self.__resolve_label_to_coordinates(step)

        # 5. Device Execution
        logger.info(f"Executing action: {step.action.to_description()}")
        execution_start_timestamp = time.time()
        action_result = await self.__execute_action(step.action)
        self.__metrics.record("action", time.time() - execution_start_timestamp)

        # 6. Post-Action Verification
        await asyncio.sleep(0.5)
        post_capture = await self.__capture_with_timeout()
        screen_changed_status = False
        pre_hash = self.__capture.compute_state(screen_capture).visual_hash
        post_hash = pre_hash

        if post_capture:
            post_state = self.__capture.compute_state(post_capture)
            post_hash = post_state.visual_hash
            screen_changed_status = pre_hash != post_hash

        # Final null-check for step to satisfy Mypy
        if not step:
            return StrategyResult(
                status=StrategyStatus.ERROR, message="Step lost during resolution"
            )

        # Capture absolute center for history
        absolute_center = None
        if step.action.bbox:
            screen_width, screen_height = await self.__device.get_screen_size()
            center_x = (step.action.bbox.x + step.action.bbox.width // 2) * screen_width // 1000
            center_y = (step.action.bbox.y + step.action.bbox.height // 2) * screen_height // 1000
            absolute_center = [center_x, center_y]

        step_result = StepResult(
            step=step,
            success=action_result.success,
            screen_changed=screen_changed_status,
            pre_hash=pre_hash,
            post_hash=post_hash,
            duration=int((time.time() - step_start_timestamp) * 1000),
        )

        self.__state.record_step(step_result)
        self.__history.save_step(step_result, absolute_center=absolute_center)

        logger.info(
            f"Step Audit | Index: {self.__state.step_count} | "
            f"LLM: {self.__metrics.analysis.total_duration / max(1, self.__metrics.analysis.call_count):.2f}s (avg)"
        )

        return StrategyResult(
            status=StrategyStatus.CONTINUE,
            step_result=step_result,
            message="Step executed successfully",
        )

    async def should_continue(self) -> bool:
        """
        Determines if execution should proceed.
        """
        if self.__state.is_complete:
            return False
        if not self.__state.can_continue:
            return False
        elapsed_time = time.time() - self.__start_time
        return elapsed_time <= (self.__max_steps * self.__step_timeout)

    def get_progress(self) -> Dict[str, object]:
        """
        Returns current progress summary.
        """
        return {
            "intent": self.__intent,
            "step_count": self.__state.step_count,
            "max_steps": self.__max_steps,
            "is_complete": self.__state.is_complete,
            "elapsed_seconds": time.time() - self.__start_time,
        }

    async def __resolve_label_to_coordinates(self, step: Step) -> Step:
        """
        Resolves a visual label ID back to normalized coordinates for execution.
        """
        label_id = step.action.label_id
        label_mapping = self.__hierarchy.label_map
        if label_id in label_mapping:
            element_data = label_mapping[label_id]
            screen_width, screen_height = await self.__device.get_screen_size()

            if screen_width > 0 and screen_height > 0:
                normalized_x = int((element_data["center_x"] / screen_width) * 1000)
                normalized_y = int((element_data["center_y"] / screen_height) * 1000)

                updated_action = step.action.model_copy(
                    update={
                        "bbox": BoundingBox(
                            x=normalized_x - 50, y=normalized_y - 50, width=100, height=100
                        )
                    }
                )
                return step.model_copy(update={"action": updated_action})
        return step

    async def __execute_action(self, action: Action) -> ActionResult:
        """
        Maps generic actions to device commands.
        """
        from fathom.utils.coordinates import CoordinateConverter

        screen_size = await self.__device.get_screen_size()
        converter = CoordinateConverter(screen_size[0], screen_size[1])

        if action.action_type == ActionType.TAP:
            coordinates = (
                converter.center_to_pixels(action.bbox)
                if action.bbox
                else (screen_size[0] // 2, screen_size[1] // 2)
            )
            return await self.__device.tap(coordinates[0], coordinates[1])
        elif action.action_type == ActionType.TYPE:
            return await self.__device.type_text(action.text or "")
        elif action.action_type == ActionType.WAIT:
            await asyncio.sleep(1.0)
            return ActionResult(success=True, duration=1000)
        elif action.action_type == ActionType.COMPLETE:
            return ActionResult(success=True, duration=0)

        return await self.__device.execute(action.model_dump())

    async def __capture_with_timeout(self) -> Optional[ScreenCapture]:
        """
        Helper to capture screen with basic error handling.
        """
        try:
            return await self.__capture.capture()
        except Exception as exception:
            logger.warning(f"Screenshot capture failed: {exception}")
            return None
