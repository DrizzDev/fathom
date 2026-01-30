from __future__ import annotations

import time
from typing import Optional

from fathom.agent.planner import StepPlanner
from fathom.agent.reasoner import Reasoner
from fathom.agent.state import AgentState
from fathom.agent.strategies.base import (
    ExecutionStrategy,
)
from fathom.constants import StrategyStatus
from fathom.schemas.results import ActionResult, StrategyResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step, StepResult
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool


class IntentStrategy(ExecutionStrategy):
    """Strategy for executing a specific intent.

    Focuses on achieving a single goal efficiently:
    - Plans steps based on current screen
    - Executes actions and tracks results
    - Detects completion or failure
    - Handles recovery when stuck

    This is the primary strategy for goal-directed automation.
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
    ) -> None:
        """Initialize intent strategy.

        Args:
            intent: Goal to achieve.
            planner: Step planner for action recommendation.
            device: Device tool for action execution.
            capture: Capture tool for screenshots.
            max_steps: Maximum steps before timeout.
            step_timeout: Per-step timeout in seconds.
        """
        self.__intent = intent
        self.__planner = planner
        self.__device = device
        self.__capture = capture
        self.__max_steps = max_steps
        self.__step_timeout = step_timeout

        self.__state = AgentState(intent, max_steps=max_steps)
        self.__reasoner = Reasoner(intent)

        self.__last_step: Optional[Step] = None
        self.__start_time = time.time()

    @property
    def name(self) -> str:
        """
        Strategy name.
        """
        return "intent"

    @property
    def state(self) -> AgentState:
        """
        Current agent state.
        """
        return self.__state

    async def execute_step(self) -> StrategyResult:
        """Execute a single step toward the intent.

        Flow:
        1. Capture current screen
        2. Compute screen state for loop detection
        3. Plan next step via vision
        4. Execute action on device
        5. Verify screen changed
        6. Update state with result

        Returns:
            Result indicating execution status.
        """
        step_start = time.time()

        screen_capture = await self.__capture_with_timeout()
        if screen_capture is None:
            return StrategyResult(
                status=StrategyStatus.ERROR,
                step_result=None,
                message="Screen capture failed",
            )

        screen_state = self.__capture.compute_state(screen_capture)
        self.__state.update_screen(screen_state)

        plan = await self.__planner.plan_step(
            self.__state,
            screen_capture,
            self.__reasoner,
        )

        if plan.is_complete:
            return StrategyResult(
                status=StrategyStatus.COMPLETE,
                step_result=None,
                message=plan.reason,
                should_checkpoint=True,
            )

        if plan.step is None:
            if self.__state.is_stuck:
                return StrategyResult(
                    status=StrategyStatus.STUCK,
                    step_result=None,
                    message=plan.reason,
                )
            return StrategyResult(
                status=StrategyStatus.ERROR,
                step_result=None,
                message=plan.reason,
            )

        screen_size = await self.__device.get_screen_size()
        execution_request = self.__planner.prepare_execution(
            plan.step,
            screen_size[0],
            screen_size[1],
        )

        action_result = await self.__execute_action(execution_request)

        await self.__wait_for_stability()
        post_capture = await self.__capture_with_timeout()

        if post_capture:
            post_state = self.__capture.compute_state(post_capture)
            screen_changed = not screen_state.is_same_screen(post_state)
        else:
            screen_changed = False
            post_state = screen_state

        step_duration = int((time.time() - step_start) * 1000)

        step_result = StepResult(
            step=plan.step,
            success=action_result.success,
            screen_changed=screen_changed,
            pre_hash=screen_state.visual_hash,
            post_hash=post_state.visual_hash,
            duration=step_duration,
            error=action_result.error,
        )

        self.__state.record_step(step_result)
        self.__last_step = plan.step

        should_checkpoint = (self.__state.step_count % 5) == 0

        return StrategyResult(
            status=StrategyStatus.CONTINUE,
            step_result=step_result,
            message=f"Executed: {plan.step.action.to_description()}",
            should_checkpoint=should_checkpoint,
        )

    async def __capture_with_timeout(self) -> Optional[ScreenCapture]:
        """
        Capture screen with timeout handling.
        """
        try:
            return await self.__capture.capture()
        except Exception:
            return None

    async def __execute_action(self, request: dict[str, object]) -> ActionResult:
        """
        Execute action on device.
        """
        return await self.__device.execute(request)

    async def __wait_for_stability(self) -> None:
        """
        Wait for screen to stabilize after action.
        """
        import asyncio

        await asyncio.sleep(0.5)

    async def should_continue(self) -> bool:
        """
        Check if execution should continue.
        """
        if self.__state.is_complete:
            return False
        if not self.__state.can_continue:
            return False
        elapsed = time.time() - self.__start_time
        max_time = self.__max_steps * self.__step_timeout
        return elapsed <= max_time

    def get_progress(self) -> dict[str, object]:
        """
        Get current progress information.
        """
        return {
            "intent": self.__intent,
            "step_count": self.__state.step_count,
            "max_steps": self.__max_steps,
            "is_complete": self.__state.is_complete,
            "is_stuck": self.__state.is_stuck,
            "elapsed_seconds": time.time() - self.__start_time,
            "last_action": (self.__last_step.action.to_description() if self.__last_step else None),
            "context": self.__state.build_context(),
        }

    def get_checkpoint(self) -> dict[str, object]:
        """
        Get checkpoint data for persistence.
        """
        return {
            "strategy": self.name,
            "intent": self.__intent,
            "state": self.__state.to_checkpoint(),
            "start_time": self.__start_time,
            "progress": self.get_progress(),
        }
