from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fathom.constants import ActionType
from fathom.exceptions import ToolError
from fathom.orchestration.context import ExecutionContext
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step, StepResult
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool


@dataclass(frozen=True)
class ExecutionResult:
    """
    Result of step execution attempt.
    """

    success: bool
    duration: int
    error: Optional[str] = None
    screen_changed: bool = False
    pre_hash: str = ""
    post_hash: str = ""


class StepExecutor:
    """Executes individual steps on a device.

    Responsible for:
    - Translating steps to device actions
    - Executing with timeout handling
    - Detecting screen changes
    - Retry logic for transient failures

    Stateless - can be shared across workflows.
    """

    def __init__(
        self,
        device: DeviceTool,
        capture: CaptureTool,
        *,
        default_timeout: float = 15.0,
        stability_wait: float = 0.5,
        max_retries: int = 2,
    ) -> None:
        """Initialize executor.

        Args:
            device: Device tool for action execution.
            capture: Capture tool for screenshots.
            default_timeout: Default step timeout in seconds.
            stability_wait: Wait time after action for screen stability.
            max_retries: Maximum retry attempts for transient failures.
        """
        self.__device = device
        self.__capture = capture
        self.__default_timeout = default_timeout
        self.__stability_wait = stability_wait
        self.__max_retries = max_retries

    async def execute(
        self,
        step: Step,
        context: ExecutionContext,
        *,
        pre_capture: Optional[ScreenCapture] = None,
    ) -> StepResult:
        """Execute a step with full lifecycle.

        Args:
            step: Step to execute.
            context: Execution context for tracking.
            pre_capture: Optional pre-captured screen state.

        Returns:
            StepResult with execution details.
        """
        step_ctx = context.start_step(step.step_number)
        start_time = time.time()

        try:
            if pre_capture is None:
                pre_capture = await self.__capture.capture()
            pre_state = self.__capture.compute_state(pre_capture)
            pre_hash = pre_state.visual_hash

            result = await self.__execute_with_retry(step)

            await asyncio.sleep(self.__stability_wait)

            post_capture = await self.__capture.capture()
            post_state = self.__capture.compute_state(post_capture)
            post_hash = post_state.visual_hash

            screen_changed = pre_hash != post_hash
            duration_ms = int((time.time() - start_time) * 1000)

            step_result = StepResult(
                step=step,
                success=result.success,
                screen_changed=screen_changed,
                pre_hash=pre_hash,
                post_hash=post_hash,
                duration=duration_ms,
                error=result.error,
            )

            context.complete_step(step_ctx, step_result)
            return step_result

        except Exception as exception:
            duration_ms = int((time.time() - start_time) * 1000)
            step_result = StepResult(
                step=step,
                success=False,
                screen_changed=False,
                pre_hash="",
                post_hash="",
                duration=duration_ms,
                error=str(exception),
            )
            context.complete_step(step_ctx, step_result)
            return step_result

    async def __execute_with_retry(self, step: Step) -> ExecutionResult:
        """Execute step with retry logic.

        Args:
            step: Step to execute.

        Returns:
            Execution result.
        """
        last_error: Optional[str] = None

        for attempt in range(self.__max_retries + 1):
            try:
                result = await self.__execute_action(step)
                if result.success:
                    return result
                last_error = result.error

            except ToolError as exception:
                last_error = str(exception)
                if not exception.retryable:
                    break

            except Exception as exception:
                last_error = str(exception)

            if attempt < self.__max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))

        return ExecutionResult(
            success=False,
            duration=0,
            error=last_error or "Unknown error",
        )

    async def __execute_action(self, step: Step) -> ExecutionResult:
        """Execute device action for step.

        Args:
            step: Step containing action to execute.

        Returns:
            Execution result.
        """
        action = step.action
        start_time = time.time()

        screen_size = await self.__device.get_screen_size()
        width, height = screen_size

        request = self.__build_request(action, width, height)

        if request.get("action") == "wait":
            await asyncio.sleep(1.0)
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(success=True, duration=duration_ms)

        if request.get("action") == "complete":
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(success=True, duration=duration_ms)

        result = await self.__device.execute(request)
        duration_ms = int((time.time() - start_time) * 1000)

        return ExecutionResult(
            success=result.success,
            duration=duration_ms,
            error=result.error,
        )

    def __build_request(
        self,
        action: "Step.action.__class__",  # type: ignore[name-defined]
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        """
        Build device request from action.
        """
        from fathom.schemas.actions import Action

        if not isinstance(action, Action):
            return {"action": "unknown"}

        if action.action_type == ActionType.TAP:
            if action.bbox:
                x = action.bbox.center_x * width // 1000
                y = action.bbox.center_y * height // 1000
            else:
                x, y = width // 2, height // 2
            return {"action": "tap", "x": x, "y": y}

        if action.action_type == ActionType.TYPE:
            return {"action": "type", "text": action.text or ""}

        if action.action_type == ActionType.SWIPE:
            cx, cy = width // 2, height // 2
            return {
                "action": "swipe",
                "x1": cx,
                "y1": cy + 300,
                "x2": cx,
                "y2": cy - 300,
            }

        if action.action_type == ActionType.SCROLL:
            cx, cy = width // 2, height // 2
            return {
                "action": "swipe",
                "x1": cx,
                "y1": cy + 400,
                "x2": cx,
                "y2": cy - 400,
                "duration": 500,
            }

        if action.action_type == ActionType.LONG_PRESS:
            if action.bbox:
                x = action.bbox.center_x * width // 1000
                y = action.bbox.center_y * height // 1000
            else:
                x, y = width // 2, height // 2
            return {
                "action": "swipe",
                "x1": x,
                "y1": y,
                "x2": x,
                "y2": y,
                "duration": 1000,
            }

        if action.action_type == ActionType.BACK:
            return {"action": "back"}

        if action.action_type == ActionType.HOME:
            return {"action": "home"}

        if action.action_type == ActionType.WAIT:
            return {"action": "wait", "duration": 1000}

        if action.action_type == ActionType.COMPLETE:
            return {"action": "complete"}

        return {"action": "unknown"}
