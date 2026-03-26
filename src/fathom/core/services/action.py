from __future__ import annotations

import asyncio
import time
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Awaitable, Callable, Optional, Set, Tuple

from fathom.base.paths import SharedPathManager
from fathom.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_SWIPE_DURATION,
    DRAIN_TIMEOUT,
    ActionType,
)
from fathom.core.exceptions import ExecutionError, PortError, ToolError
from fathom.interfaces.device import DevicePort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.processing.annotator import ImageAnnotator
from fathom.schemas.actions import Action
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.results import ActionResult, ExecutionResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step
from fathom.utils.coordinates import CoordinateConverter

logger = getLogger(__name__)


class ActionExecutor:
    """
    Executes actions on the device with retry logic and tracing.
    """

    def __init__(
        self,
        device: DevicePort,
        telemetry: TelemetryPort,
        path_manager: SharedPathManager,
        storage: Optional[StoragePort] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.__device = device
        self.__telemetry = telemetry
        self.__max_retries = max_retries

        self.__storage = storage
        self.__path_manager = path_manager
        self.__background_tasks: Set[asyncio.Task[None]] = set()

    async def act(
        self,
        step: Step,
        session_id: str,
        package_name: str,
        pre_capture: ScreenCapture,
    ) -> ExecutionResult:
        """
        Execute device action with retry logic and tracing.
        """

        last_error: Optional[str] = None
        for attempt in range(self.__max_retries + 1):
            try:
                result, coords = await self.__execute_primitive(step=step)

                if result.success and coords:
                    self.__schedule_trace(
                        step=step,
                        coords=coords,
                        session_id=session_id,
                        pre_capture=pre_capture,
                        package_name=package_name,
                    )

                if result.success:
                    return result

                last_error = result.error

            except (ToolError, PortError) as exception:
                last_error = str(exception)

                if attempt < self.__max_retries:
                    await self.__telemetry.warning(
                        "Device operation failed, retrying",
                        attempt=attempt + 1,
                        error=str(exception),
                    )

            if attempt < self.__max_retries:
                await asyncio.sleep(delay=(DEFAULT_RETRY_DELAY / 1000.0) * (attempt + 1))

        return ExecutionResult(
            duration=0,
            success=False,
            error=last_error or "Unknown error",
        )

    async def __execute_primitive(
        self, step: Step
    ) -> Tuple[ExecutionResult, Optional[Tuple[int, ...]]]:
        """
        Execute specific device primitive.
        """

        action = step.action
        start_time = time.time()

        screen_size = await self.__device.get_dimensions()
        width, height = screen_size

        configuration = self.__device.configuration or DeviceRuntimeConfiguration()
        converter = CoordinateConverter(
            screen_width=width, screen_height=height, configuration=configuration
        )

        if self.__is_non_interactive_action(action=action):
            return await self.__execute_non_interactive_action(action=action, start_time=start_time)

        try:
            device_result, coords = await self.__execute_interactive_action(
                action=action,
                converter=converter,
                width=width,
                height=height,
            )
            if device_result is None:
                return (
                    ExecutionResult(
                        duration=0,
                        success=False,
                        error=f"Unknown action type: {action.action_type}",
                    ),
                    None,
                )

            duration = int((time.time() - start_time) * 1000)
            return (
                ExecutionResult(
                    duration=duration,
                    error=device_result.error,
                    success=device_result.success,
                ),
                coords,
            )

        except Exception as exception:
            duration = int((time.time() - start_time) * 1000)
            return (
                ExecutionResult(success=False, duration=duration, error=str(exception)),
                None,
            )

    def __apply_tap_bias(
        self, x: int, y: int, action: Action, converter: CoordinateConverter
    ) -> Tuple[int, int]:
        """
        Apply upward coordinate bias for VLM-detected bounds.
        Skips adjustment for label-snapped pixel bounds which are already grounded.
        """

        if not action.bounds:
            return x, y

        # Label-snapped pixel bounds are already grounded to exact device coordinates
        is_label_snapped_pixel = bool(action.label_id) and action.bounds.system.lower() == "pixel"

        if is_label_snapped_pixel:
            return x, y

        # Apply 20% upward bias for VLM-detected bounds to account for bounding box imprecision
        _, _, width_px, height_px = converter.to_pixels(bounds=action.bounds)
        if height_px > 0:
            y = max(0, y - max(2, int(height_px * 0.20)))

        return x, y

    def __is_non_interactive_action(self, *, action: Action) -> bool:
        """
        Check whether action can be completed without device interaction.
        """

        return action.action_type in {
            ActionType.WAIT,
            ActionType.COMPLETE,
            ActionType.VALIDATE,
            ActionType.SAVE_MEMORY,
            ActionType.RETRIEVE_MEMORY,
        }

    async def __execute_non_interactive_action(
        self, *, action: Action, start_time: float
    ) -> Tuple[ExecutionResult, Optional[Tuple[int, ...]]]:
        """
        Execute non-interactive action and return success result.
        """

        if action.action_type in {
            ActionType.WAIT,
            ActionType.VALIDATE,
            ActionType.SAVE_MEMORY,
            ActionType.RETRIEVE_MEMORY,
        }:
            await asyncio.sleep(delay=float(action.wait_duration or 1.0))

        return (
            ExecutionResult(success=True, duration=int((time.time() - start_time) * 1000)),
            None,
        )

    async def __execute_interactive_action(
        self,
        *,
        action: Action,
        converter: CoordinateConverter,
        width: int,
        height: int,
    ) -> Tuple[Optional[ActionResult], Optional[Tuple[int, ...]]]:
        """
        Execute interactive action through registered handlers.
        """

        if action.action_type.value.startswith(ActionType.SWIPE.lower()):
            return await self.__execute_swipe(action, converter, width, height)

        action_handlers: dict[
            ActionType,
            Callable[[], Awaitable[Tuple[ActionResult, Optional[Tuple[int, ...]]]]],
        ] = {
            ActionType.TAP: lambda: self.__execute_tap(action, converter, width, height),
            ActionType.TYPE: lambda: self.__execute_type(action, converter, width, height),
            ActionType.SCROLL: lambda: self.__execute_scroll(width, height),
            ActionType.LONG_PRESS: lambda: self.__execute_long_press(
                action, converter, width, height
            ),
            ActionType.BACK: self.__execute_back,
            ActionType.HOME: self.__execute_home,
        }

        handler = action_handlers.get(action.action_type)
        if handler is None:
            return None, None

        return await handler()

    async def __execute_back(self) -> Tuple[ActionResult, Optional[Tuple[int, ...]]]:
        """
        Execute back action.
        """

        return await self.__device.back(), None

    async def __execute_home(self) -> Tuple[ActionResult, Optional[Tuple[int, ...]]]:
        """
        Execute home action.
        """

        return await self.__device.home(), None

    async def __execute_tap(
        self, action: Action, converter: CoordinateConverter, width: int, height: int
    ) -> Tuple[ActionResult, Tuple[int, ...]]:
        """
        Helper Method To Execute `TAP` Command
        """

        if action.bounds:
            x, y = converter.center_to_pixels(bounds=action.bounds)
            x, y = self.__apply_tap_bias(x=x, y=y, action=action, converter=converter)
        else:
            x, y = width // 2, height // 2

        coords = (x, y)
        result = await self.__device.tap(x=x, y=y)

        return result, coords

    async def __execute_type(
        self, action: Action, converter: CoordinateConverter, width: int, height: int
    ) -> Tuple[ActionResult, Tuple[int, ...]]:
        """
        Helper Method To Execute `TYPE` Command
        """

        _ = width, height

        if not action.bounds:
            raise ExecutionError("Type action requires bounds for focus tap guard")

        x, y = converter.center_to_pixels(bounds=action.bounds)
        x, y = self.__apply_tap_bias(x=x, y=y, action=action, converter=converter)
        coords = (x, y)

        focus_result = await self.__device.tap(x=x, y=y)
        if not focus_result.success:
            return focus_result, coords

        result = await self.__device.type(text=action.text or "")
        return result, coords

    async def __execute_swipe(
        self, action: Action, converter: CoordinateConverter, width: int, height: int
    ) -> Tuple[ActionResult, Tuple[int, ...]]:
        """
        Helper Method To Execute `SWIPE` Command
        """

        if "_" in action.action_type.value:
            direction = action.action_type.value.split("_")[-1]
        else:
            direction = "up"

        if action.bounds:
            x1, y1, x2, y2 = converter.swipe_coordinates(bounds=action.bounds, direction=direction)
        else:
            # Full screen swipe if no bounds
            cx, cy = width // 2, height // 2
            offset = 300  # Reasonable default swipe distance

            if direction == "up":
                x1, y1 = cx, cy + offset
                x2, y2 = cx, cy - offset
            elif direction == "down":
                x1, y1 = cx, cy - offset
                x2, y2 = cx, cy + offset
            elif direction == "left":
                x1, y1 = cx + offset, cy
                x2, y2 = cx - offset, cy
            elif direction == "right":
                x1, y1 = cx - offset, cy
                x2, y2 = cx + offset, cy
            else:
                # Default to up
                x1, y1 = cx, cy + offset
                x2, y2 = cx, cy - offset

        coords = (x1, y1, x2, y2)
        result = await self.__device.swipe(x1=x1, y1=y1, x2=x2, y2=y2)
        return result, coords

    async def __execute_scroll(
        self, width: int, height: int
    ) -> Tuple[ActionResult, Tuple[int, ...]]:
        """
        Helper Method To Execute `SCROLL` Command (Default Scroll Down)
        """

        cx, cy = width // 2, height // 2

        # Scroll down content = Swipe UP
        x1, y1 = cx, cy + 300
        x2, y2 = cx, cy - 300
        coords = (x1, y1, x2, y2)

        result = await self.__device.swipe(
            x1=x1, y1=y1, x2=x2, y2=y2, duration=DEFAULT_SWIPE_DURATION
        )
        return result, coords

    async def __execute_long_press(
        self, action: Action, converter: CoordinateConverter, width: int, height: int
    ) -> Tuple[ActionResult, Tuple[int, ...]]:
        """
        Helper Method To Execute `LONG_PRESS` Command
        """

        if action.bounds:
            x, y = converter.center_to_pixels(bounds=action.bounds)
        else:
            x, y = width // 2, height // 2

        coords = (x, y)

        # Long-press via static swipe avoids triggering a separate tap side-effect.
        long_press_result = await self.__device.swipe(x1=x, y1=y, x2=x, y2=y, duration=1000)
        return long_press_result, coords

    def __schedule_trace(
        self,
        step: Step,
        session_id: str,
        package_name: str,
        coords: Tuple[int, ...],
        pre_capture: ScreenCapture,
    ) -> None:
        """
        Schedules background trace annotation.
        """

        async def __trace_and_upload(
            trace_path: str,
            action_type: str,
            image_data: bytes,
            label_description: str,
            coordinates: Tuple[int, ...],
        ) -> None:
            try:
                await asyncio.to_thread(
                    ImageAnnotator.trace,
                    coords=coordinates,
                    image_data=image_data,
                    output_path=trace_path,
                    action_type=action_type,
                    label=label_description,
                )

                if self.__storage:
                    with Path(trace_path).open("rb") as new_file:
                        data = new_file.read()

                    filename = Path(trace_path).name
                    await self.__storage.save(
                        data=data,
                        metadata={
                            "category": "traces",
                            "filename": filename,
                            "session_id": session_id,
                            "package_name": package_name,
                        },
                    )
            except Exception as exception:
                # Use standard logger in background thread
                logger.exception(f"Tracing failed: {exception}", stack_info=True)

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"step_{step.step_number}_{step.action.action_type.value}_{timestamp}.png"

            trace_path = str(
                self.__path_manager.get_trace_path(
                    filename=filename,
                    session_id=session_id,
                    package_name=package_name,
                )
            )

            task = asyncio.create_task(
                __trace_and_upload(
                    coordinates=coords,
                    trace_path=trace_path,
                    image_data=pre_capture.image,
                    action_type=step.action.action_type.value,
                    label_description=step.action.to_description(),
                )
            )
            self.__background_tasks.add(task)
            task.add_done_callback(self.__background_tasks.discard)
        except Exception as exception:
            logger.exception(f"Failed to schedule tracing: {exception}", stack_info=True)

    async def drain_background_tasks(self) -> None:
        """
        Await all pending background trace/upload tasks with a bounded timeout.
        """

        pending = [task for task in self.__background_tasks if not task.done()]
        if not pending:
            return

        logger.info(
            f"[ActionExecutor] draining {len(pending)} background tasks (timeout={DRAIN_TIMEOUT}s)"
        )

        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=DRAIN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[ActionExecutor] drain timed out, cancelling {len(pending)} remaining tasks"
            )
            for task in pending:
                if not task.done():
                    task.cancel()
