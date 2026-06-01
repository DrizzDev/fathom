from __future__ import annotations

import asyncio
import io
import json
import time
from logging import getLogger
from typing import Awaitable, Callable, Dict, Optional, Set, Tuple

from PIL import Image

from fathom.adapters.swipe import DeviceSwipeDispatcher
from fathom.adapters.vision import PhashVisualHasher
from fathom.base.paths import SharedPathManager
from fathom.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DRAIN_TIMEOUT,
    ActionType,
)
from fathom.constants.execution import MAX_ACTION_WAIT_MS
from fathom.constants.observation import KeyboardVisibility
from fathom.core.artifact.pipeline import ArtifactPipeline
from fathom.core.exceptions import ExecutionError, PortError, ToolError
from fathom.core.swipe import SwipeRetryCoordinator, SwipeRetryPlanner
from fathom.interfaces.device import DevicePort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.actions import (
    Action,
    Bounds,
    CoordinateSource,
    CoordinateSystem,
    ExecutionRegion,
    GesturePath,
    InputContext,
)
from fathom.schemas.artifact import ArtifactRecord, TracePayload
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.execution import PrimitiveExecution
from fathom.schemas.observation import KeyboardObservation, ScreenObservation
from fathom.schemas.results import (
    ActionResult,
    ActionTraceEvent,
    ExecutionResult,
)
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step
from fathom.schemas.swipe import SwipeExecution
from fathom.utils.coordinates import CoordinateConverter

CancelCheck = Callable[[], Awaitable[bool]]

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
        *,
        pipeline: Optional[ArtifactPipeline] = None,
        perception: Optional[PerceptionPort] = None,
        swipe_coordinator: Optional[SwipeRetryCoordinator] = None,
    ) -> None:
        self.__device = device
        self.__telemetry = telemetry
        self.__max_retries = max_retries

        self.__storage = storage
        self.__path_manager = path_manager
        self.__pipeline = pipeline
        self.__perception = perception
        self.__swipe_coordinator = swipe_coordinator or SwipeRetryCoordinator(
            planner=SwipeRetryPlanner(),
            dispatcher=DeviceSwipeDispatcher(device=device, hasher=PhashVisualHasher()),
        )
        self.__visual_hasher = PhashVisualHasher()
        self.__background_tasks: Set[asyncio.Task[None]] = set()
        self.__cached_dimensions: Optional[Tuple[int, int]] = None

    async def act(
        self,
        step: Step,
        session_id: str,
        package_name: str,
        pre_capture: ScreenCapture,
        observation: Optional[ScreenObservation] = None,
        is_cancelled: CancelCheck | None = None,
    ) -> ExecutionResult:
        """
        Execute device action with retry logic and tracing.
        """

        last_error: Optional[str] = None
        for attempt in range(self.__max_retries + 1):
            try:
                result, _ = await self.__execute_primitive(
                    step=step,
                    session_id=session_id,
                    package_name=package_name,
                    pre_capture=pre_capture,
                    observation=observation,
                    is_cancelled=is_cancelled,
                )

                if result.success:
                    return result

                last_error = result.error
                if not self.__should_retry_action(step=step):
                    return result

            except (ToolError, PortError) as exception:
                last_error = str(exception)
                if not self.__should_retry_action(step=step):
                    break

                if attempt < self.__max_retries:
                    await self.__telemetry.warning(
                        "Device operation failed, retrying",
                        attempt=attempt + 1,
                        error=str(exception),
                    )

            if attempt < self.__max_retries:
                retry_delay = (DEFAULT_RETRY_DELAY / 1000.0) * (attempt + 1)
                logger.debug(
                    "[WAIT] source=retry_backoff attempt=%s delay=%.3fs",
                    attempt + 1,
                    retry_delay,
                )
                await asyncio.sleep(delay=retry_delay)

        return ExecutionResult(
            duration=0,
            success=False,
            error=last_error or "Unknown error",
        )

    @staticmethod
    def __should_retry_action(*, step: Step) -> bool:
        """
        Return whether outer action retries should wrap this action.
        """

        return step.action.action_type not in {
            ActionType.SCROLL,
            ActionType.SWIPE_UP,
            ActionType.SWIPE_DOWN,
            ActionType.SWIPE_LEFT,
            ActionType.SWIPE_RIGHT,
        }

    async def __execute_primitive(
        self,
        *,
        step: Step,
        session_id: str,
        package_name: str,
        pre_capture: ScreenCapture,
        observation: Optional[ScreenObservation],
        is_cancelled: CancelCheck | None,
    ) -> Tuple[ExecutionResult, Optional[Tuple[int, ...]]]:
        """
        Execute specific device primitive.
        """

        action = step.action
        start_time = time.time()

        if self.__cached_dimensions is None:
            self.__cached_dimensions = await self.__device.get_dimensions()
        width, height = self.__cached_dimensions

        pixel_width, pixel_height = self.__resolve_pixel_dimensions(
            capture=pre_capture,
            logical_width=width,
            logical_height=height,
        )

        configuration = self.__device.configuration or DeviceRuntimeConfiguration()
        converter = CoordinateConverter(
            logical_width=width,
            logical_height=height,
            workflow_id=session_id,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            configuration=configuration,
        )

        if self.__is_non_interactive_action(action=action):
            return await self.__execute_non_interactive_action(action=action, start_time=start_time)

        try:
            primitive = await self.__execute_interactive_action(
                action=action,
                converter=converter,
                width=width,
                height=height,
                pre_capture=pre_capture,
                observation=observation,
                is_cancelled=is_cancelled,
                step=step,
                session_id=session_id,
                package_name=package_name,
            )
            if primitive.action is None:
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
                    error=primitive.action.error,
                    success=primitive.action.success,
                    swipe_execution=primitive.swipe_execution,
                    trace_events=primitive.trace_events,
                ),
                primitive.coords,
            )

        except Exception as exception:
            duration = int((time.time() - start_time) * 1000)
            return (
                ExecutionResult(success=False, duration=duration, error=str(exception)),
                None,
            )

    @staticmethod
    def __resolve_pixel_dimensions(
        *,
        logical_width: int,
        logical_height: int,
        capture: ScreenCapture,
    ) -> Tuple[int, int]:
        """
        Read the screenshot's actual pixel dimensions.

        :class:`ScreenCapture.width` / ``height`` carry the platform's
        logical dimensions (e.g., 430x932 on iPhone 15 Pro Max). The PNG
        bytes inside ``image`` are at the device-pixel resolution (e.g.,
        1290x2796 at 3x retina). This method decodes the PNG header to
        recover that pixel resolution so the
        :class:`CoordinateConverter` can correctly translate
        ``DEVICE_PIXEL`` bounds to logical dispatch coordinates.
        """

        if not capture.image:
            return logical_width, logical_height

        try:
            with Image.open(io.BytesIO(capture.image)) as image:
                return image.width, image.height
        except Exception:
            logger.exception(
                "Failed to decode pixel dimensions from capture; falling back to logical",
                extra={
                    "logical.width": logical_width,
                    "logical.height": logical_height,
                    "component": "core.services.action",
                    "event": "executor.pixel_dimensions.failed",
                },
            )
            return logical_width, logical_height

    def __apply_tap_bias(
        self, x: int, y: int, action: Action, converter: CoordinateConverter
    ) -> Tuple[int, int]:
        """
        Apply upward coordinate bias for VLM-detected bounds.
        Skips adjustment for label-snapped pixel bounds which are already grounded.
        """

        if not action.bounds:
            return x, y

        # Label-snapped device-pixel bounds are already grounded to exact device coordinates.
        is_label_snapped_pixel = (
            bool(action.label_id) and action.bounds.system is CoordinateSystem.DEVICE_PIXEL
        )

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
        Applies wait duration capping via MAX_ACTION_WAIT_MS to prevent
        excessively long waits requested by the model.
        """

        if action.action_type in {
            ActionType.WAIT,
            ActionType.VALIDATE,
            ActionType.SAVE_MEMORY,
            ActionType.RETRIEVE_MEMORY,
        }:
            max_wait_s = MAX_ACTION_WAIT_MS / 1000.0
            requested_wait = float(action.wait_duration or 1.0)
            applied_wait = max(0.0, min(requested_wait, max_wait_s))
            if requested_wait > max_wait_s:
                logger.warning(
                    "[WAIT] Capping wait_duration from %.1fs to %.1fs (MAX_ACTION_WAIT_MS=%d).",
                    requested_wait,
                    max_wait_s,
                    MAX_ACTION_WAIT_MS,
                )
            logger.debug(
                "[WAIT] source=model_wait_duration requested=%.3fs applied=%.3fs",
                requested_wait,
                applied_wait,
            )
            await asyncio.sleep(delay=applied_wait)

        return (
            ExecutionResult(success=True, duration=int((time.time() - start_time) * 1000)),
            None,
        )

    async def __execute_interactive_action(
        self,
        *,
        width: int,
        height: int,
        action: Action,
        converter: CoordinateConverter,
        pre_capture: ScreenCapture,
        observation: Optional[ScreenObservation],
        is_cancelled: CancelCheck | None,
        step: Step,
        session_id: str,
        package_name: str,
    ) -> PrimitiveExecution:
        """
        Execute interactive action through registered handlers.
        """

        if action.action_type.value.startswith(ActionType.SWIPE.lower()):
            return await self.__execute_swipe(
                action=action,
                converter=converter,
                pre_capture=pre_capture,
                observation=observation,
                is_cancelled=is_cancelled,
                step=step,
                session_id=session_id,
                package_name=package_name,
            )

        action_handlers: Dict[
            ActionType,
            Callable[
                [],
                Awaitable[PrimitiveExecution],
            ],
        ] = {
            ActionType.TAP: lambda: self.__execute_tap(
                action=action,
                converter=converter,
                width=width,
                height=height,
                pre_capture=pre_capture,
                step=step,
                session_id=session_id,
                package_name=package_name,
            ),
            ActionType.TYPE: lambda: self.__execute_type(
                action=action,
                converter=converter,
                width=width,
                height=height,
                pre_capture=pre_capture,
                step=step,
                session_id=session_id,
                package_name=package_name,
            ),
            ActionType.SCROLL: lambda: self.__execute_scroll(
                action=action,
                converter=converter,
                pre_capture=pre_capture,
                observation=observation,
                is_cancelled=is_cancelled,
                step=step,
                session_id=session_id,
                package_name=package_name,
            ),
            ActionType.LONG_PRESS: lambda: self.__execute_long_press(
                action=action,
                converter=converter,
                width=width,
                height=height,
                pre_capture=pre_capture,
                step=step,
                session_id=session_id,
                package_name=package_name,
            ),
            ActionType.BACK: self.__execute_back,
            ActionType.HOME: self.__execute_home,
            ActionType.ENTER: self.__execute_enter,
            ActionType.HIDE_KEYBOARD: self.__execute_hide_keyboard,
        }

        handler = action_handlers.get(action.action_type)
        if handler is None:
            return PrimitiveExecution(action=None, coords=None, swipe_execution=None)

        return await handler()

    async def __execute_back(
        self,
    ) -> PrimitiveExecution:
        """
        Execute back action.
        Adapters that cannot dispatch a system back (e.g. iOS, which has no OS-level back gesture) raise :class:`NotImplementedError`
        """

        try:
            result = await self.__device.back()
        except NotImplementedError as exception:
            result = ActionResult(
                duration=0,
                success=False,
                error=f"Device does not support back action: {exception}",
            )

        return PrimitiveExecution(
            coords=None,
            action=result,
            swipe_execution=None,
        )

    async def __execute_home(
        self,
    ) -> PrimitiveExecution:
        """
        Execute home action.
        """

        return PrimitiveExecution(
            coords=None,
            swipe_execution=None,
            action=await self.__device.home(),
        )

    async def __execute_enter(
        self,
    ) -> PrimitiveExecution:
        """
        Execute the keyboard enter/search primitive when the provider supports it.
        """

        if not hasattr(self.__device, "enter"):
            return PrimitiveExecution(
                action=ActionResult(
                    duration=0,
                    success=False,
                    error="Device does not support enter action",
                ),
                coords=None,
                swipe_execution=None,
            )

        return PrimitiveExecution(
            coords=None,
            swipe_execution=None,
            action=await self.__device.enter(),
        )

    async def __execute_hide_keyboard(
        self,
    ) -> PrimitiveExecution:
        """
        Execute platform-neutral keyboard dismissal.
        """

        if hasattr(self.__device, "hide_keyboard"):
            return PrimitiveExecution(
                coords=None,
                swipe_execution=None,
                action=await self.__device.hide_keyboard(),
            )

        try:
            result = await self.__device.back()
        except NotImplementedError as exception:
            result = ActionResult(
                duration=0,
                success=False,
                error=f"Cannot hide keyboard: device does not support back fallback: {exception}",
            )

        return PrimitiveExecution(
            coords=None,
            action=result,
            swipe_execution=None,
        )

    async def __execute_tap(
        self,
        step: Step,
        width: int,
        height: int,
        action: Action,
        session_id: str,
        package_name: str,
        pre_capture: ScreenCapture,
        converter: CoordinateConverter,
    ) -> PrimitiveExecution:
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
        trace_event = ActionTraceEvent(capture=pre_capture, coords=coords)

        await self.__emit_trace_event(
            step=step,
            action=action,
            event=trace_event,
            session_id=session_id,
            package_name=package_name,
        )

        return PrimitiveExecution(
            action=result,
            coords=coords,
            swipe_execution=None,
            trace_events=(trace_event,),
        )

    async def __execute_type(
        self,
        step: Step,
        width: int,
        height: int,
        action: Action,
        session_id: str,
        package_name: str,
        pre_capture: ScreenCapture,
        converter: CoordinateConverter,
    ) -> PrimitiveExecution:
        """
        Execute TYPE command: focus tap, stabilization wait, then type.

        When ``input_context`` is present on the action (populated during resolution),
        the locator and prefilled text are forwarded to the provider. When absent,
        the provider receives a plain text input with no clear or locator fallback.
        """

        _ = width, height

        if not action.bounds:
            raise ExecutionError("Type action requires bounds for focus tap guard")

        x, y = converter.center_to_pixels(bounds=action.bounds)
        x, y = self.__apply_tap_bias(x=x, y=y, action=action, converter=converter)

        coords = (x, y)
        context = action.input_context or InputContext()
        configuration = self.__device.configuration or DeviceRuntimeConfiguration()

        wait = configuration.interaction.policy.type.delay / 1000.0

        if len(context.prefilled) > 0:
            logger.info("Existing text detected, will replace (locator=%s)", context.locator)

        if not (
            result := await self.__focus_and_type(
                text=action.text or "", context=context, x=x, y=y, wait=wait
            )
        ).success:
            logger.warning("Type failed (error=%s). Re-tapping and retrying.", result.error)
            result = await self.__focus_and_type(
                text=action.text or "", context=context, x=x, y=y, wait=wait
            )

        trace_event = ActionTraceEvent(capture=pre_capture, coords=coords)
        await self.__emit_trace_event(
            step=step,
            action=action,
            event=trace_event,
            session_id=session_id,
            package_name=package_name,
        )

        return PrimitiveExecution(
            action=result,
            coords=coords,
            swipe_execution=None,
            trace_events=(trace_event,),
        )

    async def __focus_and_type(
        self, *, text: str, context: InputContext, x: int, y: int, wait: float
    ) -> ActionResult:
        """
        Tap to focus, wait for stabilization, then send text.
        """

        if not (result := await self.__device.tap(x=x, y=y)).success:
            return result

        logger.info(f"Waiting for {wait} seconds since element was not focused")
        await asyncio.sleep(delay=wait)

        return await self.__device.type(
            text=text,
            locator=context.locator,
            prefilled=context.prefilled,
            replace=len(context.prefilled) > 0,
        )

    async def __execute_swipe(
        self,
        *,
        step: Step,
        action: Action,
        session_id: str,
        package_name: str,
        pre_capture: ScreenCapture,
        converter: CoordinateConverter,
        is_cancelled: Optional[CancelCheck],
        observation: Optional[ScreenObservation],
    ) -> PrimitiveExecution:
        """
        Dispatch one logical swipe through the keyboard-aware retry coordinator.
        """

        _ = is_cancelled

        direction = (
            action.action_type.value.split("_")[-1] if "_" in action.action_type.value else "up"
        )
        region = (
            converter.region_from_bounds(
                bounds=action.bounds,
                source=action.bounds.source or CoordinateSource.MODEL,
            )
            if action.bounds
            else converter.viewport_region()
        )
        path = converter.resolve_swipe_path(region=region, direction=direction)
        self.__log_gesture_path(
            path=path,
            kind="swipe",
            region=region,
            action=action,
            direction=direction,
        )

        return await self.__coordinate_and_emit(
            step=step,
            path=path,
            action=action,
            region=region,
            session_id=session_id,
            pre_capture=pre_capture,
            observation=observation,
            package_name=package_name,
        )

    async def __execute_scroll(
        self,
        *,
        step: Step,
        action: Action,
        session_id: str,
        package_name: str,
        pre_capture: ScreenCapture,
        converter: CoordinateConverter,
        is_cancelled: Optional[CancelCheck],
        observation: Optional[ScreenObservation],
    ) -> PrimitiveExecution:
        """
        Dispatch the downward scroll through the keyboard-aware retry coordinator.
        Resolved ``action.bounds`` confines the gesture to that region (mirrors SWIPE); otherwise viewport is used.
        """

        _ = is_cancelled

        keyboard = self.__keyboard_observation(observation=observation)

        if keyboard.visibility is KeyboardVisibility.VISIBLE:
            logger.warning(
                "Scroll dispatch blocked by visible keyboard",
                extra={
                    "workflow.id": session_id,
                    "package.name": package_name,
                    "action.target": action.target,
                    "step.number": step.step_number,
                    "event": "scroll.dispatch.blocked",
                    "component": "core.services.action",
                    "action.type": action.action_type.value,
                    "keyboard.visibility": keyboard.visibility.value,
                    "keyboard.bounds": (
                        keyboard.bounds.model_dump(mode="json") if keyboard.bounds else None
                    ),
                },
            )
            return PrimitiveExecution(
                action=ActionResult(
                    duration=0,
                    success=False,
                    error="scroll blocked by visible keyboard",
                ),
                coords=None,
                swipe_execution=None,
            )

        region = self.__scroll_region(action=action, converter=converter)
        path = converter.resolve_scroll_path(region=region, direction="down")

        logger.info(
            "Scroll dispatch allowed",
            extra={
                "event": "scroll.dispatch.allowed",
                "component": "core.services.action",
                "workflow.id": session_id,
                "package.name": package_name,
                "action.target": action.target,
                "step.number": step.step_number,
                "path": path.model_dump(mode="json"),
                "region.source": region.source.value,
                "action.type": action.action_type.value,
                "region": region.model_dump(mode="json"),
                "action.has_bounds": action.bounds is not None,
                "keyboard.visibility": keyboard.visibility.value,
            },
        )
        self.__log_gesture_path(
            path=path,
            action=action,
            kind="scroll",
            region=region,
            direction="down",
        )

        return await self.__coordinate_and_emit(
            step=step,
            path=path,
            action=action,
            region=region,
            session_id=session_id,
            pre_capture=pre_capture,
            observation=observation,
            package_name=package_name,
        )

    @staticmethod
    def __scroll_region(*, action: Action, converter: CoordinateConverter) -> ExecutionRegion:
        """
        Return the scroll execution region, preferring resolved action bounds.
        """

        if action.bounds is None:
            return converter.viewport_region()

        return converter.region_from_bounds(
            bounds=action.bounds,
            source=action.bounds.source or CoordinateSource.MODEL,
        )

    async def __coordinate_and_emit(
        self,
        *,
        step: Step,
        action: Action,
        session_id: str,
        package_name: str,
        path: GesturePath,
        region: ExecutionRegion,
        pre_capture: ScreenCapture,
        observation: Optional[ScreenObservation],
    ) -> PrimitiveExecution:
        """
        Run the swipe coordinator and emit one trace event per dispatched attempt.
        """

        original_before = self.__hash_capture(capture=pre_capture)
        keyboard = self.__keyboard_observation(observation=observation)
        viewport_bounds = self.__viewport_bounds(region=region, capture=pre_capture)
        policy = self.__device_configuration().interaction.policy.swipe.retry

        execution = await self.__swipe_coordinator.execute(
            original=path,
            policy=policy,
            keyboard=keyboard,
            bounds=viewport_bounds,
            original_before=original_before,
        )

        trace_events = await self.__emit_attempt_traces(
            step=step,
            action=action,
            execution=execution,
            session_id=session_id,
            pre_capture=pre_capture,
            package_name=package_name,
        )

        final_path = execution.final or path
        coords = final_path.to_coordinates()
        action_result = self.__action_result_from_execution(execution=execution)

        return PrimitiveExecution(
            coords=coords,
            action=action_result,
            swipe_execution=execution,
            trace_events=trace_events,
        )

    def __hash_capture(self, *, capture: ScreenCapture) -> str:
        """
        Compute the visual hash of the original pre-action capture for retry comparison.
        """

        try:
            return self.__visual_hasher.hash(image=capture.image or b"")
        except Exception as exception:
            logger.warning(
                f"ActionExecutor: pre-action hash failed ({exception}); using empty marker."
            )
            return ""

    @staticmethod
    def __keyboard_observation(*, observation: Optional[ScreenObservation]) -> KeyboardObservation:
        """
        Return the keyboard observation carried by the supervisor, or an UNKNOWN sentinel.
        """

        if observation is not None and observation.keyboard is not None:
            return observation.keyboard

        return KeyboardObservation(visibility=KeyboardVisibility.UNKNOWN)

    @staticmethod
    def __viewport_bounds(*, region: ExecutionRegion, capture: ScreenCapture) -> Bounds:
        """
        Build a viewport rectangle that covers the entire capture in the gesture's pixel space.
        """

        pixel_width, pixel_height = ActionExecutor.__resolve_pixel_dimensions(
            capture=capture,
            logical_width=region.x + region.width,
            logical_height=region.y + region.height,
        )
        width = max(1, max(pixel_width, region.x + region.width))
        height = max(1, max(pixel_height, region.y + region.height))

        return Bounds(x=0, y=0, width=width, height=height)

    @staticmethod
    def __action_result_from_execution(*, execution: SwipeExecution) -> ActionResult:
        """
        Reduce the bounded swipe execution to one device-level ActionResult.
        """

        if execution.effective:
            last = execution.attempts[-1]
            duration = last.path.duration

            return ActionResult(success=True, duration=duration)

        if execution.attempts:
            last = execution.attempts[-1]
            error = last.device.error or (
                execution.aborted_for.value
                if execution.aborted_for
                else "swipe produced no visual change"
            )
            return ActionResult(success=False, duration=last.path.duration, error=error)

        error = (
            execution.aborted_for.value
            if execution.aborted_for
            else "swipe execution produced no attempts"
        )
        return ActionResult(success=False, duration=0, error=error)

    async def __emit_attempt_traces(
        self,
        *,
        step: Step,
        action: Action,
        session_id: str,
        package_name: str,
        execution: SwipeExecution,
        pre_capture: ScreenCapture,
    ) -> Tuple[ActionTraceEvent, ...]:
        """
        Emit one ArtifactRecord trace event per dispatched attempt and return them in order.
        """

        events: list[ActionTraceEvent] = []
        for attempt in execution.attempts:
            event = ActionTraceEvent(
                capture=pre_capture,
                coords=attempt.path.to_coordinates(),
            )
            await self.__emit_trace_event(
                step=step,
                event=event,
                action=action,
                session_id=session_id,
                package_name=package_name,
            )
            events.append(event)

        return tuple(events)

    @staticmethod
    def __log_gesture_path(
        *,
        kind: str,
        direction: str,
        path: GesturePath,
        region: ExecutionRegion,
        action: Optional[Action],
    ) -> None:
        """
        Log the executable gesture path with compact routing context.
        """

        target = (action.natural_language_target or action.target) if action else "viewport"

        path_payload = path.model_dump()
        path_payload["distance"] = path.distance

        logger.info(
            json.dumps(
                {
                    "kind": kind,
                    "target": target,
                    "path": path_payload,
                    "direction": direction,
                    "component": "ActionExecutor",
                    "region": region.model_dump(),
                    "event": "gesture.path.resolved",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def __device_configuration(self) -> DeviceRuntimeConfiguration:
        """
        Return the active runtime configuration.
        """

        return self.__device.configuration or DeviceRuntimeConfiguration()

    async def __execute_long_press(
        self,
        step: Step,
        width: int,
        height: int,
        action: Action,
        session_id: str,
        package_name: str,
        pre_capture: ScreenCapture,
        converter: CoordinateConverter,
    ) -> PrimitiveExecution:
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
        trace_event = ActionTraceEvent(capture=pre_capture, coords=coords)

        await self.__emit_trace_event(
            step=step,
            action=action,
            event=trace_event,
            session_id=session_id,
            package_name=package_name,
        )
        return PrimitiveExecution(
            coords=coords,
            swipe_execution=None,
            action=long_press_result,
            trace_events=(trace_event,),
        )

    async def __emit_trace_event(
        self,
        *,
        step: Step,
        action: Action,
        session_id: str,
        package_name: str,
        event: ActionTraceEvent,
    ) -> None:
        """
        Emit one concrete trace artifact immediately after device dispatch.
        """

        if self.__pipeline is None:
            return

        await self.__pipeline.emit(
            record=ArtifactRecord(
                session_id=session_id,
                package_name=package_name,
                step_number=step.step_number,
                created=int(time.time() * 1000),
                payload=TracePayload(
                    action=action,
                    coords=event.coords,
                    capture=event.capture,
                    attempt=event.attempt,
                ),
            ),
        )

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
