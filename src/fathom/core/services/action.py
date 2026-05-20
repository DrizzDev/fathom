from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Awaitable, Callable, Optional, Set, Tuple

from fathom.base.paths import SharedPathManager
from fathom.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DRAIN_TIMEOUT,
    ActionType,
)
from fathom.constants.execution import MAX_ACTION_WAIT_MS
from fathom.constants.storage import StorageBackend
from fathom.core.exceptions import ExecutionError, PortError, ToolError
from fathom.interfaces.device import DevicePort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.processing.annotator import ImageAnnotator
from fathom.schemas.actions import Action, ExecutionRegion, GesturePath, InputContext
from fathom.schemas.artifacts import ScreenArtifact
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
        self.__cached_dimensions: Optional[Tuple[int, int]] = None

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
                result, coords, bounds_px = await self.__execute_primitive(step=step)

                if result.success:
                    trace_artifact: Optional[ScreenArtifact] = None
                    if coords:
                        trace_artifact = await self.__capture_trace(
                            step=step,
                            coords=coords,
                            bounds_px=bounds_px,
                            session_id=session_id,
                            pre_capture=pre_capture,
                            package_name=package_name,
                        )
                    # ExecutionResult is frozen — use model_copy to attach the trace.
                    return result.model_copy(update={"trace_artifact": trace_artifact})

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

    async def __execute_primitive(
        self, step: Step
    ) -> Tuple[ExecutionResult, Optional[Tuple[int, ...]], Optional[Tuple[int, int, int, int]]]:
        """
        Execute specific device primitive.

        Returns ``(result, coords, bounds_px)`` where ``bounds_px`` is the
        target element's pixel rectangle ``(x, y, width, height)`` when the
        action targets a labeled element (so the trace can draw a bounding
        box around the element, matching the orange-rectangle style normal
        steps render). Returns ``None`` for coord-only actions (grid taps,
        scroll/swipe gestures) — those fall back to the trace's coord-
        centered indicator.
        """

        action = step.action
        start_time = time.time()

        if self.__cached_dimensions is None:
            self.__cached_dimensions = await self.__device.get_dimensions()
        width, height = self.__cached_dimensions

        configuration = self.__device.configuration or DeviceRuntimeConfiguration()
        converter = CoordinateConverter(
            screen_width=width, screen_height=height, configuration=configuration
        )

        # Resolve element bounds → pixel rect once, so every return path below
        # carries the same value. Conversion failure is non-fatal: we just lose
        # the rectangle style and fall back to a circle in the trace.
        bounds_px: Optional[Tuple[int, int, int, int]] = None
        if action.bounds:
            try:
                bounds_px = converter.to_pixels(bounds=action.bounds)
            except Exception:
                bounds_px = None

        if self.__is_non_interactive_action(action=action):
            result, coords = await self.__execute_non_interactive_action(
                action=action, start_time=start_time
            )
            return result, coords, bounds_px

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
                    bounds_px,
                )

            duration = int((time.time() - start_time) * 1000)
            return (
                ExecutionResult(
                    duration=duration,
                    error=device_result.error,
                    success=device_result.success,
                ),
                coords,
                bounds_px,
            )

        except Exception as exception:
            duration = int((time.time() - start_time) * 1000)
            return (
                ExecutionResult(success=False, duration=duration, error=str(exception)),
                None,
                bounds_px,
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
        action: Action,
        converter: CoordinateConverter,
        width: int,
        height: int,
    ) -> Tuple[Optional[ActionResult], Optional[Tuple[int, ...]]]:
        """
        Execute interactive action through registered handlers.
        """

        if action.action_type.value.startswith(ActionType.SWIPE.lower()):
            return await self.__execute_swipe(action=action, converter=converter)

        action_handlers: dict[
            ActionType,
            Callable[[], Awaitable[Tuple[ActionResult, Optional[Tuple[int, ...]]]]],
        ] = {
            ActionType.TAP: lambda: self.__execute_tap(
                action=action, converter=converter, width=width, height=height
            ),
            ActionType.TYPE: lambda: self.__execute_type(
                action=action, converter=converter, width=width, height=height
            ),
            ActionType.SCROLL: lambda: self.__execute_scroll(converter=converter),
            ActionType.LONG_PRESS: lambda: self.__execute_long_press(
                action=action, converter=converter, width=width, height=height
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
        self,
        width: int,
        height: int,
        action: Action,
        converter: CoordinateConverter,
    ) -> Tuple[ActionResult, Tuple[int, ...]]:
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

        return result, coords

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
        self, *, action: Action, converter: CoordinateConverter
    ) -> Tuple[ActionResult, Tuple[int, ...]]:
        """
        Helper Method To Execute `SWIPE` Command
        """

        if "_" in action.action_type.value:
            direction = action.action_type.value.split("_")[-1]
        else:
            direction = "up"

        if action.bounds:
            region = converter.region_from_bounds(
                bounds=action.bounds,
                source=action.bounds.source or "model",
            )
        else:
            region = converter.viewport_region()

        path = converter.resolve_swipe_path(region=region, direction=direction)
        coords = path.to_coordinates()

        self.__log_gesture_path(
            path=path,
            kind="swipe",
            region=region,
            action=action,
            direction=direction,
        )

        result = await self.__device.swipe(
            x1=path.start_x,
            y1=path.start_y,
            x2=path.end_x,
            y2=path.end_y,
            duration=path.duration,
        )
        return result, coords

    async def __execute_scroll(
        self, *, converter: CoordinateConverter
    ) -> Tuple[ActionResult, Tuple[int, ...]]:
        """
        Helper Method To Execute `SCROLL` Command (Default Scroll Down)
        """

        region = converter.viewport_region()
        path = converter.resolve_scroll_path(region=region, direction="down")

        coords = path.to_coordinates()
        self.__log_gesture_path(
            path=path,
            action=None,
            kind="scroll",
            region=region,
            direction="down",
        )
        result = await self.__device.swipe(
            x1=path.start_x,
            y1=path.start_y,
            x2=path.end_x,
            y2=path.end_y,
            duration=path.duration,
        )
        return result, coords

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
                    "event": "gesture_path_resolved",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )

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

    async def __capture_trace(
        self,
        step: Step,
        session_id: str,
        package_name: str,
        coords: Tuple[int, ...],
        pre_capture: ScreenCapture,
        bounds_px: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[ScreenArtifact]:
        """
        Annotate the pre-action screen with the action target and upload it
        via the storage adapter, returning a ``ScreenArtifact`` for embedding
        in the step result.

        Visual style — ``ImageAnnotator.trace`` draws an orange rectangle
        around ``bounds_px`` when the action targets a labeled element (so
        healing matches the regular step's bounding-box style); otherwise it
        falls back to a coord-centered red circle. Swipe/scroll keep the
        arrow indicator regardless of bounds.

        Returns ``None`` when no storage adapter is configured, or when any
        step (drawing, file write, upload) fails — callers treat None as
        "no trace available" rather than an error.
        """
        if not self.__storage:
            return None

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

            # Drawing + disk write are CPU/IO bound; offload to a thread so
            # we don't block the event loop.
            annotated_path = await asyncio.to_thread(
                ImageAnnotator.trace,
                coords=coords,
                image_data=pre_capture.image,
                output_path=trace_path,
                action_type=step.action.action_type.value,
                label=step.action.to_description(),
                bounds=bounds_px,
            )

            # ImageAnnotator.trace returns None on failure (PIL decode error,
            # font load failure, etc.) — it has already logged the cause and
            # no file was written. Short-circuit so we don't raise a
            # misleading FileNotFoundError from read_bytes() below and
            # double-log the same underlying failure.
            if annotated_path is None:
                return None

            data = await asyncio.to_thread(Path(trace_path).read_bytes)
            uri = await self.__storage.save(
                data=data,
                metadata={
                    "category": "traces",
                    "filename": Path(trace_path).name,
                    "session_id": session_id,
                    "package_name": package_name,
                },
            )

            return ScreenArtifact(
                uri=uri,
                storage_backend=self.__resolve_storage_backend(),
                captured_at=int(time.time() * 1000),
                mime_type="image/png",
            )

        except Exception as exception:
            logger.exception(f"Tracing failed: {exception}", stack_info=True)
            return None

    def __resolve_storage_backend(self) -> StorageBackend:
        """Read the adapter's declared ``backend`` attribute, falling back to ``LOCAL``.

        Mirrors the same helper in ``intent/nodes.py`` — kept local to avoid
        a service-layer→strategy-layer import.
        """
        backend = getattr(self.__storage, "backend", None)
        if isinstance(backend, StorageBackend):
            return backend
        return StorageBackend.LOCAL

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
