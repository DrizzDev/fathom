"""
Core execution engine implementing DAG-based execution flow.

This module contains the ExecutionEngine which orchestrates the seven-phase
execution cycle: SignalCheck → Perceive → Reason → Act → Learn → Checkpoint → Evaluate
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Optional, Tuple

from fathom.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_STABILITY_WAIT,
    DEFAULT_SWIPE_DURATION,
    ActionType,
    SignalType,
)
from fathom.core.exceptions import ExecutionError, PortError
from fathom.exceptions import ToolError
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.processing.annotator import ImageAnnotator
from fathom.schemas.actions import Action
from fathom.schemas.results import ExecutionResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step, StepResult
from fathom.utils.coordinates import CoordinateConverter


class ExecutionEngine:
    """
    Core execution engine implementing the DAG-based execution flow.

    Phases: SignalCheck → Perceive → Reason → Act → Learn → Checkpoint → Evaluate

    This engine is stateless and delegates all I/O to ports. It orchestrates
    the execution flow but doesn't own any infrastructure concerns.
    """

    def __init__(
        self,
        device: DevicePort,
        llm: LLMPort,
        memory: MemoryPort,
        signal: SignalPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        stability_wait: float = DEFAULT_STABILITY_WAIT / 1000.0,  # Convert ms to seconds
    ) -> None:
        """
        Initialize execution engine with ports.

        Args:
            device: Device port for mobile device interactions
            llm: LLM port for reasoning and analysis
            memory: Memory port for state and knowledge storage
            signal: Signal port for HITL control
            storage: Storage port for artifact persistence
            telemetry: Telemetry port for logging and observability
            max_retries: Maximum retry attempts for transient failures
            stability_wait: Wait time after action for screen stability (seconds)
        """
        self.__device = device
        self.__llm = llm
        self.__memory = memory
        self.__signal = signal
        self.__storage = storage
        self.__telemetry = telemetry
        self.__max_retries = max_retries
        self.__stability_wait = stability_wait

    async def execute_step(
        self,
        step: Step,
        *,
        pre_capture: Optional[ScreenCapture] = None,
    ) -> StepResult:
        """
        Execute one step of the execution DAG.

        Implements the seven-phase execution cycle:
        1. SignalCheck: Check for HITL control signals
        2. Perceive: Capture screen state
        3. Reason: Analyze with LLM (if needed)
        4. Act: Execute device action
        5. Learn: Store experience in memory
        6. Checkpoint: Log execution state
        7. Evaluate: Determine if terminal

        Args:
            step: Step to execute
            pre_capture: Optional pre-captured screen state

        Returns:
            StepResult with execution details
        """
        start_time = time.time()

        try:
            # Phase 1: Signal Check
            injected_context = await self.__check_signal()

            # Phase 2: Perceive (capture pre-action state)
            if pre_capture is None:
                pre_capture = await self.__perceive()

            pre_hash = self.__compute_visual_hash(capture=pre_capture)

            # Phase 3: Reason (handled by caller - LLM analysis happens before step creation)
            # This phase is implicit - the Step already contains the reasoned action
            # If context was injected, it should be passed back to caller for re-reasoning
            if injected_context:
                # Store injected context for strategy to use
                step = step.model_copy(
                    update={
                        "metadata": {**(step.metadata or {}), "injected_context": injected_context}
                    }
                )

            # Phase 4: Act
            result = await self.__act(step=step, pre_capture=pre_capture)

            # Wait for screen stability
            # Phase 2 (post-action): Perceive after stability
            await asyncio.sleep(delay=self.__stability_wait)
            post_capture = await self.__perceive()
            post_hash = self.__compute_visual_hash(capture=post_capture)

            # Phase 5: Learn
            await self.__learn(
                visual_hash=pre_hash,
                action=step.action,
                success=result.success,
            )

            # Phase 6: Checkpoint
            duration = int((time.time() - start_time) * 1000)
            screen_changed = pre_hash != post_hash

            step_result = StepResult(
                step=step,
                duration=duration,
                pre_hash=pre_hash,
                post_hash=post_hash,
                success=result.success,
                error=result.error,
                screen_changed=screen_changed,
            )

            self.__checkpoint(step_result=step_result)

            # Phase 7: Evaluate (terminal check handled by caller)
            return step_result

        except (ToolError, PortError) as exception:
            duration = int((time.time() - start_time) * 1000)
            self.__telemetry.error(
                "Step execution failed",
                step_number=step.step_number,
                error=str(exception),
            )

            return StepResult(
                step=step,
                pre_hash="",
                post_hash="",
                success=False,
                duration=duration,
                error=str(exception),
                screen_changed=False,
            )
        except Exception as exception:
            duration = int((time.time() - start_time) * 1000)
            self.__telemetry.error(
                "Unexpected error in step execution",
                step_number=step.step_number,
                error=str(exception),
            )
            raise ExecutionError(f"Step {step.step_number} failed unexpectedly") from exception

    async def __check_signal(self) -> Optional[str]:
        """
        Phase 1: Check for HITL control signals.

        Handles PAUSE, RESUME, INJECT, ASK signals.

        Returns:
            Injected context if available, None otherwise
        """
        signal = await self.__signal.check_signal()

        if signal == SignalType.PAUSE.value:
            self.__telemetry.info("Execution paused by signal")
            await self.__signal.wait_for_resume()
            self.__telemetry.info("Execution resumed")

            # Check if user injected context during pause
            if hasattr(self.__signal, "get_injected_context"):
                injected = self.__signal.get_injected_context()
                if injected:
                    self.__telemetry.info("Context injected by user", context=injected)
                    return injected

        elif signal == SignalType.INJECT.value:
            self.__telemetry.info("Injection signal received")
            # Get injected context from signal adapter
            if hasattr(self.__signal, "get_injected_context"):
                injected = self.__signal.get_injected_context()
                if injected:
                    self.__telemetry.info("Context injected", context=injected)
                    return injected

        elif signal == SignalType.ASK.value:
            self.__telemetry.info("Ask signal received")
            # Ask handling is done by strategy layer

        return None

    async def __perceive(self) -> ScreenCapture:
        """
        Phase 2: Capture current screen state via DevicePort.

        Returns:
            ScreenCapture with screenshot data
        """
        screenshot_bytes = await self.__device.capture_screen()

        # Get screen dimensions
        width, height = await self.__device.get_screen_size()

        # Get current activity
        try:
            activity = await self.__device.get_current_package()
        except Exception:
            activity = "unknown"

        return ScreenCapture(
            width=width,
            height=height,
            activity=activity,
            image=screenshot_bytes,
            timestamp=int(time.time() * 1000),
        )

    def __compute_visual_hash(self, capture: ScreenCapture) -> str:
        """
        Compute visual hash for screen capture.

        Args:
            capture: Screen capture to hash

        Returns:
            Visual hash string
        """
        return hashlib.sha256(capture.image).hexdigest()[:16]

    async def __act(self, step: Step, pre_capture: ScreenCapture) -> ExecutionResult:
        """
        Phase 4: Execute device action with retry logic and tracing.

        Args:
            step: Step containing action to execute
            pre_capture: Pre-action screen capture for tracing

        Returns:
            ExecutionResult with success status
        """
        last_error: Optional[str] = None

        for attempt in range(self.__max_retries + 1):
            try:
                result, coords = await self.__execute_action(step=step)

                # Perform tracing if successful and coordinates are available
                if result.success and coords:

                    def _do_trace(
                        img_data: bytes,
                        t_path: str,
                        a_type: str,
                        c: Tuple[int, ...],
                        label_desc: str,
                    ) -> None:
                        try:
                            ImageAnnotator.trace(
                                image_data=img_data,
                                output_path=t_path,
                                action_type=a_type,
                                coords=c,
                                label=label_desc,
                            )
                        except Exception as exception:
                            self.__telemetry.warning(f"Tracing failed: {exception}")

                    try:
                        from datetime import datetime

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"step_{step.step_number}_{step.action.action_type.value}_{timestamp}.png"
                        trace_path = f"assets/traces/{filename}"

                        # Run tracing in background thread to avoid latency
                        asyncio.create_task(
                            asyncio.to_thread(
                                _do_trace,
                                pre_capture.image,
                                trace_path,
                                step.action.action_type.value,
                                coords,
                                step.action.to_description(),
                            )
                        )
                    except Exception as exception:
                        self.__telemetry.warning(f"Failed to schedule tracing: {exception}")

                if result.success:
                    return result
                last_error = result.error

            except (ToolError, PortError) as exception:
                last_error = str(exception)
                if attempt < self.__max_retries:
                    self.__telemetry.warning(
                        "Device operation failed, retrying",
                        attempt=attempt + 1,
                        error=str(exception),
                    )

            if attempt < self.__max_retries:
                await asyncio.sleep(delay=(DEFAULT_RETRY_DELAY / 1000.0) * (attempt + 1))

        return ExecutionResult(
            success=False,
            duration=0,
            error=last_error or "Unknown error",
        )

    async def __execute_action(
        self, step: Step
    ) -> Tuple[ExecutionResult, Optional[Tuple[int, ...]]]:
        """
        Execute device action for step using CoordinateConverter.

        Args:
            step: Step containing action to execute

        Returns:
            Tuple of (ExecutionResult, optional physical coordinates)
        """
        action = step.action
        start_time = time.time()
        coords: Optional[Tuple[int, ...]] = None

        # Get screen dimensions for coordinate conversion
        screen_size = await self.__device.get_screen_size()
        width, height = screen_size

        # Use the configuration from the device if available, otherwise default
        config = self.__device.configuration
        from fathom.schemas.configuration import ADBConfig

        converter = CoordinateConverter(
            screen_width=width, screen_height=height, configuration=config or ADBConfig()
        )

        # Handle special actions that don't need device interaction
        if action.action_type == ActionType.WAIT:
            await asyncio.sleep(delay=1.0)
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(success=True, duration=duration_ms), None

        if action.action_type == ActionType.COMPLETE:
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(success=True, duration=duration_ms), None

        # Execute device actions
        try:
            device_result = None

            if action.action_type == ActionType.TAP:
                if action.bounds:
                    x, y = converter.center_to_pixels(bounds=action.bounds)
                else:
                    x, y = width // 2, height // 2
                coords = (x, y)
                device_result = await self.__device.tap(x=x, y=y)

            elif action.action_type == ActionType.TYPE:
                if not action.bounds:
                    return (
                        ExecutionResult(
                            duration=0,
                            success=False,
                            error="Type action requires bounds for focus tap guard",
                        ),
                        None,
                    )
                x, y = converter.center_to_pixels(bounds=action.bounds)
                coords = (x, y)
                focus_result = await self.__device.tap(x=x, y=y)
                if not focus_result.success:
                    return (
                        ExecutionResult(
                            duration=0,
                            success=False,
                            error=f"Focus tap failed before typing: {focus_result.error or 'unknown error'}",
                        ),
                        None,
                    )
                device_result = await self.__device.type_text(text=action.text or "")

            elif action.action_type == ActionType.SWIPE:
                if action.bounds:
                    x1, y1, x2, y2 = converter.swipe_coordinates(
                        bounds=action.bounds, direction="up"
                    )
                else:
                    cx, cy = width // 2, height // 2
                    x1, y1 = cx, cy + 300
                    x2, y2 = cx, cy - 300
                coords = (x1, y1, x2, y2)
                device_result = await self.__device.swipe(x1=x1, y1=y1, x2=x2, y2=y2)

            elif action.action_type == ActionType.SCROLL:
                cx, cy = width // 2, height // 2
                # Scroll down (swipe up)
                x1, y1 = cx, cy + 200
                x2, y2 = cx, cy - 200
                coords = (x1, y1, x2, y2)
                device_result = await self.__device.swipe(
                    x1=x1, y1=y1, x2=x2, y2=y2, duration=DEFAULT_SWIPE_DURATION
                )

            elif action.action_type == ActionType.LONG_PRESS:
                if action.bounds:
                    x, y = converter.center_to_pixels(bounds=action.bounds)
                else:
                    x, y = width // 2, height // 2
                coords = (x, y)
                device_result = await self.__device.tap(x=x, y=y)

            elif action.action_type == ActionType.BACK:
                device_result = await self.__device.back()

            elif action.action_type == ActionType.HOME:
                device_result = await self.__device.home()

            else:
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

        except PortError as exception:
            duration = int((time.time() - start_time) * 1000)
            return (
                ExecutionResult(
                    success=False,
                    duration=duration,
                    error=f"Device communication failed: {exception}",
                ),
                None,
            )
        except Exception as exception:
            duration = int((time.time() - start_time) * 1000)
            self.__telemetry.error(
                "Unexpected error executing action",
                action_type=action.action_type.value,
                error=str(exception),
            )
            raise ExecutionError(
                f"Action execution failed: {action.action_type.value}"
            ) from exception

    async def __learn(self, visual_hash: str, action: Action, success: bool) -> None:
        """
        Phase 5: Store experience in memory.

        Args:
            visual_hash: Visual hash of the screen
            action: Action that was executed
            success: Whether the action succeeded
        """
        try:
            await self.__memory.store_experience(
                visual_hash=visual_hash,
                action=action,
                success=success,
            )
        except PortError as exception:
            self.__telemetry.warning(
                "Failed to store experience",
                error=str(exception),
            )

    def __checkpoint(self, step_result: StepResult) -> None:
        """
        Phase 6: Log execution state.

        Args:
            step_result: Result of step execution
        """
        self.__telemetry.info(
            "Step completed",
            step_number=step_result.step.step_number,
            success=step_result.success,
            duration_ms=step_result.duration,
            action=step_result.step.action.action_type.value,
        )
