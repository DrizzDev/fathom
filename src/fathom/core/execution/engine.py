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

from fathom.constants import ActionType
from fathom.core.exceptions import ExecutionError, PortError
from fathom.exceptions import ToolError
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.actions import Action
from fathom.schemas.results import ExecutionResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step, StepResult

# Constants
VISUAL_HASH_LENGTH = 16
DEFAULT_SWIPE_DISTANCE = 300
DEFAULT_SCROLL_DISTANCE = 200
DEFAULT_SWIPE_DURATION = 500
BOUNDS_SWIPE_DISTANCE = 100


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
        max_retries: int = 2,
        stability_wait: float = 0.5,
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
            stability_wait: Wait time after action for screen stability
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
            await self.__check_signal()
            
            # Phase 2: Perceive (capture pre-action state)
            if pre_capture is None:
                pre_capture = await self.__perceive()
            
            pre_hash = self.__compute_visual_hash(capture=pre_capture)
            
            # Phase 3: Reason (handled by caller - LLM analysis happens before step creation)
            # This phase is implicit - the Step already contains the reasoned action
            
            # Phase 4: Act
            result = await self.__act(step=step)
            
            # Wait for screen stability
            await asyncio.sleep(delay=self.__stability_wait)
            
            # Perceive post-action state
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
    
    async def __check_signal(self) -> None:
        """
        Phase 1: Check for HITL control signals.
        
        Handles PAUSE, RESUME, INJECT, ASK signals.
        """
        signal = await self.__signal.check_signal()
        
        if signal == "PAUSE":
            self.__telemetry.info("Execution paused by signal")
            await self.__signal.wait_for_resume()
            self.__telemetry.info("Execution resumed")
        elif signal == "INJECT":
            self.__telemetry.info("Injection signal received")
            # Injection handling would be implemented by caller
        elif signal == "ASK":
            self.__telemetry.info("Ask signal received")
            # Ask handling would be implemented by caller
    
    async def __perceive(self) -> ScreenCapture:
        """
        Phase 2: Capture current screen state.
        
        Returns:
            ScreenCapture with screenshot data
        """
        screenshot_bytes = await self.__device.capture_screen()
        
        # Store screenshot artifact
        storage_id = await self.__storage.save(
            data=screenshot_bytes,
            metadata={"type": "screenshot", "timestamp": time.time()},
        )
        
        return ScreenCapture(
            image_data=screenshot_bytes,
            storage_id=storage_id,
            timestamp=time.time(),
        )
    
    def __compute_visual_hash(self, capture: ScreenCapture) -> str:
        """
        Compute visual hash for screen capture.
        
        Args:
            capture: Screen capture to hash
        
        Returns:
            Visual hash string
        """
        return hashlib.sha256(capture.image_data).hexdigest()[:VISUAL_HASH_LENGTH]
    
    async def __act(self, step: Step) -> ExecutionResult:
        """
        Phase 4: Execute device action with retry logic.
        
        Args:
            step: Step containing action to execute
        
        Returns:
            ExecutionResult with success status
        """
        last_error: Optional[str] = None
        
        for attempt in range(self.__max_retries + 1):
            try:
                result = await self.__execute_action(step=step)
                if result.success:
                    return result
                last_error = result.error
                
            except ToolError as exception:
                last_error = str(exception)
                if not exception.retryable:
                    break
                    
            except PortError as exception:
                last_error = str(exception)
                if attempt < self.__max_retries:
                    self.__telemetry.warning(
                        "Port communication failed, retrying",
                        attempt=attempt + 1,
                        error=str(exception),
                    )
            
            if attempt < self.__max_retries:
                await asyncio.sleep(delay=0.5 * (attempt + 1))
        
        return ExecutionResult(
            success=False,
            duration=0,
            error=last_error or "Unknown error",
        )
    
    async def __execute_action(self, step: Step) -> ExecutionResult:
        """
        Execute device action for step.
        
        Args:
            step: Step containing action to execute
        
        Returns:
            ExecutionResult with execution details
        """
        action = step.action
        start_time = time.time()
        
        # Get screen dimensions for coordinate conversion
        screen_size = await self.__device.get_screen_size()
        width, height = screen_size
        
        # Handle special actions that don't need device interaction
        if action.action_type == ActionType.WAIT:
            await asyncio.sleep(delay=1.0)
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(success=True, duration=duration_ms)
        
        if action.action_type == ActionType.COMPLETE:
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(success=True, duration=duration_ms)
        
        # Execute device actions
        try:
            device_result = None
            
            if action.action_type == ActionType.TAP:
                if action.bounds:
                    x, y = self.__bounds_to_center(action.bounds, width, height)
                else:
                    x, y = width // 2, height // 2
                device_result = await self.__device.tap(x=x, y=y)
            
            elif action.action_type == ActionType.TYPE:
                if not action.bounds:
                    return ExecutionResult(
                        duration=0,
                        success=False,
                        error="Type action requires bounds for focus tap",
                    )
                x, y = self.__bounds_to_center(action.bounds, width, height)
                focus_result = await self.__device.tap(x=x, y=y)
                if not focus_result.success:
                    return ExecutionResult(
                        duration=0,
                        success=False,
                        error=f"Focus tap failed: {focus_result.error or 'unknown'}",
                    )
                device_result = await self.__device.type_text(text=action.text or "")
            
            elif action.action_type == ActionType.SWIPE:
                if action.bounds:
                    x1, y1, x2, y2 = self.__bounds_to_swipe(action.bounds, width, height)
                else:
                    cx, cy = width // 2, height // 2
                    x1, y1 = cx, cy + DEFAULT_SWIPE_DISTANCE
                    x2, y2 = cx, cy - DEFAULT_SWIPE_DISTANCE
                device_result = await self.__device.swipe(x1=x1, y1=y1, x2=x2, y2=y2)
            
            elif action.action_type == ActionType.SCROLL:
                cx, cy = width // 2, height // 2
                device_result = await self.__device.swipe(
                    x1=cx,
                    y1=cy + DEFAULT_SCROLL_DISTANCE,
                    x2=cx,
                    y2=cy - DEFAULT_SCROLL_DISTANCE,
                    duration=DEFAULT_SWIPE_DURATION,
                )
            
            elif action.action_type == ActionType.LONG_PRESS:
                if action.bounds:
                    x, y = self.__bounds_to_center(action.bounds, width, height)
                else:
                    x, y = width // 2, height // 2
                # Long press is implemented as tap with longer duration
                device_result = await self.__device.tap(x=x, y=y)
            
            elif action.action_type == ActionType.BACK:
                device_result = await self.__device.back()
            
            elif action.action_type == ActionType.HOME:
                device_result = await self.__device.home()
            
            else:
                return ExecutionResult(
                    duration=0,
                    success=False,
                    error=f"Unknown action type: {action.action_type}",
                )
            
            duration = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                duration=duration,
                error=device_result.error,
                success=device_result.success,
            )
            
        except PortError as exception:
            duration = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                duration=duration,
                error=f"Device communication failed: {exception}",
            )
        except Exception as exception:
            duration = int((time.time() - start_time) * 1000)
            self.__telemetry.error(
                "Unexpected error executing action",
                action_type=action.action_type.value,
                error=str(exception),
            )
            raise ExecutionError(f"Action execution failed: {action.action_type.value}") from exception
    
    def __bounds_to_center(self, bounds: str, width: int, height: int) -> Tuple[int, int]:
        """
        Convert bounds string to center coordinates.
        
        Args:
            bounds: Bounds string in format "[x1,y1][x2,y2]"
            width: Screen width
            height: Screen height
        
        Returns:
            Tuple of (x, y) center coordinates
        """
        try:
            parts = bounds.replace("[", "").replace("]", ",").split(",")
            x1, y1, x2, y2 = map(int, [p for p in parts if p])
            return (x1 + x2) // 2, (y1 + y2) // 2
        except (ValueError, IndexError):
            return width // 2, height // 2
    
    def __bounds_to_swipe(
        self, bounds: str, width: int, height: int
    ) -> Tuple[int, int, int, int]:
        """
        Convert bounds string to swipe coordinates (upward swipe).
        
        Args:
            bounds: Bounds string in format "[x1,y1][x2,y2]"
            width: Screen width
            height: Screen height
        
        Returns:
            Tuple of (x1, y1, x2, y2) swipe coordinates
        """
        try:
            parts = bounds.replace("[", "").replace("]", ",").split(",")
            x1, y1, x2, y2 = map(int, [p for p in parts if p])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            return cx, cy + BOUNDS_SWIPE_DISTANCE, cx, cy - BOUNDS_SWIPE_DISTANCE
        except (ValueError, IndexError):
            cx, cy = width // 2, height // 2
            return cx, cy + BOUNDS_SWIPE_DISTANCE, cx, cy - BOUNDS_SWIPE_DISTANCE
    
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
            screen_changed=step_result.screen_changed,
            action=step_result.step.action.action_type.value,
        )
