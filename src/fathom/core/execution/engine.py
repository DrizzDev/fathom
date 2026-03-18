from __future__ import annotations

import asyncio
import time
from typing import Optional

from fathom.base.paths import SharedPathManager
from fathom.constants import DEFAULT_MAX_RETRIES, DEFAULT_STABILITY_WAIT, SignalType
from fathom.constants.screen import ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD
from fathom.core.exceptions import ExecutionError, PortError, ToolError
from fathom.core.services.action import ActionExecutor
from fathom.core.services.perception import PerceptionService
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.processing.parsers.signature import HierarchySignatureBuilder
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult


class ExecutionEngine:
    """
    Core execution engine implementing the DAG-based execution flow.

    Phases: SignalCheck → Perceive → Reason → Act → Learn → Checkpoint → Evaluate

    This engine is stateless and delegates all I/O to ports. It orchestrates
    the execution flow but doesn't own any infrastructure concerns.
    """

    def __init__(
        self,
        llm: LLMPort,
        device: DevicePort,
        perception: PerceptionPort,
        memory: MemoryPort,
        signal: SignalPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        path_manager: SharedPathManager,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        stability_wait: float = DEFAULT_STABILITY_WAIT / 1000.0,  # Convert ms to seconds
    ) -> None:
        self.__llm = llm
        self.__device = device
        self.__memory = memory
        self.__signal = signal
        self.__storage = storage
        self.__telemetry = telemetry
        self.__path_manager = path_manager

        self.__max_retries = max_retries
        self.__stability_wait = stability_wait

        # Initialize Domain Services
        self.__perception_service = PerceptionService(
            storage=storage,
            perception=perception,
            hierarchy_signature_builder=HierarchySignatureBuilder(),
        )
        self.__action_executor = ActionExecutor(
            device=device,
            telemetry=telemetry,
            max_retries=max_retries,
            path_manager=path_manager,
        )

    async def execute_step(
        self,
        step: Step,
        *,
        session_id: str,
        package_name: str = "unknown",
        pre_capture: Optional[ScreenCapture] = None,
    ) -> StepResult:
        """
        Execute one step of the execution DAG.
        """

        start_time = time.time()

        try:
            # Phase 1: Signal Check
            injected_context = await self.__check_signal()

            # Phase 2: Perceive (capture pre-action state)
            if pre_capture is None:
                pre_capture = await self.__perception_service.perceive(session_id=session_id)

            pre_hash = self.__perception_service.compute_visual_hash(capture=pre_capture)

            # Phase 3: Reason (Implicit)
            if injected_context:
                step = step.model_copy(
                    update={
                        "metadata": {**(step.metadata or {}), "injected_context": injected_context}
                    }
                )

            # Phase 4: Act
            result = await self.__action_executor.act(
                step=step,
                session_id=session_id,
                pre_capture=pre_capture,
                package_name=package_name,
            )

            # Wait for screen stability
            await asyncio.sleep(delay=self.__stability_wait)
            post_capture = await self.__perception_service.perceive(session_id=session_id)
            post_hash = self.__perception_service.compute_visual_hash(capture=post_capture)

            # Phase 5: Learn
            await self.__learn(
                action=step.action,
                visual_hash=pre_hash,
                success=result.success,
            )

            # Phase 6: Checkpoint
            duration = int((time.time() - start_time) * 1000)
            screen_changed = (
                ScreenState.hamming_distance(
                    left_hash=pre_hash,
                    right_hash=post_hash,
                )
                > ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD
            )

            step_result = StepResult(
                step=step,
                duration=duration,
                pre_hash=pre_hash,
                error=result.error,
                post_hash=post_hash,
                success=result.success,
                screen_changed=screen_changed,
            )

            await self.__checkpoint(step_result=step_result)

            return step_result

        except (ToolError, PortError) as exception:
            duration = int((time.time() - start_time) * 1000)
            await self.__telemetry.error(
                "Step execution failed", error=str(exception), step_number=step.step_number
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
            await self.__telemetry.error(
                "Unexpected error in step execution",
                error=str(exception),
                step_number=step.step_number,
            )
            raise ExecutionError(f"Step {step.step_number} failed unexpectedly") from exception

    async def __check_signal(self) -> Optional[str]:
        """
        Phase 1: Check for HITL control signals.
        """

        signal = await self.__signal.check_signal()

        if signal == SignalType.PAUSE.value:
            await self.__telemetry.info("Execution paused by signal")

            await self.__signal.wait_for_resume()
            await self.__telemetry.info("Execution resumed")

            if injected := await self.__signal.get_injected_context():
                await self.__telemetry.info("Context injected by user", context=injected)
                return injected

        elif signal == SignalType.INJECT.value:
            await self.__telemetry.info("Injection signal received")

            if injected := await self.__signal.get_injected_context():
                await self.__telemetry.info("Context injected", context=injected)
                return injected

        return None

    async def __learn(self, visual_hash: str, action: Action, success: bool) -> None:
        """
        Phase 5: Store experience in memory.
        """

        try:
            await self.__memory.store_experience(
                action=action,
                success=success,
                visual_hash=visual_hash,
            )
        except PortError as exception:
            await self.__telemetry.warning(
                "Failed to store experience",
                error=str(exception),
            )

    async def __checkpoint(self, step_result: StepResult) -> None:
        """
        Phase 6: Log execution state.
        """

        await self.__telemetry.info(
            "Step completed",
            success=step_result.success,
            duration_ms=step_result.duration,
            step_number=step_result.step.step_number,
            action=step_result.step.action.action_type.value,
        )
