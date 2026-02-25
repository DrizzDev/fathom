from __future__ import annotations

from logging import getLogger
from typing import Optional

from fathom.constants.events import FathomEvent
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.telemetry import TelemetryPort

logger = getLogger(__name__)


class HITLService:
    """
    Application service for Human-In-The-Loop operations.

    Orchestrates signal operations with proper telemetry event emission.
    Belongs to Application layer - has access to both ports and context.
    """

    def __init__(
        self,
        signal: SignalPort,
        telemetry: TelemetryPort,
    ) -> None:
        """
        Initialize HITL service with required ports.
        """

        self.__signal = signal
        self.__telemetry = telemetry

    async def check_signal(self) -> Optional[str]:
        """
        Check for control signal.
        """

        return await self.__signal.check_signal()

    async def wait_for_pause(self) -> None:
        """
        Wait for pause signal.
        """

        await self.__signal.wait_for_pause()

    async def wait_for_resume(self) -> None:
        """
        Wait for resume signal.
        """

        await self.__signal.wait_for_resume()

    async def is_pause_requested(self) -> bool:
        """
        Check if pause is requested.
        """

        return await self.__signal.is_pause_requested()

    async def has_injected_context(self) -> bool:
        """
        Check if context is available.
        """

        return await self.__signal.has_injected_context()

    async def get_injected_context(self, *, step: int) -> Optional[str]:
        """
        Retrieve injected context and emit HITL_RECEIVED event.
        """

        context = await self.__signal.get_injected_context()
        logger.info(f"Injected context for step number: {step} is {context}")

        if context:
            await self.__telemetry.info(
                step=step,
                context=context,
                type=FathomEvent.HITL_RECEIVED,
                message=f"User injected context: {context}",
            )

        return context

    async def ask(self, *, prompt: str, step: int) -> str:
        """
        Request human input with proper event emission.

        Emits HITL_REQUESTED before asking, HITL_RECEIVED after response.
        """

        await self.__telemetry.info(
            step=step,
            original_action=prompt,
            type=FathomEvent.HITL_REQUESTED,
            message=f"Action Paused: {prompt}",
        )

        response = await self.__signal.ask(prompt=prompt)
        logger.info(f"Received HITL response: {response} from user")

        await self.__telemetry.info(
            step=step,
            context=response,
            type=FathomEvent.HITL_RECEIVED,
            message=f"User injected context: {response}",
        )

        return response
