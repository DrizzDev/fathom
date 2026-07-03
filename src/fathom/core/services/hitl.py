from __future__ import annotations

from logging import getLogger
from typing import Optional

from fathom.constants.events import FathomEvent
from fathom.core.exceptions import HITLNotAvailableError
from fathom.core.services.telemetry import PhaseAnnouncer
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.capabilities import RuntimeCapabilities

logger = getLogger(__name__)


class HITLService:
    """
    Application-layer service that orchestrates signal and telemetry for HITL operations.
    """

    def __init__(
        self,
        *,
        signal: SignalPort,
        phase: PhaseAnnouncer,
        telemetry: TelemetryPort,
        capabilities: RuntimeCapabilities,
    ) -> None:
        """
        Initialize HITL service with ports and runtime capabilities.
        """

        self.__phase = phase
        self.__signal = signal
        self.__telemetry = telemetry
        self.__capabilities = capabilities

    async def check_signal(self) -> Optional[str]:
        """
        Return the current control signal, if any.
        """

        return await self.__signal.check_signal()

    async def wait_for_pause(self) -> None:
        """
        Block until a pause signal arrives.
        """

        await self.__signal.wait_for_pause()

    async def wait_for_resume(self) -> None:
        """
        Block until a resume signal arrives.
        """

        await self.__signal.wait_for_resume()

    async def is_pause_requested(self) -> bool:
        """
        Return whether a pause is currently requested.
        """

        return await self.__signal.is_pause_requested()

    async def has_injected_context(self) -> bool:
        """
        Return whether injected context is available.
        """

        return await self.__signal.has_injected_context()

    async def peek_next_context(self) -> Optional[str]:
        """
        Return the next injected context without consuming it.
        """

        return await self.__signal.peek_next_context()

    async def consume_context(self) -> None:
        """
        Consume the next injected context.
        """

        await self.__signal.consume_context()

    async def get_injected_context(self, *, step: int) -> Optional[str]:
        """
        Retrieve injected context and emit the HITL_RECEIVED event.
        """

        context = await self.__signal.get_injected_context()
        logger.info(f"Injected context for step number: {step} is {context}")

        if context:
            await self.__telemetry.info(
                message="Got your message — picking up from here.",
                step=step,
                context=context,
                type=FathomEvent.HITL_RECEIVED,
            )

            try:
                await self.__phase.resume()
            except Exception as exception:
                logger.warning(
                    "phase resume after HITL_RECEIVED failed (telemetry already emitted): %s",
                    exception,
                )

        return context

    async def ask(self, *, prompt: str, step: int) -> str:
        """
        Request human input; raise HITLNotAvailableError when the runtime cannot service HITL.
        """

        if not self.__capabilities.hitl.enabled:
            raise HITLNotAvailableError()

        await self.__telemetry.info(
            message=f"Paused — need your input: {prompt}",
            step=step,
            original_action=prompt,
            type=FathomEvent.HITL_REQUESTED,
        )

        try:
            await self.__phase.pause()
        except Exception as exception:
            logger.warning(
                "phase pause after HITL_REQUESTED failed (telemetry already emitted): %s", exception
            )

        response = await self.__signal.ask(prompt=prompt)
        logger.info(f"Received HITL response: {response} from user")

        await self.__telemetry.info(
            message="Got it — continuing.",
            step=step,
            context=response,
            type=FathomEvent.HITL_RECEIVED,
        )

        try:
            await self.__phase.resume()
        except Exception as exception:
            logger.warning(
                "phase resume after HITL_RECEIVED failed (telemetry already emitted): %s", exception
            )

        return response
