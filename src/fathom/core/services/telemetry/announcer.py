from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fathom.constants.events import FathomEvent
from fathom.constants.phase import PhaseKind
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.telemetry import PhaseMessage


class PhaseAnnouncer:
    """
    Emits client-facing phase events through the telemetry port and keeps a
    single background heartbeat alive while the most recently announced phase is still in flight.
    """

    def __init__(self, *, message: PhaseMessage, telemetry: TelemetryPort) -> None:
        """
        Bind the announcer to a telemetry port and the deployment's phase message configuration.
        """

        self.__message = message
        self.__telemetry = telemetry

        self.__swap_lock = asyncio.Lock()
        self.__pulse: Optional[asyncio.Task[None]] = None

    async def intent_qualifying(self, *, intent: str) -> None:
        """
        Announce qualifier phase entry and start a heartbeat pulse for it.
        """

        await self.__open(
            intent=intent,
            kind=PhaseKind.QUALIFYING,
            event=FathomEvent.INTENT_QUALIFYING,
            message=self.__message.intent.qualifying,
        )

    async def intent_decomposing(self, *, intent: str) -> None:
        """
        Announce decomposer phase entry and start a heartbeat pulse for it.
        """

        await self.__open(
            intent=intent,
            kind=PhaseKind.DECOMPOSING,
            event=FathomEvent.INTENT_DECOMPOSING,
            message=self.__message.intent.decomposing,
        )

    async def grounding(self, *, intent: str) -> None:
        """
        Announce GROUND-node entry for the next step and start a heartbeat pulse for it.
        """

        await self.__open(
            intent=intent,
            kind=PhaseKind.GROUNDING,
            event=FathomEvent.GROUNDING,
            message=self.__message.step.grounding,
        )

    async def plan_synthesized(self, *, intent: str, sub_goals: List[Dict[str, Any]]) -> None:
        """
        Stop the pulse and emit the plan-synthesized event carrying the sub-goal breakdown.
        """

        await self.__close()

        await self.__telemetry.info(
            intent=intent,
            sub_goals=sub_goals,
            type=FathomEvent.PLAN_SYNTHESIZED,
            message=self.__message.intent.derived,
        )

    async def shutdown(self) -> None:
        """
        Cancel any in-flight pulse; called by the runtime at workflow termination.
        """

        await self.__close()

    async def __open(
        self,
        *,
        intent: str,
        message: str,
        kind: PhaseKind,
        event: FathomEvent,
    ) -> None:
        """
        Cancel any prior pulse, emit the start event, then schedule a fresh heartbeat loop.
        """

        async with self.__swap_lock:
            await self.__close_locked()
            await self.__telemetry.info(message, type=event, intent=intent)
            self.__pulse = asyncio.create_task(self.__heartbeat(kind=kind, intent=intent))

    async def __close(self) -> None:
        """
        Acquire the swap lock and cancel the active pulse if one is in flight.
        """

        async with self.__swap_lock:
            await self.__close_locked()

    async def __close_locked(self) -> None:
        """
        Cancel and await the active pulse under a caller-held swap lock.
        """

        task = self.__pulse
        self.__pulse = None

        if task is None or task.done():
            return

        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            return

    async def __heartbeat(self, *, kind: PhaseKind, intent: str) -> None:
        """
        Emit the heartbeat message at the configured cadence, bounded by the
        configured beat limit; cancellation stops the loop immediately.
        """

        budget = self.__message.heartbeat

        try:
            for _ in range(budget.limit):
                await asyncio.sleep(budget.threshold)
                await self.__telemetry.info(
                    intent=intent,
                    phase=kind.value,
                    message=budget.message,
                    type=FathomEvent.PHASE_HEARTBEAT,
                )
        except asyncio.CancelledError:
            return
