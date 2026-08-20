from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fathom.constants.events import FathomEvent
from fathom.constants.phase import PhaseKind
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.telemetry import PhaseMessage

logger = logging.getLogger(__name__)


class _ActivePhase:
    """
    Snapshot of the phase currently driving the keep-alive heartbeat.
    Kept across pause/resume so the same phase identity survives a quiet window.
    """

    def __init__(self, *, kind: PhaseKind, intent: str) -> None:
        """
        Bind the active-phase snapshot to a kind and intent payload.
        """

        self.kind = kind
        self.intent = intent


class PhaseAnnouncer:
    """
    Emits client-facing phase events through the telemetry port and keeps a single background
    heartbeat alive while the most recently announced phase is still in flight.
    """

    def __init__(self, *, message: PhaseMessage, telemetry: TelemetryPort) -> None:
        """
        Bind the announcer to a telemetry port and the deployment's phase message configuration.
        """

        self.__message = message
        self.__telemetry = telemetry

        self.__swap_lock = asyncio.Lock()
        self.__pulse: Optional[asyncio.Task[None]] = None

        self.__paused: bool = False
        self.__active: Optional[_ActivePhase] = None

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

    async def planning(self, *, intent: str) -> None:
        """
        Announce planner entry for the next step and start a heartbeat pulse across its internal retries.
        """

        await self.__open(
            intent=intent,
            kind=PhaseKind.PLANNING,
            event=FathomEvent.PLANNING,
            message=self.__message.step.planning,
        )

    async def verifying(self, *, intent: str) -> None:
        """
        Announce the post-action completion check and start a heartbeat pulse for it.
        """

        await self.__open(
            intent=intent,
            kind=PhaseKind.VERIFYING,
            event=FathomEvent.VERIFYING,
            message=self.__message.step.verifying,
        )

    async def authoring(self, *, intent: str) -> None:
        """
        Announce post-execution script authoring and start a heartbeat pulse for it.
        """

        await self.__open(
            intent=intent,
            kind=PhaseKind.AUTHORING,
            event=FathomEvent.AUTHORING,
            message=self.__message.intent.authoring,
        )

    async def plan_synthesized(self, *, intent: str, sub_goals: List[Dict[str, Any]]) -> None:
        """
        Stop the pulse and emit the plan-synthesized event carrying the sub-goal breakdown.
        """

        await self.__close()

        await self.__telemetry.info(
            self.__message.intent.derived,
            intent=intent,
            sub_goals=sub_goals,
            type=FathomEvent.PLAN_SYNTHESIZED,
        )

    async def pause(self) -> None:
        """
        Cancel the pending heartbeat but keep the active phase identity so resume can restart it.
        Safe to call when no pulse is active or when already paused.
        """

        async with self.__swap_lock:
            await self.__close_locked()
            self.__paused = True

    async def resume(self) -> None:
        """
        Reschedule a heartbeat for the previously active phase if one was paused.
        Safe to call when not currently paused or when no phase is active.
        """

        async with self.__swap_lock:
            if not self.__paused:
                return

            self.__paused = False

            if self.__active is None:
                return

            self.__pulse = asyncio.create_task(
                self.__heartbeat(kind=self.__active.kind, intent=self.__active.intent),
            )

    async def shutdown(self) -> None:
        """
        Cancel any in-flight pulse and clear all phase state; called at workflow termination.
        """

        async with self.__swap_lock:
            await self.__close_locked()
            self.__active = None
            self.__paused = False

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

            self.__paused = False
            self.__active = _ActivePhase(kind=kind, intent=intent)

            await self.__telemetry.info(message, type=event, intent=intent)
            self.__pulse = asyncio.create_task(self.__heartbeat(kind=kind, intent=intent))

    async def __close(self) -> None:
        """
        Acquire the swap lock and cancel the active pulse if one is in flight.
        """

        async with self.__swap_lock:
            await self.__close_locked()

            self.__active = None
            self.__paused = False

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
        except Exception as exception:
            logger.warning(
                "phase pulse raised during cancellation; suppressed to avoid blinding telemetry: %s",
                exception,
            )

    async def __heartbeat(self, *, kind: PhaseKind, intent: str) -> None:
        """
        Emit the heartbeat message at the configured cadence, bounded by the configured beat limit;
        cancellation stops the loop immediately.
        """

        budget = self.__message.heartbeat

        try:
            for _ in range(budget.limit):
                await asyncio.sleep(budget.threshold)
                await self.__telemetry.info(
                    budget.message,
                    intent=intent,
                    phase=kind.value,
                    type=FathomEvent.PHASE_HEARTBEAT,
                )
        except asyncio.CancelledError:
            return
