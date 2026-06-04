from __future__ import annotations

import asyncio
import unittest
from typing import Any, Dict, List, Optional, Tuple

from fathom.constants.events import FathomEvent
from fathom.constants.phase import PhaseKind
from fathom.core.services.telemetry.announcer import PhaseAnnouncer
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.telemetry import HeartbeatBudget, IntentMessage, PhaseMessage


class _RecordingTelemetry(TelemetryPort):
    """
    Test-only telemetry port that captures every emitted event.
    """

    def __init__(self) -> None:
        """
        Initialise the recorder with an empty event list.
        """

        self.events: List[Tuple[str, FathomEvent, Dict[str, Any]]] = []

    async def info(self, message: str, **context: Any) -> None:
        """
        Record the message, event type, and remaining context for later assertions.
        """

        event_type = context.pop("type")
        self.events.append((message, event_type, context))

    async def warning(self, message: str, **context: Any) -> None:
        """
        Capture warning-level events with the same shape as info.
        """

        await self.info(message, **context)

    async def error(self, message: str, **context: Any) -> None:
        """
        Capture error-level events with the same shape as info.
        """

        await self.info(message, **context)

    async def debug(self, message: str, **context: Any) -> None:
        """
        Capture debug-level events with the same shape as info.
        """

        await self.info(message, **context)

    async def exception(
        self,
        message: str,
        *,
        exception: Optional[BaseException] = None,
        **context: Any,
    ) -> None:
        """
        Capture exception-level events with the same shape as info.
        """

        await self.info(message, **context)


class PhaseAnnouncerStartEventTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the start events emitted by each announce method.
    """

    @staticmethod
    def __build(
        *,
        threshold: float = 60.0,
        limit: int = 1,
    ) -> Tuple[PhaseAnnouncer, _RecordingTelemetry]:
        """
        Build an announcer with a high threshold so the pulse will not fire during the test.
        """

        telemetry = _RecordingTelemetry()
        message = PhaseMessage(
            intent=IntentMessage(
                qualifying="Reading...",
                decomposing="Breaking down...",
                derived="Plan ready.",
            ),
            heartbeat=HeartbeatBudget.model_construct(
                threshold=threshold, limit=limit, message="Beat..."
            ),
        )
        return PhaseAnnouncer(telemetry=telemetry, message=message), telemetry

    async def test_intent_qualifying_emits_start_event(self) -> None:
        """
        Calling intent_qualifying emits one INTENT_QUALIFYING event with the configured message and intent.
        """

        announcer, telemetry = self.__build()

        await announcer.intent_qualifying(intent="open the app")
        await announcer.shutdown()

        starts = [event for event in telemetry.events if event[1] is FathomEvent.INTENT_QUALIFYING]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0][0], "Reading...")
        self.assertEqual(starts[0][2]["intent"], "open the app")

    async def test_intent_decomposing_emits_start_event(self) -> None:
        """
        Calling intent_decomposing emits one INTENT_DECOMPOSING event with the configured message and intent.
        """

        announcer, telemetry = self.__build()

        await announcer.intent_decomposing(intent="open the app")
        await announcer.shutdown()

        starts = [event for event in telemetry.events if event[1] is FathomEvent.INTENT_DECOMPOSING]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0][0], "Breaking down...")
        self.assertEqual(starts[0][2]["intent"], "open the app")

    async def test_grounding_emits_start_event(self) -> None:
        """
        Calling grounding emits one GROUNDING event with the configured step message and intent.
        """

        announcer, telemetry = self.__build()

        await announcer.grounding(intent="open the app")
        await announcer.shutdown()

        starts = [event for event in telemetry.events if event[1] is FathomEvent.GROUNDING]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0][2]["intent"], "open the app")

    async def test_plan_synthesized_emits_terminal_event(self) -> None:
        """
        plan_synthesized emits PLAN_SYNTHESIZED carrying the sub-goal list and intent.
        """

        announcer, telemetry = self.__build()
        sub_goals = [{"index": 0, "directive": "TAP", "description": "tap submit"}]

        await announcer.plan_synthesized(intent="x", sub_goals=sub_goals)
        await announcer.shutdown()

        terminals = [
            event for event in telemetry.events if event[1] is FathomEvent.PLAN_SYNTHESIZED
        ]
        self.assertEqual(len(terminals), 1)
        self.assertEqual(terminals[0][0], "Plan ready.")
        self.assertEqual(terminals[0][2]["intent"], "x")
        self.assertEqual(terminals[0][2]["sub_goals"], sub_goals)


class PhaseAnnouncerPulseTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the pulse lifecycle: fires at threshold, swap cancels prior pulse,
    plan_synthesized cancels pulse, shutdown cancels pulse, beat limit bounds the loop.
    """

    @staticmethod
    def __build(
        *,
        threshold: float,
        limit: int = 60,
    ) -> Tuple[PhaseAnnouncer, _RecordingTelemetry]:
        """
        Build an announcer with the supplied heartbeat threshold and limit.
        """

        telemetry = _RecordingTelemetry()
        message = PhaseMessage(
            intent=IntentMessage(),
            heartbeat=HeartbeatBudget.model_construct(
                threshold=threshold, limit=limit, message="Beat..."
            ),
        )
        return PhaseAnnouncer(telemetry=telemetry, message=message), telemetry

    async def test_pulse_fires_after_threshold(self) -> None:
        """
        After the configured threshold elapses, one PHASE_HEARTBEAT is emitted tagged with the active phase.
        """

        announcer, telemetry = self.__build(threshold=0.05, limit=1)

        await announcer.intent_qualifying(intent="long task")
        await asyncio.sleep(0.15)
        await announcer.shutdown()

        beats = [event for event in telemetry.events if event[1] is FathomEvent.PHASE_HEARTBEAT]
        self.assertGreaterEqual(len(beats), 1)
        self.assertEqual(beats[0][2]["phase"], PhaseKind.QUALIFYING.value)
        self.assertEqual(beats[0][2]["intent"], "long task")

    async def test_swap_cancels_prior_pulse(self) -> None:
        """
        Announcing a new phase cancels the previous pulse so no beats are emitted under the prior phase tag.
        """

        announcer, telemetry = self.__build(threshold=0.5, limit=1)

        await announcer.intent_qualifying(intent="x")
        await announcer.intent_decomposing(intent="x")
        await asyncio.sleep(0.05)
        await announcer.shutdown()

        beats_under_qualifying = [
            event
            for event in telemetry.events
            if event[1] is FathomEvent.PHASE_HEARTBEAT
            and event[2]["phase"] == PhaseKind.QUALIFYING.value
        ]
        self.assertEqual(beats_under_qualifying, [])

    async def test_plan_synthesized_cancels_pulse(self) -> None:
        """
        plan_synthesized cancels any active pulse before emitting its terminal event.
        """

        announcer, telemetry = self.__build(threshold=0.5, limit=1)

        await announcer.intent_decomposing(intent="x")
        await announcer.plan_synthesized(intent="x", sub_goals=[])
        await asyncio.sleep(0.6)
        await announcer.shutdown()

        beats = [event for event in telemetry.events if event[1] is FathomEvent.PHASE_HEARTBEAT]
        self.assertEqual(beats, [])

    async def test_shutdown_cancels_pulse(self) -> None:
        """
        shutdown cancels the active pulse so no beats fire after the workflow terminates.
        """

        announcer, telemetry = self.__build(threshold=0.5, limit=1)

        await announcer.intent_qualifying(intent="x")
        await announcer.shutdown()
        await asyncio.sleep(0.6)

        beats = [event for event in telemetry.events if event[1] is FathomEvent.PHASE_HEARTBEAT]
        self.assertEqual(beats, [])

    async def test_limit_bounds_the_loop(self) -> None:
        """
        After ``limit`` beats the pulse stops on its own, even without an external cancel.
        """

        announcer, telemetry = self.__build(threshold=0.02, limit=3)

        await announcer.intent_qualifying(intent="x")
        await asyncio.sleep(0.25)
        await announcer.shutdown()

        beats = [event for event in telemetry.events if event[1] is FathomEvent.PHASE_HEARTBEAT]
        self.assertLessEqual(len(beats), 3)

    async def test_shutdown_when_no_pulse_active_is_noop(self) -> None:
        """
        Calling shutdown without an in-flight phase does not raise and emits no events.
        """

        announcer, telemetry = self.__build(threshold=0.5, limit=1)

        await announcer.shutdown()

        self.assertEqual(telemetry.events, [])


if __name__ == "__main__":
    unittest.main()
