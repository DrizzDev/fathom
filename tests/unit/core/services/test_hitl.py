from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

import pytest

from fathom.constants.events import FathomEvent
from fathom.core.exceptions import HITLNotAvailableError
from fathom.core.services.hitl import HITLService
from fathom.core.services.telemetry import PhaseAnnouncer
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities


class _Doubles:
    """
    Centralized AsyncMock factory so every test draws from the same shape and
    a forgotten signature update fails in one place, not scattered across tests.
    """

    @staticmethod
    def signal() -> AsyncMock:
        """
        Build a signal port double with a default 'user reply' answer.
        """

        signal = AsyncMock()
        signal.ask = AsyncMock(return_value="user reply")
        signal.get_injected_context = AsyncMock(return_value=None)

        return signal

    @staticmethod
    def telemetry() -> AsyncMock:
        """
        Build a telemetry port double that records every info emit.
        """

        telemetry = AsyncMock()
        telemetry.info = AsyncMock()

        return telemetry

    @staticmethod
    def phase() -> AsyncMock:
        """
        Build a phase announcer double exposing pause/resume/shutdown.
        """

        phase = AsyncMock(spec=PhaseAnnouncer)

        phase.pause = AsyncMock()
        phase.resume = AsyncMock()
        phase.shutdown = AsyncMock()

        return phase


class HITLServiceCapabilityBoundaryTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the HITLService precondition that no fabricated answer is ever returned when the capability is off.
    """

    async def test_ask_raises_when_capability_disabled(self) -> None:
        """
        ask() raises HITLNotAvailableError when no human is available.
        """

        service = HITLService(
            phase=_Doubles.phase(),
            signal=_Doubles.signal(),
            telemetry=_Doubles.telemetry(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        with pytest.raises(HITLNotAvailableError):
            await service.ask(prompt="any", step=1)

    async def test_ask_does_not_touch_signal_when_capability_disabled(self) -> None:
        """
        No signal.ask() call when capability is off; no phantom response can leak through.
        """

        signal = _Doubles.signal()

        service = HITLService(
            signal=signal,
            phase=_Doubles.phase(),
            telemetry=_Doubles.telemetry(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        with pytest.raises(HITLNotAvailableError):
            await service.ask(prompt="any", step=1)

        signal.ask.assert_not_awaited()

    async def test_ask_does_not_emit_telemetry_when_capability_disabled(self) -> None:
        """
        No HITL_REQUESTED / HITL_RECEIVED telemetry on refusal.
        """

        telemetry = _Doubles.telemetry()

        service = HITLService(
            telemetry=telemetry,
            phase=_Doubles.phase(),
            signal=_Doubles.signal(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        with pytest.raises(HITLNotAvailableError):
            await service.ask(prompt="any", step=1)

        telemetry.info.assert_not_awaited()

    async def test_ask_does_not_pause_phase_when_capability_disabled(self) -> None:
        """
        No phase.pause() on the refusal path; the announcer state stays clean.
        """

        phase = _Doubles.phase()

        service = HITLService(
            phase=phase,
            signal=_Doubles.signal(),
            telemetry=_Doubles.telemetry(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        with pytest.raises(HITLNotAvailableError):
            await service.ask(prompt="any", step=1)

        phase.pause.assert_not_awaited()
        phase.resume.assert_not_awaited()

    async def test_ask_delegates_to_signal_when_capability_enabled(self) -> None:
        """
        When HITL is enabled, ask() forwards to the signal port and returns its reply.
        """

        signal = _Doubles.signal()

        service = HITLService(
            signal=signal,
            phase=_Doubles.phase(),
            telemetry=_Doubles.telemetry(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )

        response = await service.ask(prompt="what's the OTP?", step=3)

        self.assertEqual(response, "user reply")
        signal.ask.assert_awaited_once_with(prompt="what's the OTP?")


class HITLPhaseAnnouncerWiringTest(unittest.IsolatedAsyncioTestCase):
    """
    Regression: HITL_REQUESTED must pause the phase pulse and HITL_RECEIVED must resume it.
    These were missing in prod, leading to 'Still working...' beats during paused HITL waits.
    """

    async def test_ask_pauses_phase_after_hitl_requested(self) -> None:
        """
        ask() must emit HITL_REQUESTED and then pause the announcer so beats stop while waiting.
        """

        phase = _Doubles.phase()
        signal = _Doubles.signal()
        telemetry = _Doubles.telemetry()

        service = HITLService(
            phase=phase,
            signal=signal,
            telemetry=telemetry,
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )

        await service.ask(prompt="OTP?", step=2)

        emitted_types = [call.kwargs.get("type") for call in telemetry.info.await_args_list]
        self.assertIn(FathomEvent.HITL_REQUESTED, emitted_types)

        phase.pause.assert_awaited()

    async def test_ask_resumes_phase_after_hitl_received(self) -> None:
        """
        ask() must emit HITL_RECEIVED after the user replies and resume the announcer.
        """

        phase = _Doubles.phase()
        telemetry = _Doubles.telemetry()

        service = HITLService(
            phase=phase,
            telemetry=telemetry,
            signal=_Doubles.signal(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )

        await service.ask(prompt="OTP?", step=2)

        emitted_types = [call.kwargs.get("type") for call in telemetry.info.await_args_list]
        self.assertIn(FathomEvent.HITL_RECEIVED, emitted_types)

        phase.resume.assert_awaited()

    async def test_ask_pause_failure_does_not_swallow_telemetry(self) -> None:
        """
        If phase.pause raises, the HITL_REQUESTED telemetry must already have been emitted
        and the signal.ask must still proceed; the user cannot go blind because of a pulse-management bug.
        """

        phase = _Doubles.phase()
        phase.pause = AsyncMock(side_effect=RuntimeError("pause kaboom"))

        signal = _Doubles.signal()
        telemetry = _Doubles.telemetry()

        service = HITLService(
            phase=phase,
            signal=signal,
            telemetry=telemetry,
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )

        response = await service.ask(prompt="hi", step=1)

        self.assertEqual(response, "user reply")
        signal.ask.assert_awaited_once()

        emitted_types = [call.kwargs.get("type") for call in telemetry.info.await_args_list]
        self.assertIn(FathomEvent.HITL_REQUESTED, emitted_types)

    async def test_get_injected_context_resumes_phase(self) -> None:
        """
        When a context arrives via signal injection (no ask flow), the announcer must resume.
        """

        phase = _Doubles.phase()
        signal = _Doubles.signal()
        signal.get_injected_context = AsyncMock(return_value="injected!")

        service = HITLService(
            phase=phase,
            signal=signal,
            telemetry=_Doubles.telemetry(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )

        context = await service.get_injected_context(step=5)

        self.assertEqual(context, "injected!")
        phase.resume.assert_awaited()

    async def test_get_injected_context_does_not_resume_when_no_context(self) -> None:
        """
        No context, no telemetry, no resume. The announcer stays in its previous state.
        """

        phase = _Doubles.phase()

        service = HITLService(
            phase=phase,
            signal=_Doubles.signal(),
            telemetry=_Doubles.telemetry(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )

        context = await service.get_injected_context(step=5)

        self.assertIsNone(context)
        phase.resume.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
