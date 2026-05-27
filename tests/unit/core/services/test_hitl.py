from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

import pytest

from fathom.core.exceptions import HITLNotAvailableError
from fathom.core.services.hitl import HITLService
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities


class HITLServiceCapabilityBoundaryTest(unittest.IsolatedAsyncioTestCase):
    """Pins the HITLService precondition that no fabricated answer is ever returned."""

    @staticmethod
    def __signal() -> AsyncMock:
        """Build an AsyncMock signal port double."""

        signal = AsyncMock()
        signal.ask = AsyncMock(return_value="user reply")
        return signal

    @staticmethod
    def __telemetry() -> AsyncMock:
        """Build an AsyncMock telemetry port double."""

        telemetry = AsyncMock()
        telemetry.info = AsyncMock()
        return telemetry

    async def test_ask_raises_when_capability_disabled(self) -> None:
        """ask() raises HITLNotAvailableError when no human is available."""

        service = HITLService(
            signal=self.__signal(),
            telemetry=self.__telemetry(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        with pytest.raises(HITLNotAvailableError):
            await service.ask(prompt="any", step=1)

    async def test_ask_does_not_touch_signal_when_capability_disabled(self) -> None:
        """No signal.ask() call when capability is off — no phantom response."""

        signal = self.__signal()
        service = HITLService(
            signal=signal,
            telemetry=self.__telemetry(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        with pytest.raises(HITLNotAvailableError):
            await service.ask(prompt="any", step=1)

        signal.ask.assert_not_awaited()

    async def test_ask_does_not_emit_telemetry_when_capability_disabled(self) -> None:
        """No HITL_REQUESTED/HITL_RECEIVED telemetry on refusal."""

        telemetry = self.__telemetry()
        service = HITLService(
            signal=self.__signal(),
            telemetry=telemetry,
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        with pytest.raises(HITLNotAvailableError):
            await service.ask(prompt="any", step=1)

        telemetry.info.assert_not_awaited()

    async def test_ask_delegates_to_signal_when_capability_enabled(self) -> None:
        """When HITL is enabled, ask() forwards to the signal port and returns its reply."""

        signal = self.__signal()
        service = HITLService(
            signal=signal,
            telemetry=self.__telemetry(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )

        response = await service.ask(prompt="what's the OTP?", step=3)

        self.assertEqual(response, "user reply")
        signal.ask.assert_awaited_once_with(prompt="what's the OTP?")
