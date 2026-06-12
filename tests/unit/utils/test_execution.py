"""
Unit tests for device-action execution guards.
"""

from __future__ import annotations

import pytest

from fathom.constants import ActionType
from fathom.schemas.actions import Action, Bounds
from fathom.tools.device.mock import MockDeviceTool
from fathom.utils.execution import execute_device_action


class TestExecuteDeviceAction:
    """
    The TYPE focus guard requires bounds before tapping and typing.
    """

    @pytest.mark.asyncio
    async def test_type_requires_bounds(self) -> None:
        device = MockDeviceTool()
        action = Action(
            action_type=ActionType.TYPE,
            target="search field",
            text="hello",
            rationale="Type query text",
        )

        result = await execute_device_action(device, action)

        assert not result.success
        assert result.error == "Type action requires bounds for focus tap guard"
        assert device.tap_calls == []
        assert device.type_calls == []

    @pytest.mark.asyncio
    async def test_type_taps_before_typing_when_bounds_present(self) -> None:
        device = MockDeviceTool(screen_size=(1000, 1000))
        action = Action(
            action_type=ActionType.TYPE,
            target="search field",
            text="hello",
            rationale="Type query text",
            bounds=Bounds(x=100, y=200, width=300, height=80),
        )

        result = await execute_device_action(device, action)

        assert result.success
        assert result.error is None
        # Legacy bbox: cx = 100 + 300/2 = 250, cy = 200 + 80/4 = 220 (w/4 fallback)
        assert device.tap_calls == [(250, 220)]
        assert device.type_calls == ["hello"]
        assert device.type_replace_calls == [True]
