from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fathom.adapters.device.local.adb import ADBDevice
from fathom.constants.observation import KeyboardVisibility
from fathom.schemas.results import ActionResult


class ADBDeviceDetectKeyboardTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the dumpsys-based keyboard detection on Android.
    """

    def __device(self) -> ADBDevice:
        """
        Build a local ADB device for keyboard-detection tests.
        """

        return ADBDevice(serial="emulator-5554")

    async def test_visible_keyboard_returns_bounds(self) -> None:
        """
        ``mInputShown=true`` and a parseable SkRegion produce VISIBLE with device-pixel bounds.
        """

        device = self.__device()
        responses = [
            ActionResult(success=True, duration=1, output="mInputShown=true"),
            ActionResult(
                success=True,
                duration=1,
                output="touchable region=SkRegion((0,1507,1080,2208))",
            ),
        ]

        with patch.object(
            device,
            "_ADBDevice__shell",
            new_callable=AsyncMock,
            side_effect=responses,
        ):
            observation = await device.detect_keyboard()

        self.assertEqual(observation.visibility, KeyboardVisibility.VISIBLE)
        self.assertIsNotNone(observation.bounds)
        self.assertEqual(observation.bounds.x, 0)
        self.assertEqual(observation.bounds.y, 1507)
        self.assertEqual(observation.bounds.width, 1080)
        self.assertEqual(observation.bounds.height, 701)

    async def test_hidden_keyboard_short_circuits(self) -> None:
        """
        ``mInputShown=false`` returns HIDDEN without querying the touchable region.
        """

        device = self.__device()
        with patch.object(
            device,
            "_ADBDevice__shell",
            new_callable=AsyncMock,
            side_effect=[ActionResult(success=True, duration=1, output="mInputShown=false")],
        ) as mock_shell:
            observation = await device.detect_keyboard()

        self.assertEqual(observation.visibility, KeyboardVisibility.HIDDEN)
        self.assertIsNone(observation.bounds)
        self.assertEqual(mock_shell.await_count, 1)

    async def test_unknown_when_dumpsys_fails(self) -> None:
        """
        A failed ``dumpsys input_method`` collapses to UNKNOWN.
        """

        device = self.__device()
        with patch.object(
            device,
            "_ADBDevice__shell",
            new_callable=AsyncMock,
            side_effect=[ActionResult(success=False, duration=1, error="device offline")],
        ):
            observation = await device.detect_keyboard()

        self.assertEqual(observation.visibility, KeyboardVisibility.UNKNOWN)

    async def test_visible_without_bounds_when_region_unparseable(self) -> None:
        """
        Visible keyboard with unparseable touchable region returns VISIBLE with None bounds.
        """

        device = self.__device()
        responses = [
            ActionResult(success=True, duration=1, output="mInputShown=true"),
            ActionResult(success=True, duration=1, output="(no SkRegion here)"),
        ]
        with patch.object(
            device,
            "_ADBDevice__shell",
            new_callable=AsyncMock,
            side_effect=responses,
        ):
            observation = await device.detect_keyboard()

        self.assertEqual(observation.visibility, KeyboardVisibility.VISIBLE)
        self.assertIsNone(observation.bounds)
