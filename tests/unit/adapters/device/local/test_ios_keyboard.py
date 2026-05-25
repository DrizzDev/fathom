from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fathom.adapters.device.local.ios import IOSDevice
from fathom.constants.observation import KeyboardVisibility


_HIERARCHY_VISIBLE_KEYBOARD = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<XCUIElementTypeApplication name="App" x="0" y="0" width="430" height="932">'
    '  <XCUIElementTypeKeyboard visible="true" x="0" y="640" width="430" height="292"/>'
    "</XCUIElementTypeApplication>"
)

_HIERARCHY_NO_KEYBOARD = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<XCUIElementTypeApplication name="App" x="0" y="0" width="430" height="932"/>'
)


class IOSDeviceDetectKeyboardTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover XCUITest XML-walk keyboard detection on iOS with logical-to-pixel scaling.
    """

    def __device(self) -> IOSDevice:
        """
        Build a default iOS device for keyboard-detection tests.
        """

        return IOSDevice()

    async def test_visible_keyboard_scales_logical_bounds(self) -> None:
        """
        A visible keyboard element is scaled from logical points (430x932) to device pixels (1290x2796).
        """

        device = self.__device()
        with patch.object(
            device, "dump_hierarchy", new_callable=AsyncMock,
            return_value=_HIERARCHY_VISIBLE_KEYBOARD,
        ), patch.object(
            device, "get_dimensions", new_callable=AsyncMock, return_value=(1290, 2796),
        ):
            observation = await device.detect_keyboard()

        self.assertEqual(observation.visibility, KeyboardVisibility.VISIBLE)
        self.assertIsNotNone(observation.bounds)
        self.assertEqual(observation.bounds.x, 0)
        self.assertEqual(observation.bounds.y, 1920)
        self.assertEqual(observation.bounds.width, 1290)
        self.assertEqual(observation.bounds.height, 876)

    async def test_hidden_keyboard_when_element_absent(self) -> None:
        """
        Absent ``XCUIElementTypeKeyboard`` returns HIDDEN.
        """

        device = self.__device()
        with patch.object(
            device, "dump_hierarchy", new_callable=AsyncMock,
            return_value=_HIERARCHY_NO_KEYBOARD,
        ), patch.object(
            device, "get_dimensions", new_callable=AsyncMock, return_value=(1290, 2796),
        ):
            observation = await device.detect_keyboard()

        self.assertEqual(observation.visibility, KeyboardVisibility.HIDDEN)
        self.assertIsNone(observation.bounds)

    async def test_unknown_when_hierarchy_unavailable(self) -> None:
        """
        Missing hierarchy collapses to UNKNOWN.
        """

        device = self.__device()
        with patch.object(
            device, "dump_hierarchy", new_callable=AsyncMock, return_value=None,
        ):
            observation = await device.detect_keyboard()

        self.assertEqual(observation.visibility, KeyboardVisibility.UNKNOWN)

    async def test_unknown_when_hierarchy_unparseable(self) -> None:
        """
        Malformed XML collapses to UNKNOWN.
        """

        device = self.__device()
        with patch.object(
            device, "dump_hierarchy", new_callable=AsyncMock, return_value="<not xml",
        ), patch.object(
            device, "get_dimensions", new_callable=AsyncMock, return_value=(1290, 2796),
        ):
            observation = await device.detect_keyboard()

        self.assertEqual(observation.visibility, KeyboardVisibility.UNKNOWN)
