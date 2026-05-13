from __future__ import annotations

import unittest
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, Mock

from fathom.constants import ActionType
from fathom.core.services.resolution import ReferenceResolutionService
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.resolution import ResolveStatus


class ReferenceResolutionInputContextTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover input_context enrichment during label snapping, including
    placeholder detection and TYPE-only gating.
    """

    @staticmethod
    def __build_service() -> ReferenceResolutionService:
        """
        Build a resolution service with a no-op memory ledger.
        """

        ledger = Mock()
        ledger.get = AsyncMock(return_value=None)
        return ReferenceResolutionService(ledger=ledger)

    @staticmethod
    def __build_action(
        *,
        text: Optional[str] = None,
        label_id: Optional[str] = None,
        action_type: ActionType = ActionType.TYPE,
    ) -> Action:
        """
        Build a minimal action for resolution testing.
        """

        return Action(
            text=text,
            rationale="test",
            label_id=label_id,
            action_type=action_type,
            bounds=Bounds(x=0, y=0, width=100, height=100, coord_system="normalized"),
        )

    @staticmethod
    def __build_elements(
        label_id: str,
        *,
        text: str = "",
        hint: str = "",
        resource_id: str = "",
        bounds: str = "[100,200][500,300]",
    ) -> Dict[str, Any]:
        """
        Build an elements dict for a single label.
        """

        element: Dict[str, Any] = {"bounds": bounds, "text": text}

        if hint:
            element["hint"] = hint

        if resource_id:
            element["resource-id"] = resource_id

        return {label_id: element}

    async def test_text_differs_from_hint_sets_prefilled(self) -> None:
        """
        When element text differs from hint, prefilled is set to the text value.
        """

        service = self.__build_service()
        action = self.__build_action(label_id="5", text="search")
        elements = self.__build_elements(
            "5", text="chennai adyar", hint="Search an area", resource_id="com.app:id/search"
        )

        resolved = await service.resolve(action=action, elements=elements)

        self.assertIsNotNone(resolved.action.input_context)
        self.assertEqual(resolved.action.input_context.source, "xml")
        self.assertEqual(resolved.action.input_context.prefilled, "chennai adyar")
        self.assertEqual(resolved.action.input_context.locator, "com.app:id/search")

    async def test_text_equals_hint_skips_prefilled(self) -> None:
        """
        When element text equals the hint/placeholder, prefilled is empty.
        """

        service = self.__build_service()
        action = self.__build_action(label_id="3", text="search")
        elements = self.__build_elements(
            "3", text="Search an area", hint="Search an area", resource_id="com.app:id/input"
        )

        resolved = await service.resolve(action=action, elements=elements)

        self.assertIsNotNone(resolved.action.input_context)
        self.assertEqual(resolved.action.input_context.prefilled, "")
        self.assertEqual(resolved.action.input_context.locator, "com.app:id/input")

    async def test_text_present_hint_missing_sets_prefilled(self) -> None:
        """
        When element has text but no hint field, treat text as prefilled.
        """

        service = self.__build_service()
        action = self.__build_action(label_id="2", text="type")
        elements = self.__build_elements("2", text="existing value", resource_id="com.app:id/field")

        resolved = await service.resolve(action=action, elements=elements)

        self.assertIsNotNone(resolved.action.input_context)
        self.assertEqual(resolved.action.input_context.prefilled, "existing value")

    async def test_empty_text_empty_hint_no_locator_skips_input_context(self) -> None:
        """
        When element has no text, no hint, and no resource-id, input_context stays None.
        """

        service = self.__build_service()
        action = self.__build_action(label_id="1", text="type")
        elements = self.__build_elements("1", text="", hint="")

        resolved = await service.resolve(action=action, elements=elements)

        self.assertIsNone(resolved.action.input_context)

    async def test_no_elements_skips_input_context(self) -> None:
        """
        When no elements dict is provided, the spatial action cannot be
        snapped to a manifest element — :class:`ResolveResult` reports
        UNRESOLVED and ``input_context`` stays ``None``.
        """

        service = self.__build_service()
        action = self.__build_action(label_id="5", text="type")

        resolved = await service.resolve(action=action, elements=None)

        self.assertEqual(resolved.status, ResolveStatus.UNRESOLVED)
        self.assertIsNone(resolved.action.input_context)

    async def test_text_equals_placeholder_skips_prefilled(self) -> None:
        """
        When element text equals a placeholder field (not hint), prefilled is empty.
        """

        service = self.__build_service()
        action = self.__build_action(label_id="6", text="search")
        elements = {
            "6": {
                "text": "Search here",
                "bounds": "[0,0][500,100]",
                "placeholder": "Search here",
                "resource-id": "com.app:id/search",
            }
        }

        resolved = await service.resolve(action=action, elements=elements)

        self.assertIsNotNone(resolved.action.input_context)
        self.assertEqual(resolved.action.input_context.prefilled, "")
        self.assertEqual(resolved.action.input_context.locator, "com.app:id/search")

    async def test_tap_action_does_not_get_input_context(self) -> None:
        """
        Input context is only attached for TYPE actions, not TAP.
        """

        service = self.__build_service()
        action = self.__build_action(action_type=ActionType.TAP, label_id="5")
        elements = self.__build_elements("5", text="some text", resource_id="com.app:id/btn")

        resolved = await service.resolve(action=action, elements=elements)

        self.assertIsNone(resolved.action.input_context)

    async def test_bounds_still_snapped_without_input_context(self) -> None:
        """
        Label snapping sets pixel bounds regardless of whether
        ``input_context`` is populated. The result status must report
        RESOLVED so EXECUTE consumes the snapped bounds.
        """

        service = self.__build_service()
        action = self.__build_action(action_type=ActionType.TAP, label_id="7")
        elements = self.__build_elements("7", bounds="[10,20][200,50]")

        resolved = await service.resolve(action=action, elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.RESOLVED)
        self.assertIsNotNone(resolved.action.bounds)
        self.assertIsNone(resolved.action.input_context)

        self.assertEqual(resolved.action.bounds.x, 10)
        self.assertEqual(resolved.action.bounds.y, 20)
        self.assertEqual(resolved.action.bounds.width, 190)
        self.assertEqual(resolved.action.bounds.height, 30)
