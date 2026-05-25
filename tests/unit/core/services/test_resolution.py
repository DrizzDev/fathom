from __future__ import annotations

import unittest
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, Mock

from fathom.constants import ActionType
from fathom.core.services.resolution import ReferenceResolutionService
from fathom.schemas.actions import Action, Bounds, CoordinateSource, CoordinateSystem
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
            bounds=Bounds(
                x=0,
                y=0,
                width=100,
                height=100,
                coordinate_system=CoordinateSystem.NORMALIZED,
            ),
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

    async def test_swipe_without_label_id_resolves_to_viewport_gesture(self) -> None:
        """
        Viewport gestures are spatial, but they are not element-targeted.
        They must remain executable without a manifest label.
        """

        service = self.__build_service()
        action = Action(
            rationale="scroll list",
            target="auto suggest page",
            action_type=ActionType.SWIPE_UP,
            confidence=1.0,
        )

        resolved = await service.resolve(action=action, elements=None)

        self.assertEqual(resolved.status, ResolveStatus.RESOLVED)
        self.assertIs(resolved.action.bounds, None)

    async def test_model_bbox_without_label_id_is_unresolved(self) -> None:
        """
        Model coordinates are not trusted by reference resolution. They
        require perception-backed localization before execution.
        """

        service = self.__build_service()
        bounds = Bounds(
            x=10,
            y=20,
            width=100,
            height=40,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            source=CoordinateSource.MODEL,
        )
        action = Action(
            bounds=bounds,
            rationale="tap visible button",
            target="visible button",
            action_type=ActionType.TAP,
            confidence=1.0,
        )

        resolved = await service.resolve(action=action, elements=None)

        self.assertEqual(resolved.status, ResolveStatus.UNRESOLVED)
        self.assertEqual(resolved.action.bounds, bounds)

    async def test_model_bbox_with_hallucinated_label_is_unresolved(self) -> None:
        """
        A hallucinated label_id cannot be rescued by model coordinates in
        reference resolution.
        """

        service = self.__build_service()
        bounds = Bounds(
            x=429,
            y=543,
            width=348,
            height=120,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            source=CoordinateSource.MODEL,
        )
        action = Action(
            bounds=bounds,
            label_id="110",
            rationale="tap visible Alright button",
            target="Alright, got it button",
            action_type=ActionType.TAP,
            confidence=1.0,
        )
        elements = {"11": {"bounds": "[387,1185][720,1608]", "type": "View"}}

        resolved = await service.resolve(action=action, elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.UNRESOLVED)
        self.assertEqual(resolved.action.bounds, bounds)

    async def test_model_bbox_does_not_beat_generic_textless_container_label(self) -> None:
        """
        A textless View label is often a layout container. Reference
        resolution must not silently choose the model bbox instead.
        """

        service = self.__build_service()
        bounds = Bounds(
            x=429,
            y=522,
            width=348,
            height=120,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            source=CoordinateSource.MODEL,
        )
        action = Action(
            bounds=bounds,
            label_id="11",
            rationale="tap visible Alright button",
            target="Alright, got it button",
            action_type=ActionType.TAP,
            confidence=1.0,
        )
        elements = {"11": {"bounds": "[387,1185][720,1608]", "type": "View"}}

        resolved = await service.resolve(action=action, elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.UNRESOLVED)
        self.assertEqual(resolved.action.bounds, bounds)

    async def test_ocr_manifest_label_snaps_with_ocr_coordinate_source(self) -> None:
        """
        OCR-backed manifest entries must resolve as trusted OCR coordinates.
        """

        service = self.__build_service()
        action = Action(
            label_id="1",
            rationale="open app",
            target="Swiggy app icon",
            action_type=ActionType.TAP,
            confidence=1.0,
        )
        elements = {
            "1": {
                "text": "Swiggy",
                "source": "ocr",
                "bounds": "[284,383][392,414]",
            },
        }

        resolved = await service.resolve(action=action, elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.RESOLVED)
        self.assertIsNotNone(resolved.action.bounds)
        self.assertEqual(resolved.action.bounds.source, CoordinateSource.OCR)
        self.assertEqual(resolved.action.bounds.x, 284)
        self.assertEqual(resolved.action.bounds.y, 383)

    async def test_semantic_label_still_beats_model_bbox(self) -> None:
        """
        Manifest snapping remains preferred when the chosen label has
        semantic metadata. The model bbox fallback is only for missing or
        generic manifest grounding.
        """

        service = self.__build_service()
        model_bounds = Bounds(
            x=429,
            y=522,
            width=348,
            height=120,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            source=CoordinateSource.MODEL,
        )
        action = Action(
            bounds=model_bounds,
            label_id="28",
            rationale="tap close button",
            target="close button",
            action_type=ActionType.TAP,
            confidence=1.0,
        )
        elements = {"28": {"bounds": "[1050,186][1158,294]", "label": "Close"}}

        resolved = await service.resolve(action=action, elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.RESOLVED)
        self.assertIsNotNone(resolved.action.bounds)
        self.assertEqual(resolved.action.bounds.x, 1050)
        self.assertEqual(resolved.action.bounds.y, 186)
        self.assertEqual(resolved.action.bounds.width, 108)
        self.assertEqual(resolved.action.bounds.height, 108)

    async def test_cv_label_still_beats_model_bbox(self) -> None:
        """
        CV labels are already visual grounding evidence. If the model
        chooses a CV label, execute the CV bounds rather than a possibly
        noisy bbox from the same turn.
        """

        service = self.__build_service()
        model_bounds = Bounds(
            x=429,
            y=543,
            width=348,
            height=120,
            coordinate_system=CoordinateSystem.NORMALIZED,
            source=CoordinateSource.MODEL,
        )
        action = Action(
            bounds=model_bounds,
            label_id="37",
            rationale="tap visible Alright button",
            target="Alright, got it button",
            action_type=ActionType.TAP,
            confidence=1.0,
        )
        elements = {
            "37": {
                "bounds": "[429,1148][806,1256]",
                "class": "VisualControl",
                "source": "cv",
            }
        }

        resolved = await service.resolve(action=action, elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.RESOLVED)
        self.assertIsNotNone(resolved.action.bounds)
        self.assertEqual(resolved.action.bounds.x, 429)
        self.assertEqual(resolved.action.bounds.y, 1148)
        self.assertEqual(resolved.action.bounds.width, 377)
        self.assertEqual(resolved.action.bounds.height, 108)

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

    async def test_negative_y_bounds_are_clamped_to_viewport(self) -> None:
        """
        Bounds with a negative origin (off-viewport scroll containers)
        must be clamped to zero so the element stays snappable; the
        post-clamp rect carries the visible portion only.
        """

        service = self.__build_service()
        action = self.__build_action(action_type=ActionType.TAP, label_id="9")
        elements = self.__build_elements("9", bounds="[0,-720][1206,1905]")

        resolved = await service.resolve(action=action, elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.RESOLVED)
        self.assertIsNotNone(resolved.action.bounds)
        self.assertEqual(resolved.action.bounds.x, 0)
        self.assertEqual(resolved.action.bounds.y, 0)
        self.assertEqual(resolved.action.bounds.width, 1206)
        self.assertEqual(resolved.action.bounds.height, 1905)

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
