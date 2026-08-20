from __future__ import annotations

import unittest
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, Mock

from fathom.constants import ActionType
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.services.resolution import ReferenceResolutionService
from fathom.schemas.actions import Action, Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.resolution import ResolveStatus, UnresolvedKind


class ResolutionUnresolvedKindTest(unittest.IsolatedAsyncioTestCase):
    """
    One test per rejection branch in
    :meth:`ReferenceResolutionService.resolve`; each pins the typed ``unresolved_kind`` it must stamp.
    """

    @staticmethod
    def __build_service() -> ReferenceResolutionService:
        """
        Build a resolution service with a no-op memory ledger.
        """

        ledger = Mock()
        ledger.get = AsyncMock(return_value=None)
        return ReferenceResolutionService(ledger=ledger, catalog=CommandCatalogProvider().build())

    @staticmethod
    def __tap_action(
        *,
        label_id: Optional[str] = None,
        bounds: Optional[Bounds] = None,
    ) -> Action:
        """
        Build a TAP action used by the rejection-path fixtures.
        """

        if bounds is None:
            bounds = Bounds(
                x=10,
                y=20,
                width=100,
                height=40,
                source=CoordinateSource.MODEL,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            )

        return Action(
            target="X",
            bounds=bounds,
            confidence=0.95,
            label_id=label_id,
            rationale="tap visible X",
            action_type=ActionType.TAP,
        )

    async def test_spatial_action_missing_label_id_stamps_label_not_found(self) -> None:
        """
        Spatial action emitted without a label_id stamps LABEL_NOT_FOUND.
        """

        service = self.__build_service()
        resolved = await service.resolve(action=self.__tap_action(label_id=None), elements={})

        self.assertEqual(resolved.status, ResolveStatus.UNRESOLVED)
        self.assertEqual(resolved.unresolved_kind, UnresolvedKind.LABEL_NOT_FOUND)

    async def test_empty_manifest_stamps_empty_manifest(self) -> None:
        """
        A spatial tap with an empty manifest stamps EMPTY_MANIFEST.
        """

        service = self.__build_service()
        resolved = await service.resolve(
            elements=None,
            action=self.__tap_action(label_id="11"),
        )

        self.assertEqual(resolved.status, ResolveStatus.UNRESOLVED)
        self.assertEqual(resolved.unresolved_kind, UnresolvedKind.EMPTY_MANIFEST)

    async def test_label_not_in_manifest_stamps_label_not_found(self) -> None:
        """
        label_id absent from the manifest stamps LABEL_NOT_FOUND.
        """

        service = self.__build_service()
        elements: Dict[str, Any] = {"7": {"bounds": "[0,0][50,50]", "type": "View"}}

        resolved = await service.resolve(action=self.__tap_action(label_id="99"), elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.UNRESOLVED)
        self.assertEqual(resolved.unresolved_kind, UnresolvedKind.LABEL_NOT_FOUND)

    async def test_generic_visual_container_stamps_generic_container(self) -> None:
        """
        Text less View label with model bbox stamps GENERIC_CONTAINER — the signal supervise reads to set ``skip_label_id``.
        """

        service = self.__build_service()
        elements: Dict[str, Any] = {
            "13": {"bounds": "[136,0][290,110]", "type": "View"},
        }
        resolved = await service.resolve(action=self.__tap_action(label_id="13"), elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.UNRESOLVED)
        self.assertEqual(resolved.unresolved_kind, UnresolvedKind.GENERIC_CONTAINER)

    async def test_missing_bounds_stamps_missing_bounds(self) -> None:
        """
        Element with no bounds string stamps MISSING_BOUNDS.
        """

        service = self.__build_service()
        elements: Dict[str, Any] = {
            "5": {"text": "Submit", "type": "Button"},
        }
        action = Action(
            label_id="5",
            target="Submit",
            confidence=0.9,
            rationale="tap submit",
            action_type=ActionType.TAP,
        )

        resolved = await service.resolve(action=action, elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.UNRESOLVED)
        self.assertEqual(resolved.unresolved_kind, UnresolvedKind.MISSING_BOUNDS)

    async def test_invalid_bounds_format_stamps_invalid_bounds(self) -> None:
        """
        Bounds string that fails to match the expected pattern stamps INVALID_BOUNDS.
        """

        service = self.__build_service()
        elements: Dict[str, Any] = {
            "5": {"text": "Submit", "type": "Button", "bounds": "garbage"},
        }
        action = Action(
            label_id="5",
            target="Submit",
            confidence=0.9,
            rationale="tap submit",
            action_type=ActionType.TAP,
        )

        resolved = await service.resolve(action=action, elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.UNRESOLVED)
        self.assertEqual(resolved.unresolved_kind, UnresolvedKind.INVALID_BOUNDS)

    async def test_zero_area_bounds_stamp_invalid_bounds(self) -> None:
        """
        Bounds that clamp to zero area stamp INVALID_BOUNDS.
        """

        service = self.__build_service()
        elements: Dict[str, Any] = {
            "5": {"text": "Submit", "type": "Button", "bounds": "[100,100][100,100]"},
        }
        action = Action(
            label_id="5",
            target="Submit",
            confidence=0.9,
            rationale="tap submit",
            action_type=ActionType.TAP,
        )

        resolved = await service.resolve(action=action, elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.UNRESOLVED)
        self.assertEqual(resolved.unresolved_kind, UnresolvedKind.INVALID_BOUNDS)

    async def test_low_iou_text_less_view_stamps_generic_container(self) -> None:
        """
        Step-12 trap: LLM bbox far from label bounds (IoU=0) on a text less
        View must still escalate to perception via GENERIC_CONTAINER.
        """

        service = self.__build_service()
        elements: Dict[str, Any] = {
            "13": {"bounds": "[136,0][290,110]", "type": "View"},
        }
        action = Action(
            target="X",
            label_id="13",
            confidence=0.95,
            action_type=ActionType.TAP,
            rationale="tap close X in top right corner",
            bounds=Bounds(
                x=1936,
                y=15,
                width=98,
                height=95,
                source=CoordinateSource.MODEL,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )

        resolved = await service.resolve(action=action, elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.UNRESOLVED)
        self.assertEqual(resolved.unresolved_kind, UnresolvedKind.GENERIC_CONTAINER)

    async def test_high_iou_textless_view_snaps_label(self) -> None:
        """
        Text less View whose bounds the LLM bbox overlaps (IoU above the agreement floor) must trust the label_id and resolve via snap — avoiding an unnecessary perception round-trip when the LLM and manifest already agree on the region.
        """

        service = self.__build_service()
        elements: Dict[str, Any] = {
            "7": {"bounds": "[100,200][500,400]", "type": "View"},
        }
        action = Action(
            target="banner",
            label_id="7",
            confidence=0.9,
            action_type=ActionType.TAP,
            rationale="tap banner",
            bounds=Bounds(
                x=110,
                y=210,
                width=380,
                height=180,
                source=CoordinateSource.MODEL,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )

        resolved = await service.resolve(action=action, elements=elements)

        self.assertIsNone(resolved.unresolved_kind)
        self.assertEqual(resolved.status, ResolveStatus.RESOLVED)

    async def test_semantic_label_bypasses_iou_branch(self) -> None:
        """
        Element with semantic descriptor (text="Submit") never falls into
        the generic-container path; IoU is not consulted at all.
        """

        service = self.__build_service()
        elements: Dict[str, Any] = {
            "5": {
                "type": "Button",
                "text": "Submit",
                "bounds": "[100,100][300,200]",
            },
        }
        action = Action(
            target="Submit",
            label_id="5",
            confidence=0.9,
            action_type=ActionType.TAP,
            rationale="tap submit",
            bounds=Bounds(
                x=2000,
                y=2000,
                width=10,
                height=10,
                source=CoordinateSource.MODEL,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )

        resolved = await service.resolve(action=action, elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.RESOLVED)
        self.assertIsNone(resolved.unresolved_kind)

    async def test_textless_view_without_llm_bbox_resolves_via_snap(self) -> None:
        """
        Without an LLM bbox the generic-container branch is not entered
        regardless of the element's semantic emptiness.
        """

        service = self.__build_service()
        elements: Dict[str, Any] = {
            "5": {"bounds": "[100,200][500,400]", "type": "View"},
        }
        action = Action(
            target="banner",
            label_id="5",
            confidence=0.9,
            action_type=ActionType.TAP,
            rationale="tap banner",
        )

        resolved = await service.resolve(action=action, elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.RESOLVED)
        self.assertIsNone(resolved.unresolved_kind)

    async def test_resolved_action_carries_no_unresolved_kind(self) -> None:
        """
        Successfully resolved actions must not stamp ``unresolved_kind`` so
        downstream consumers can treat its presence as a strict signal.
        """

        service = self.__build_service()
        elements: Dict[str, Any] = {
            "5": {"text": "Submit", "type": "Button", "bounds": "[100,100][300,200]"},
        }
        action = Action(
            label_id="5",
            target="Submit",
            confidence=0.9,
            rationale="tap submit",
            action_type=ActionType.TAP,
        )

        resolved = await service.resolve(action=action, elements=elements)

        self.assertEqual(resolved.status, ResolveStatus.RESOLVED)
        self.assertIsNone(resolved.unresolved_kind)
