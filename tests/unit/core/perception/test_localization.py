from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.constants.observation import KeyboardVisibility
from fathom.core.perception.localization import TargetLocalizationService
from fathom.schemas.actions import Action, Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.budgets import LocalizationBudget
from fathom.schemas.localization import LocalizationStatus
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    PerceivedElement,
    ScreenObservation,
)
from fathom.schemas.screens import ScreenHashBundle


class TargetLocalizationServiceTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers perception-backed target localization behavior.
    """

    @staticmethod
    def __budget() -> LocalizationBudget:
        """
        Build a local-only localization budget.
        """

        return LocalizationBudget(local=120, vision=False, attempts=0, threshold=0.72)

    @staticmethod
    def __element(*, identifier: str, bounds: Bounds, text: str | None = None) -> PerceivedElement:
        """
        Build one tappable perceived element.
        """

        return PerceivedElement(
            identifier=identifier,
            bounds=bounds,
            source=ElementSource.VISION,
            role=ElementRole.BUTTON,
            confidence=0.9,
            text=text,
            tappable=True,
            parent=None,
        )

    @classmethod
    def __observation(cls, *, elements: tuple[PerceivedElement, ...]) -> ScreenObservation:
        """
        Build a minimal screen observation.
        """

        return ScreenObservation(
            activity="app",
            hashes=ScreenHashBundle(
                visual_hash="0" * 16,
                xml_hash="0" * 16,
                interaction_hash="0" * 16,
            ),
            elements=elements,
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
        )

    async def test_runtime_identifier_resolves(self) -> None:
        """
        The model may name a runtime observation identifier such as cv_1.
        """

        service = TargetLocalizationService()
        element = self.__element(
            identifier="cv_1",
            bounds=Bounds(
                x=100,
                y=200,
                width=120,
                height=60,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )
        action = Action(
            action_type=ActionType.TAP,
            target="cv_1",
            rationale="tap observed button",
            confidence=1.0,
        )

        result = await service.localize(
            action=action,
            observation=self.__observation(elements=(element,)),
            image=b"",
            budget=self.__budget(),
        )

        self.assertEqual(result.status, LocalizationStatus.RESOLVED)
        self.assertIsNotNone(result.bounds)
        self.assertEqual(result.bounds, element.bounds)

    async def test_resolved_method_logs_selected_element_for_rca(self) -> None:
        """
        A resolved localization must log the method and selected element so
        wrong-label allegations can be answered from logs alone.
        """

        service = TargetLocalizationService(workflow_id="wf-localization")
        element = self.__element(
            identifier="body_text",
            text="You can now find all your categories on top of the page",
            bounds=Bounds(
                x=10,
                y=20,
                width=300,
                height=80,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                source=CoordinateSource.XML,
            ),
        )
        action = Action(
            action_type=ActionType.TAP,
            target="Alright, got it button",
            natural_language_target="Alright, got it button",
            rationale="test",
            confidence=1.0,
            bounds=Bounds(
                x=10,
                y=20,
                width=300,
                height=80,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                source=CoordinateSource.MODEL,
            ),
        )

        with self.assertLogs("fathom.core.perception.localization", level="INFO") as captured:
            result = await service.localize(
                action=action,
                observation=self.__observation(elements=(element,)),
                image=b"",
                budget=self.__budget(),
            )

        self.assertEqual(result.status, LocalizationStatus.RESOLVED)
        records = [
            record
            for record in captured.records
            if record.__dict__.get("event") == "localization.method.evaluated"
            and record.__dict__.get("localization.status") == LocalizationStatus.RESOLVED.value
        ]
        self.assertTrue(records)
        record = records[-1]
        self.assertEqual(record.__dict__["localization.method"], "model_bounds")
        self.assertEqual(record.__dict__["action.target"], action.target)
        self.assertEqual(
            record.__dict__["action.natural_language_target"],
            action.natural_language_target,
        )
        self.assertEqual(
            record.__dict__["selected.element"]["text"],
            "You can now find all your categories on top of the page",
        )
        self.assertEqual(record.__dict__["selected.element"]["role"], ElementRole.BUTTON.value)
        self.assertEqual(record.__dict__["selected.element"]["source"], ElementSource.VISION.value)
        self.assertEqual(record.__dict__["candidate.count"], 1)

    async def test_model_bounds_without_perceived_overlap_are_unresolved(self) -> None:
        """
        Raw model bounds are not executable unless corroborated by perception.
        """

        service = TargetLocalizationService()
        element = self.__element(
            identifier="cv_1",
            bounds=Bounds(
                x=300,
                y=300,
                width=100,
                height=50,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )
        action = Action(
            action_type=ActionType.TAP,
            target="visible button",
            rationale="tap visible button",
            confidence=1.0,
            bounds=Bounds(
                x=10,
                y=10,
                width=100,
                height=50,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                source=CoordinateSource.MODEL,
            ),
        )

        result = await service.localize(
            action=action,
            observation=self.__observation(elements=(element,)),
            image=b"",
            budget=self.__budget(),
        )

        self.assertEqual(result.status, LocalizationStatus.UNRESOLVED)

    async def test_swipe_without_label_resolves_as_gesture(self) -> None:
        """
        Free viewport gestures do not require element localization.
        """

        service = TargetLocalizationService()
        action = Action(
            action_type=ActionType.SWIPE_UP,
            target="page",
            rationale="scroll",
            confidence=1.0,
        )

        result = await service.localize(
            action=action,
            observation=self.__observation(elements=()),
            image=b"",
            budget=self.__budget(),
        )

        self.assertEqual(result.status, LocalizationStatus.RESOLVED)
