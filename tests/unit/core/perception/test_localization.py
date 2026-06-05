from __future__ import annotations

import unittest
from typing import Tuple

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
            text=text,
            parent=None,
            bounds=bounds,
            tappable=True,
            confidence=0.9,
            identifier=identifier,
            role=ElementRole.BUTTON,
            source=ElementSource.VISION,
        )

    @classmethod
    def __observation(cls, *, elements: tuple[PerceivedElement, ...]) -> ScreenObservation:
        """
        Build a minimal screen observation.
        """

        return ScreenObservation(
            activity="app",
            elements=elements,
            hashes=ScreenHashBundle(
                xml_hash="0" * 16,
                visual_hash="0" * 16,
                interaction_hash="0" * 16,
            ),
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
            target="cv_1",
            confidence=1.0,
            action_type=ActionType.TAP,
            rationale="tap observed button",
        )

        result = await service.localize(
            image=b"",
            action=action,
            budget=self.__budget(),
            observation=self.__observation(elements=(element,)),
        )

        self.assertIsNotNone(result.bounds)
        self.assertEqual(result.bounds, element.bounds)
        self.assertEqual(result.status, LocalizationStatus.RESOLVED)

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
                source=CoordinateSource.XML,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )
        action = Action(
            confidence=1.0,
            rationale="test",
            action_type=ActionType.TAP,
            target="Alright, got it button",
            natural_language_target="Alright, got it button",
            bounds=Bounds(
                x=10,
                y=20,
                width=300,
                height=80,
                source=CoordinateSource.MODEL,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )

        with self.assertLogs("fathom.core.perception.localization", level="INFO") as captured:
            result = await service.localize(
                image=b"",
                action=action,
                budget=self.__budget(),
                observation=self.__observation(elements=(element,)),
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
        self.assertEqual(record.__dict__["localization.method"], "blind_model_bounds")
        self.assertEqual(record.__dict__["localization.source"], CoordinateSource.MODEL.value)
        self.assertEqual(record.__dict__["action.target"], action.target)
        self.assertEqual(
            record.__dict__["action.natural_language_target"],
            action.natural_language_target,
        )
        self.assertIsNone(record.__dict__["selected.element"])
        self.assertEqual(record.__dict__["candidate.count"], 1)

    async def test_model_bounds_without_perceived_overlap_dispatched_blindly(self) -> None:
        """
        Model bounds with no perception overlap dispatch and tag the result ``MODEL``.
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
            confidence=1.0,
            target="visible button",
            action_type=ActionType.TAP,
            rationale="tap visible button",
            bounds=Bounds(
                x=10,
                y=10,
                width=100,
                height=50,
                source=CoordinateSource.MODEL,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )

        result = await service.localize(
            image=b"",
            action=action,
            budget=self.__budget(),
            observation=self.__observation(elements=(element,)),
        )

        self.assertIsNotNone(result.bounds)
        self.assertEqual(result.bounds.x, 10)
        self.assertEqual(result.bounds.y, 10)
        self.assertEqual(result.bounds.source, CoordinateSource.MODEL)
        self.assertEqual(result.status, LocalizationStatus.RESOLVED)

    async def test_swipe_without_label_resolves_as_gesture(self) -> None:
        """
        Free viewport gestures do not require element localization.
        """

        service = TargetLocalizationService()

        action = Action(
            target="page",
            confidence=1.0,
            rationale="scroll",
            action_type=ActionType.SWIPE_UP,
        )

        result = await service.localize(
            image=b"",
            action=action,
            budget=self.__budget(),
            observation=self.__observation(elements=()),
        )

        self.assertEqual(result.status, LocalizationStatus.RESOLVED)


class FragmentedOcrTargetResolutionTest(unittest.IsolatedAsyncioTestCase):
    """
    Cascade resolution when OCR fragments a multi-word target across tokens.
    """

    @staticmethod
    def __ocr(
        *,
        text: str,
        x: int,
        y: int,
        width: int,
        height: int,
        identifier: str,
    ) -> PerceivedElement:
        """
        Build one OCR-sourced perceived element with stable defaults.
        """

        return PerceivedElement(
            text=text,
            parent=None,
            tappable=False,
            confidence=0.95,
            label_id=identifier,
            identifier=identifier,
            role=ElementRole.TEXT,
            source=ElementSource.OCR,
            bounds=Bounds(
                x=x,
                y=y,
                width=width,
                height=height,
                source=CoordinateSource.OCR,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )

    @staticmethod
    def __fragmented_row() -> Tuple[PerceivedElement, ...]:
        """
        Five OCR tokens on a single baseline forming one merged phrase.
        """

        ocr = FragmentedOcrTargetResolutionTest.__ocr

        return (
            ocr(text="Buy", x=327, y=2034, width=70, height=33, identifier="ocr_1"),
            ocr(text="tickets", x=405, y=2033, width=124, height=33, identifier="ocr_2"),
            ocr(text="from", x=540, y=2033, width=84, height=33, identifier="ocr_3"),
            ocr(text="$", x=637, y=2033, width=23, height=31, identifier="ocr_4"),
            ocr(text="0.00", x=664, y=2033, width=84, height=31, identifier="ocr_5"),
        )

    @staticmethod
    def __observation() -> ScreenObservation:
        """
        Observation carrying just the fragmented OCR row.
        """

        return ScreenObservation(
            activity="app",
            hashes=ScreenHashBundle(
                xml_hash="a" * 16,
                visual_hash="0" * 16,
                interaction_hash="b" * 16,
            ),
            elements=FragmentedOcrTargetResolutionTest.__fragmented_row(),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
        )

    @staticmethod
    def __action(*, target: str) -> Action:
        """
        Tap action with model bounds enclosing the fragmented OCR row.
        """

        return Action(
            confidence=1.0,
            target=target,
            action_type=ActionType.TAP,
            rationale="cascade replay",
            natural_language_target=target,
            bounds=Bounds(
                x=31,
                y=1989,
                width=937,
                height=73,
                source=CoordinateSource.MODEL,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )

    @staticmethod
    def __budget() -> LocalizationBudget:
        """
        Localization budget; per-attempt timeouts live on the adapter config.
        """

        return LocalizationBudget(vision=False, attempts=0, local=60_000, threshold=0.5)

    async def test_target_prefix_resolves_via_regional_evidence(self) -> None:
        """
        A target that is a prefix of the OCR phrase resolves via regional evidence.
        """

        service = TargetLocalizationService()
        with self.assertLogs("fathom.core.perception.localization", level="INFO") as captured:
            result = await service.localize(
                image=b"",
                budget=self.__budget(),
                observation=self.__observation(),
                action=self.__action(target="Buy tickets"),
            )

        self.assertIsNotNone(result.bounds)
        self.assertEqual(result.bounds.x, 327)
        self.assertEqual(result.bounds.y, 2033)
        self.assertEqual(result.bounds.width, 421)
        self.assertEqual(result.bounds.height, 34)
        self.assertEqual(result.status, LocalizationStatus.RESOLVED)
        self.assertEqual(result.bounds.source, CoordinateSource.MODEL_GROUNDED)

        regional = next(
            record
            for record in captured.records
            if record.__dict__.get("event") == "localization.regional_evidence.evaluated"
        )

        self.assertTrue(regional.__dict__["regional.resolved"])
        self.assertEqual(regional.__dict__["regional.decision"], "RESOLVED")
        self.assertGreater(regional.__dict__["regional.metrics.fused"], 0.55)
        self.assertEqual(regional.__dict__["regional.cluster.token_count"], 5)
        self.assertEqual(regional.__dict__["regional.in_region_token_count"], 5)
        self.assertAlmostEqual(regional.__dict__["regional.metrics.recall"], 1.0)
        self.assertGreater(regional.__dict__["regional.metrics.containment"], 0.5)

    async def test_full_phrase_target_resolves_via_ocr_phrase_fallback(self) -> None:
        """
        A target carrying every word of the OCR phrase resolves at the OCR stage.
        """

        service = TargetLocalizationService()
        with self.assertLogs("fathom.core.perception.localization", level="INFO") as captured:
            result = await service.localize(
                image=b"",
                budget=self.__budget(),
                observation=self.__observation(),
                action=self.__action(target="Buy tickets from $0.00"),
            )

        self.assertIsNotNone(result.bounds)
        self.assertEqual(result.bounds.x, 327)
        self.assertEqual(result.bounds.y, 2033)
        self.assertEqual(result.bounds.width, 421)
        self.assertEqual(result.bounds.height, 34)
        self.assertEqual(result.status, LocalizationStatus.RESOLVED)
        self.assertEqual(result.bounds.source, CoordinateSource.OCR)

        phrase = next(
            record
            for record in captured.records
            if record.__dict__.get("event") == "localization.phrase_fallback.evaluated"
        )
        self.assertTrue(phrase.__dict__["phrase.matched"])
        self.assertGreater(phrase.__dict__["phrase.score"], 0.7)
        self.assertEqual(phrase.__dict__["phrase.token_count"], 5)
        self.assertEqual(phrase.__dict__["observation.ocr_token_count"], 5)


if __name__ == "__main__":
    unittest.main()
