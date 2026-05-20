from __future__ import annotations

import unittest

from fathom.adapters.localization.document.layout import DocumentAiLayoutLocalizer
from fathom.constants import ActionType
from fathom.schemas.actions import Action, Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.budgets import LocalizationBudget
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    PerceivedElement,
    ScreenObservation,
)
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle


class DocumentAiLayoutLocalizerTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the ensemble-member contract for the OCR-token layout localizer.

    The localizer scans the already-merged :class:`ScreenObservation` for an
    OCR-sourced :class:`PerceivedElement` whose normalized text matches the
    action target. The pins cover the four routing decisions: match, wrong
    source, text mismatch, and empty target short-circuit.
    """

    @staticmethod
    def __bounds() -> Bounds:
        """
        Pixel-space :class:`Bounds` for the fixture OCR token. Values are
        arbitrary; only the bounds being non-empty matters.
        """

        return Bounds(
            x=10,
            y=20,
            width=100,
            height=40,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            source=CoordinateSource.OCR,
        )

    @staticmethod
    def __action(*, target: str) -> Action:
        """
        :class:`Action` fixture parameterised on the semantic target string
        so each test can drive a different routing branch.
        """

        return Action(
            action_type=ActionType.TAP,
            target=target,
            rationale="t",
            confidence=1.0,
        )

    @staticmethod
    def __capture() -> ScreenCapture:
        """
        :class:`ScreenCapture` fixture; the layout localizer does not read
        the image bytes, so the payload is a placeholder.
        """

        return ScreenCapture(
            width=1000,
            height=2000,
            activity="app",
            image=b"PNG",
            timestamp=0,
        )

    @staticmethod
    def __budget() -> LocalizationBudget:
        """
        :class:`LocalizationBudget` with paid vision disabled. The layout
        localizer is a local-only ensemble member and ignores the budget.
        """

        return LocalizationBudget(vision=False, attempts=0, local=500, threshold=0.5)

    def __observation(
        self,
        *,
        text: str,
        source: ElementSource = ElementSource.OCR,
    ) -> ScreenObservation:
        """
        Single-element :class:`ScreenObservation` fixture parameterised on
        the perceived-element text and source. Other text-mismatch tests
        keep the source as OCR; the wrong-source test flips it to XML to
        verify the localizer's source filter.
        """

        return ScreenObservation(
            activity="app",
            hashes=ScreenHashBundle(
                visual_hash="0" * 16,
                xml_hash="a" * 16,
                interaction_hash="b" * 16,
            ),
            elements=(
                PerceivedElement(
                    identifier="ocr_1",
                    text=text,
                    parent=None,
                    bounds=self.__bounds(),
                    source=source,
                    role=ElementRole.TEXT,
                    confidence=0.9,
                    tappable=False,
                ),
            ),
            keyboard=KeyboardObservation(visible=False),
        )

    async def test_matching_ocr_token_produces_proposal(self) -> None:
        """
        An OCR token whose normalized text equals the action target must
        produce a proposal sourced as ``document.ai.layout``.
        """

        result = await DocumentAiLayoutLocalizer().locate(
            action=self.__action(target="Continue"),
            capture=self.__capture(),
            budget=self.__budget(),
            observation=self.__observation(text="continue"),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.source, "document.ai.layout")

    async def test_non_ocr_source_does_not_match(self) -> None:
        """
        Only OCR-sourced elements are eligible. The localizer filters by
        source so XML/accessibility text never triggers a layout proposal
        even when the surface form matches.
        """

        result = await DocumentAiLayoutLocalizer().locate(
            action=self.__action(target="Continue"),
            capture=self.__capture(),
            budget=self.__budget(),
            observation=self.__observation(text="Continue", source=ElementSource.XML),
        )

        self.assertIsNone(result)

    async def test_target_text_mismatch_returns_none(self) -> None:
        """
        OCR tokens whose normalized text does not equal the target must be
        skipped silently rather than returning a low-confidence proposal.
        """

        result = await DocumentAiLayoutLocalizer().locate(
            action=self.__action(target="Continue"),
            capture=self.__capture(),
            budget=self.__budget(),
            observation=self.__observation(text="Cancel"),
        )

        self.assertIsNone(result)

    async def test_blank_target_short_circuits(self) -> None:
        """
        A blank or whitespace-only target text must short-circuit before
        any matching is attempted; the localizer should not match the
        empty string against tokens.
        """

        result = await DocumentAiLayoutLocalizer().locate(
            action=self.__action(target="   "),
            capture=self.__capture(),
            budget=self.__budget(),
            observation=self.__observation(text="Continue"),
        )

        self.assertIsNone(result)
