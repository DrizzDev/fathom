from __future__ import annotations

import unittest
from typing import Tuple

from fathom.core.perception.icon import IconEnsembleService
from fathom.interfaces.icon import IconDetectorPort
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.icon import IconDetectionResult, IconKind, IconMatch
from fathom.schemas.screens import ScreenCapture


class _StaticIcons(IconDetectorPort):
    """
    :class:`IconDetectorPort` test double returning a frozen match tuple.

    Drives the dedup tests with deterministic member output so the merge
    algorithm — sort by confidence, suppress overlapping survivors per
    :class:`IconKind` — can be pinned without OpenCV templates.
    """

    def __init__(self, *, matches: Tuple[IconMatch, ...]) -> None:
        """
        Initialise with the match tuple every ``detect`` call returns.
        """

        self.__matches = matches

    async def detect(self, *, capture, budget):  # type: ignore[no-untyped-def]
        """
        Return the preconfigured :class:`IconDetectionResult` verbatim.
        Capture and budget are ignored because the ensemble path is
        merge-only.
        """

        _ = (capture, budget)
        return IconDetectionResult(matches=self.__matches, duration=0)


class _RaisingIcons(IconDetectorPort):
    """
    :class:`IconDetectorPort` test double that raises on every call.
    Drives the ensemble's per-member exception isolation path.
    """

    async def detect(self, *, capture, budget):  # type: ignore[no-untyped-def]
        """
        Raise :class:`RuntimeError`. The ensemble must log a structured
        warning and treat the member as silent so other members still
        contribute matches.
        """

        _ = (capture, budget)
        raise RuntimeError("member failure")


class IconEnsembleServiceTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the icon ensemble's fan-out, merge, and isolation contract.

    The ensemble fans out to every member concurrently, merges matches
    by ``(kind, IoU)`` keeping the highest-confidence survivor, and
    isolates per-member failures. Tests cover the empty-membership
    short-circuit, overlap dedup, disjoint preservation, and failure
    isolation.
    """

    @staticmethod
    def __capture() -> ScreenCapture:
        """
        :class:`ScreenCapture` fixture; the ensemble does not decode
        bytes, so the payload is a placeholder.
        """

        return ScreenCapture(
            width=1000,
            height=2000,
            activity="app",
            image=b"PNG",
            timestamp=0,
        )

    @staticmethod
    def __budget() -> PerceptionBudget:
        """
        Permissive :class:`PerceptionBudget` so every member call clears
        the local timeout comfortably.
        """

        return PerceptionBudget(ocr=500, local=500, localization=500)

    @staticmethod
    def __bounds(*, x: int) -> Bounds:
        """
        Square :class:`Bounds` fixture parameterised on x offset, so the
        overlap and disjoint tests can place icons deterministically.
        """

        return Bounds(
            x=x,
            y=0,
            width=48,
            height=48,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            source=CoordinateSource.VIEWPORT,
        )

    @staticmethod
    def __match(*, kind: IconKind, confidence: float, bounds: Bounds) -> IconMatch:
        """
        :class:`IconMatch` fixture parameterised on kind and confidence
        so dedup behaviour can be pinned per :class:`IconKind`.
        """

        return IconMatch(kind=kind, bounds=bounds, confidence=confidence)

    async def test_empty_ensemble_short_circuits(self) -> None:
        """
        With no members configured, ``detect`` must return an empty
        :class:`IconDetectionResult` without dispatching any work.
        """

        result = await IconEnsembleService().detect(
            capture=self.__capture(),
            budget=self.__budget(),
        )

        self.assertEqual(result.matches, ())

    async def test_overlapping_same_kind_collapses_to_higher_confidence(self) -> None:
        """
        Same :class:`IconKind` overlapping above the IoU threshold must
        collapse to a single survivor — the higher-confidence one.
        """

        bounds = self.__bounds(x=10)
        high = self.__match(kind=IconKind.HEART, confidence=0.95, bounds=bounds)
        low = self.__match(kind=IconKind.HEART, confidence=0.6, bounds=bounds)
        service = IconEnsembleService(
            members=(
                _StaticIcons(matches=(low,)),
                _StaticIcons(matches=(high,)),
            ),
        )

        result = await service.detect(capture=self.__capture(), budget=self.__budget())

        self.assertEqual(len(result.matches), 1)
        self.assertAlmostEqual(result.matches[0].confidence, 0.95)

    async def test_disjoint_same_kind_matches_both_survive(self) -> None:
        """
        Same-kind matches in non-overlapping regions are different
        instances of the icon (e.g. two heart icons on a feed) and must
        both survive merging.
        """

        a = self.__match(
            kind=IconKind.HEART,
            confidence=0.9,
            bounds=self.__bounds(x=10),
        )
        b = self.__match(
            kind=IconKind.HEART,
            confidence=0.85,
            bounds=self.__bounds(x=500),
        )
        service = IconEnsembleService(members=(_StaticIcons(matches=(a, b)),))

        result = await service.detect(capture=self.__capture(), budget=self.__budget())

        self.assertEqual(len(result.matches), 2)

    async def test_failing_member_is_isolated_from_ensemble(self) -> None:
        """
        A member that raises must be silently dropped; the remaining
        members still contribute matches. One bad provider cannot tear
        down the whole observation pipeline.
        """

        match = self.__match(
            kind=IconKind.HEART,
            confidence=0.9,
            bounds=self.__bounds(x=10),
        )
        service = IconEnsembleService(
            members=(_RaisingIcons(), _StaticIcons(matches=(match,))),
        )

        result = await service.detect(capture=self.__capture(), budget=self.__budget())

        self.assertEqual(len(result.matches), 1)
