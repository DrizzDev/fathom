from __future__ import annotations

import unittest
from typing import Tuple

from fathom.core.exceptions import OcrError
from fathom.core.perception.ocr import OcrEnsembleService
from fathom.interfaces.ocr import OcrPort
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.ocr import OcrConfidence, OcrResult, OcrToken
from fathom.schemas.screens import ScreenCapture


class _StaticOcr(OcrPort):
    """
    :class:`OcrPort` test double returning a frozen token tuple.

    Used by the dedup tests to drive the ensemble with deterministic
    member output so the merge algorithm — sort by raw score, suppress
    overlapping survivors — can be pinned without a real OCR provider.
    """

    def __init__(self, *, tokens: Tuple[OcrToken, ...], duration: int = 0) -> None:
        """
        Initialise with the token tuple every ``extract`` call returns
        and an optional duration that flows into :class:`OcrResult`.
        """

        self.__tokens = tokens
        self.__duration = duration

    async def extract(self, *, capture, budget):  # type: ignore[no-untyped-def]
        """
        Return the preconfigured :class:`OcrResult` verbatim. The
        capture and budget are ignored because the ensemble exercises
        the merge path, not the provider's image-decoding code.
        """

        _ = (capture, budget)
        return OcrResult(tokens=self.__tokens, duration=self.__duration)


class _FailingOcr(OcrPort):
    """
    :class:`OcrPort` test double that raises :class:`OcrError` on every
    call. Drives the ensemble's per-member exception isolation path.
    """

    async def extract(self, *, capture, budget):  # type: ignore[no-untyped-def]
        """
        Raise a non-retryable :class:`OcrError`. The ensemble must catch
        this, log a structured warning, and treat the member as silent
        so the remaining members can still contribute.
        """

        _ = (capture, budget)
        raise OcrError("boom", retryable=False)


class OcrEnsembleServiceTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the OCR ensemble's fan-out, merge, and per-member isolation.

    The ensemble fans out to every member concurrently, merges tokens
    by ``(text, IoU)`` dedup keeping the highest-scoring survivor, and
    treats any per-member exception as a silent miss. Tests cover the
    empty-membership short-circuit, overlap dedup, disjoint preservation,
    and failure isolation.
    """

    @staticmethod
    def __capture() -> ScreenCapture:
        """
        :class:`ScreenCapture` fixture; the ensemble does not decode the
        image bytes, so the payload is a placeholder.
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
        the local timeout window comfortably.
        """

        return PerceptionBudget(ocr=500, local=500, localization=500)

    @staticmethod
    def __bounds(*, x: int) -> Bounds:
        """
        :class:`Bounds` fixture parameterised on the x offset so the
        overlap/disjoint dedup tests can place tokens deliberately.
        """

        return Bounds(
            x=x,
            y=0,
            width=100,
            height=50,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            source=CoordinateSource.OCR,
        )

    @staticmethod
    def __token(*, text: str, score: float, bounds: Bounds) -> OcrToken:
        """
        :class:`OcrToken` fixture. Confidence is always ``HIGH`` band
        because the merge algorithm sorts by ``raw_score``, not band.
        """

        return OcrToken(
            text=text,
            bounds=bounds,
            confidence=OcrConfidence.HIGH,
            raw_score=score,
        )

    async def test_empty_ensemble_short_circuits(self) -> None:
        """
        With no members configured, ``extract`` must return an empty
        :class:`OcrResult` without dispatching any work.
        """

        result = await OcrEnsembleService().extract(
            capture=self.__capture(),
            budget=self.__budget(),
        )

        self.assertEqual(result.tokens, ())

    async def test_overlapping_text_collapses_to_highest_score(self) -> None:
        """
        Two members proposing the same text at the same coordinates must
        collapse to one survivor — the one with the higher ``raw_score``.
        """

        bounds = self.__bounds(x=10)
        high_score = self.__token(text="Hello", score=0.9, bounds=bounds)
        low_score = self.__token(text="hello", score=0.6, bounds=bounds)
        service = OcrEnsembleService(
            members=(
                _StaticOcr(tokens=(low_score,)),
                _StaticOcr(tokens=(high_score,)),
            ),
        )

        result = await service.extract(capture=self.__capture(), budget=self.__budget())

        self.assertEqual(len(result.tokens), 1)
        self.assertAlmostEqual(result.tokens[0].raw_score, 0.9)

    async def test_disjoint_tokens_both_survive(self) -> None:
        """
        Two members proposing the same text in non-overlapping regions
        must both survive merging — same text in different places is two
        distinct screen elements.
        """

        token_a = self.__token(text="Hello", score=0.9, bounds=self.__bounds(x=10))
        token_b = self.__token(text="Hello", score=0.85, bounds=self.__bounds(x=500))
        service = OcrEnsembleService(
            members=(
                _StaticOcr(tokens=(token_a,)),
                _StaticOcr(tokens=(token_b,)),
            ),
        )

        result = await service.extract(capture=self.__capture(), budget=self.__budget())

        self.assertEqual(len(result.tokens), 2)

    async def test_failing_member_is_isolated_from_ensemble(self) -> None:
        """
        A member that raises :class:`OcrError` must be silently dropped;
        the remaining members still produce their tokens. This is the
        ensemble's resilience contract — one bad provider cannot tear
        down the whole observation pipeline.
        """

        token = self.__token(text="Hello", score=0.9, bounds=self.__bounds(x=10))
        service = OcrEnsembleService(
            members=(_FailingOcr(), _StaticOcr(tokens=(token,))),
        )

        result = await service.extract(capture=self.__capture(), budget=self.__budget())

        self.assertEqual(len(result.tokens), 1)
