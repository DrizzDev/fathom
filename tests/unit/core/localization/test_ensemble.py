from __future__ import annotations

import asyncio
import unittest
from typing import Optional, Tuple

from fathom.constants.observation import KeyboardVisibility
from fathom.core.localization.ensemble import EnsembleLocalizerService
from fathom.interfaces.localization import TargetLocalizerPort
from fathom.schemas.actions import Action, Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.budgets import LocalizationBudget
from fathom.schemas.localization import LocalizationProposal
from fathom.schemas.observation import KeyboardObservation, ScreenObservation
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle


class _FakeLocalizer(TargetLocalizerPort):
    """
    :class:`TargetLocalizerPort` test double with three operating modes:
    return a frozen proposal, sleep past the budget to force a timeout,
    or raise to drive the per-member exception path. The mode is fixed
    at construction; the test parameterises ``locate`` indirectly via
    its initialiser flags.
    """

    def __init__(
        self,
        *,
        name: str,
        proposal: Optional[LocalizationProposal] = None,
        sleep: float = 0.0,
        raise_error: bool = False,
    ) -> None:
        """
        Pick exactly one of: return ``proposal`` immediately, ``sleep``
        for the given seconds (to drive ``asyncio.wait_for`` timeout),
        or ``raise_error`` to exercise the catch path. All three are
        deliberately permissible together so the test can layer them.
        """

        self.__name = name
        self.__proposal = proposal
        self.__sleep = sleep
        self.__raise = raise_error

    @property
    def name(self) -> str:
        """
        Stable identifier surfaced in structured logs and in the fused
        consensus proposal's ``source`` field.
        """

        return self.__name

    async def locate(self, *, action, observation, capture, budget):  # type: ignore[no-untyped-def]
        """
        Apply the configured mode in order: sleep, then raise, then
        return the proposal. Action / observation / capture / budget
        are accepted but ignored — the ensemble dispatch path is what's
        under test, not the member's own logic.
        """

        _ = (action, observation, capture, budget)
        if self.__sleep:
            await asyncio.sleep(self.__sleep)
        if self.__raise:
            raise RuntimeError("member failure")
        return self.__proposal


class EnsembleLocalizerServiceTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the ensemble vision-localizer consensus contract.

    :class:`EnsembleLocalizerService` fans out to every member
    concurrently and only commits when two or more members agree above
    the IoU floor. The tests cover: empty membership short-circuit,
    two-member consensus, disjoint-proposal rejection, per-member
    exception isolation, per-member timeout, and the quorum floor
    (single agreeing member is not consensus).
    """

    @staticmethod
    def __bounds(*, x: int, y: int, width: int, height: int) -> Bounds:
        """
        Pixel-space :class:`Bounds` fixture in viewport coordinates. The
        x/y/width/height parameters let each test place proposals so the
        IoU dedup behaviour can be asserted explicitly.
        """

        return Bounds(
            x=x,
            y=y,
            width=width,
            height=height,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            source=CoordinateSource.MODEL,
        )

    @staticmethod
    def __capture() -> ScreenCapture:
        """
        :class:`ScreenCapture` fixture. The ensemble does not decode the
        image bytes; the payload is a placeholder.
        """

        return ScreenCapture(
            width=1000,
            height=2000,
            activity="app",
            image=b"PNG",
            timestamp=0,
        )

    @staticmethod
    def __action() -> Action:
        """
        :class:`Action` fixture with a non-empty target. Required so
        the ensemble's pre-flight target check does not short-circuit.
        """

        return Action(  # type: ignore[arg-type]
            action_type="tap",
            target="Continue",
            rationale="test",
            confidence=1.0,
        )

    @staticmethod
    def __budget() -> LocalizationBudget:
        """
        Permissive :class:`LocalizationBudget`. The timeout-isolation
        test overrides ``local`` so the slow member exceeds it.
        """

        return LocalizationBudget(vision=True, attempts=2, local=2000, threshold=0.5)

    @staticmethod
    def __observation() -> ScreenObservation:
        """
        Minimal :class:`ScreenObservation` fixture. Some members may
        consume it (the layout localizer reads OCR-sourced elements);
        the fake member ignores it entirely.
        """

        return ScreenObservation(
            activity="app",
            hashes=ScreenHashBundle(
                visual_hash="0" * 16,
                xml_hash="a" * 16,
                interaction_hash="b" * 16,
            ),
            elements=(),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
        )

    @staticmethod
    def __proposal(*, source: str, bounds: Bounds, confidence: float) -> LocalizationProposal:
        """
        :class:`LocalizationProposal` fixture parameterised on source
        name, bounds, and confidence so the consensus tests can drive
        each member's response explicitly.
        """

        return LocalizationProposal(bounds=bounds, source=source, confidence=confidence)

    @staticmethod
    def __members(*entries: TargetLocalizerPort) -> Tuple[TargetLocalizerPort, ...]:
        """
        Convenience tuple constructor over variadic member arguments —
        keeps the test bodies free of explicit tuple-conversion noise.
        """

        return tuple(entries)

    async def test_returns_none_when_no_members_configured(self) -> None:
        """
        An empty ensemble short-circuits and returns no proposal.
        """

        service = EnsembleLocalizerService(members=())

        result = await service.locate(
            action=self.__action(),
            observation=self.__observation(),
            capture=self.__capture(),
            budget=self.__budget(),
        )

        self.assertIsNone(result)

    async def test_consensus_returned_when_two_members_agree(self) -> None:
        """
        Two members whose bounds overlap above the agreement floor produce a consensus proposal.
        """

        agreeing = self.__bounds(x=100, y=100, width=200, height=100)
        nearby = self.__bounds(x=110, y=110, width=200, height=100)
        service = EnsembleLocalizerService(
            members=self.__members(
                _FakeLocalizer(
                    name="a",
                    proposal=self.__proposal(source="a", bounds=agreeing, confidence=0.9),
                ),
                _FakeLocalizer(
                    name="b",
                    proposal=self.__proposal(source="b", bounds=nearby, confidence=0.8),
                ),
            ),
            minimum_agreeing=2,
        )

        result = await service.locate(
            action=self.__action(),
            observation=self.__observation(),
            capture=self.__capture(),
            budget=self.__budget(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.source, "a,b")
        self.assertGreater(result.confidence, 0.0)

    async def test_no_consensus_when_bounds_disjoint(self) -> None:
        """
        Two members proposing disjoint bounds yield no consensus.
        """

        far_left = self.__bounds(x=0, y=0, width=50, height=50)
        far_right = self.__bounds(x=900, y=1800, width=50, height=50)
        service = EnsembleLocalizerService(
            members=self.__members(
                _FakeLocalizer(
                    name="a",
                    proposal=self.__proposal(source="a", bounds=far_left, confidence=0.9),
                ),
                _FakeLocalizer(
                    name="b",
                    proposal=self.__proposal(source="b", bounds=far_right, confidence=0.9),
                ),
            ),
            minimum_agreeing=2,
        )

        result = await service.locate(
            action=self.__action(),
            observation=self.__observation(),
            capture=self.__capture(),
            budget=self.__budget(),
        )

        self.assertIsNone(result)

    async def test_member_exception_does_not_break_ensemble(self) -> None:
        """
        Member exceptions are swallowed and treated as missing proposals.
        """

        good = self.__bounds(x=10, y=10, width=20, height=20)
        service = EnsembleLocalizerService(
            members=self.__members(
                _FakeLocalizer(name="boom", raise_error=True),
                _FakeLocalizer(
                    name="a",
                    proposal=self.__proposal(source="a", bounds=good, confidence=0.9),
                ),
                _FakeLocalizer(
                    name="b",
                    proposal=self.__proposal(source="b", bounds=good, confidence=0.9),
                ),
            ),
            minimum_agreeing=2,
        )

        result = await service.locate(
            action=self.__action(),
            observation=self.__observation(),
            capture=self.__capture(),
            budget=self.__budget(),
        )

        self.assertIsNotNone(result)

    async def test_member_timeout_is_treated_as_no_proposal(self) -> None:
        """
        Members that exceed the local budget are dropped via wait_for timeout.
        """

        good = self.__bounds(x=10, y=10, width=20, height=20)
        service = EnsembleLocalizerService(
            members=self.__members(
                _FakeLocalizer(
                    name="slow",
                    proposal=self.__proposal(source="slow", bounds=good, confidence=0.5),
                    sleep=0.5,
                ),
                _FakeLocalizer(
                    name="a",
                    proposal=self.__proposal(source="a", bounds=good, confidence=0.9),
                ),
            ),
            minimum_agreeing=2,
        )
        budget = LocalizationBudget(vision=True, attempts=1, local=10, threshold=0.5)

        result = await service.locate(
            action=self.__action(),
            observation=self.__observation(),
            capture=self.__capture(),
            budget=budget,
        )

        self.assertIsNone(result)

    async def test_single_high_confidence_proposal_is_accepted_when_others_are_silent(
        self,
    ) -> None:
        """
        A lone high-confidence proposal is executable when no other member contradicts it.
        """

        good = self.__bounds(x=10, y=10, width=20, height=20)
        service = EnsembleLocalizerService(
            members=self.__members(
                _FakeLocalizer(
                    name="a",
                    proposal=self.__proposal(source="a", bounds=good, confidence=0.99),
                ),
                _FakeLocalizer(name="silent", proposal=None),
            ),
            minimum_agreeing=2,
        )

        result = await service.locate(
            action=self.__action(),
            observation=self.__observation(),
            capture=self.__capture(),
            budget=self.__budget(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.source, "a")

    async def test_single_low_confidence_proposal_yields_no_consensus(self) -> None:
        """
        A lone proposal below the single-proposal confidence floor is rejected.
        """

        good = self.__bounds(x=10, y=10, width=20, height=20)
        service = EnsembleLocalizerService(
            members=self.__members(
                _FakeLocalizer(
                    name="a",
                    proposal=self.__proposal(source="a", bounds=good, confidence=0.7),
                ),
                _FakeLocalizer(name="silent", proposal=None),
            ),
            minimum_agreeing=2,
        )

        result = await service.locate(
            action=self.__action(),
            observation=self.__observation(),
            capture=self.__capture(),
            budget=self.__budget(),
        )

        self.assertIsNone(result)
