"""
Unit pins for :class:`CriterionObserver`.

Covers the symbolic match layer, the LLM fallback layer (mocked), the
action-criterion guard ("don't infer that actions happened"), and the
cache replay path.
"""

from __future__ import annotations

import unittest
from typing import Optional, Sequence
from unittest.mock import AsyncMock

from fathom.constants import ActionType
from fathom.constants.observation import KeyboardVisibility
from fathom.core.services.criterion import CriterionObserver
from fathom.interfaces.llm import LLMPort
from fathom.schemas.actions import Bounds
from fathom.schemas.criterion import CriterionSource, CriterionVerdict
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    PerceivedElement,
    ScreenObservation,
)
from fathom.schemas.results import GenerateResult
from fathom.schemas.screens import ScreenHashBundle
from fathom.schemas.subgoal import SubGoal


class _StubLLM(LLMPort):
    """
    Test-only LLM port returning a queued response per generate() call.
    """

    def __init__(self, *, responses: Sequence[str]) -> None:
        self.__responses = list(responses)
        self.calls: int = 0

    @property
    def model_name(self) -> str:
        return "stub"

    async def generate(
        self,
        *,
        use_cache: bool,
        prompt,
        tools=None,
        system_instruction=None,
        conversation_history=None,
    ) -> GenerateResult:
        self.calls += 1
        content = self.__responses[self.calls - 1] if self.__responses else ""
        return GenerateResult(content=content, tool_calls=[], metrics={})

    async def cleanup(self) -> None:
        return None


def _element(
    *,
    identifier: str,
    text: Optional[str],
) -> PerceivedElement:
    """
    Build a minimal :class:`PerceivedElement` carrying just text.
    """

    return PerceivedElement(
        identifier=identifier,
        bounds=Bounds(x=0, y=0, width=10, height=10),
        source=ElementSource.XML,
        role=ElementRole.TEXT,
        confidence=1.0,
        text=text,
        tappable=False,
    )


def _observation(*, texts: Sequence[str], visual_hash: str = "h0") -> ScreenObservation:
    """
    Build a :class:`ScreenObservation` whose elements carry the given texts.
    """

    elements = tuple(_element(identifier=f"e{idx}", text=text) for idx, text in enumerate(texts))
    return ScreenObservation(
        activity="com.test.app",
        elements=elements,
        hashes=ScreenHashBundle(
            visual_hash=visual_hash,
            xml_hash="0000000000000000",
            interaction_hash="0000000000000000",
        ),
        overlays=(),
        keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
        scroll=(),
        calls_to_action=(),
        focused=None,
    )


def _sub_goal(
    *,
    index: int,
    criterion: Optional[str],
    description: str = "test sub-goal",
    directive: Optional[ActionType] = ActionType.TAP,
) -> SubGoal:
    """
    Build a :class:`SubGoal` with the supplied criterion.
    """

    return SubGoal(
        index=index,
        description=description,
        directive=directive,
        criterion=criterion,
    )


class CriterionObserverSymbolicTest(unittest.IsolatedAsyncioTestCase):
    """
    Symbolic-layer pins: hit-ratio threshold, evidence capture, miss-then-fallback.
    """

    async def test_symbolic_hit_returns_satisfied_without_llm(self) -> None:
        """
        Criterion tokens present on the screen → SATISFIED, source=SYMBOLIC, no LLM call.
        """

        llm = _StubLLM(responses=[])
        checker = CriterionObserver(llm=llm)
        observation = _observation(
            texts=["Jars", "&", "Containers", "Top rated"],
        )
        sub_goal = _sub_goal(
            index=0,
            criterion="'Jars & containers' is visible on the screen.",
        )

        decision = await checker.check(
            workflow_id="wf-1",
            sub_goal=sub_goal,
            observation=observation,
        )

        self.assertEqual(decision.verdict, CriterionVerdict.SATISFIED)
        self.assertEqual(decision.source, CriterionSource.SYMBOLIC)
        self.assertGreaterEqual(decision.confidence, 0.85)
        self.assertIn("jars", decision.evidence)
        self.assertEqual(llm.calls, 0)

    async def test_symbolic_miss_falls_back_to_llm(self) -> None:
        """
        Too few criterion tokens match → LLM is invoked.
        """

        llm = _StubLLM(responses=["SATISFIED — page header confirms."])
        checker = CriterionObserver(llm=llm)
        observation = _observation(texts=["Home", "Categories"])
        sub_goal = _sub_goal(
            index=0,
            criterion="Login flow has been triggered and overlay is visible.",
        )

        decision = await checker.check(
            workflow_id="wf-1",
            sub_goal=sub_goal,
            observation=observation,
        )

        self.assertEqual(decision.source, CriterionSource.LLM)
        self.assertEqual(decision.verdict, CriterionVerdict.SATISFIED)
        self.assertEqual(llm.calls, 1)


class CriterionObserverLLMLayerTest(unittest.IsolatedAsyncioTestCase):
    """
    LLM-layer pins: tri-state mapping, error degrades to UNCLEAR, leading-line evidence.
    """

    async def test_llm_unsatisfied_response_propagates(self) -> None:
        """
        LLM reply starting with 'UNSATISFIED' maps to UNSATISFIED verdict.
        """

        llm = _StubLLM(responses=["UNSATISFIED — login card not present."])
        checker = CriterionObserver(llm=llm)
        observation = _observation(texts=["Home", "Categories"])
        sub_goal = _sub_goal(
            index=0,
            criterion="Login half card is displayed on the screen.",
        )

        decision = await checker.check(
            workflow_id="wf-1",
            sub_goal=sub_goal,
            observation=observation,
        )

        self.assertEqual(decision.verdict, CriterionVerdict.UNSATISFIED)
        self.assertEqual(decision.source, CriterionSource.LLM)

    async def test_llm_unclear_response_propagates(self) -> None:
        """
        Anything that is not SATISFIED / UNSATISFIED maps to UNCLEAR.
        """

        llm = _StubLLM(responses=["UNCLEAR — not enough evidence on this screen."])
        checker = CriterionObserver(llm=llm)
        observation = _observation(texts=["Home", "Categories"])
        sub_goal = _sub_goal(
            index=0,
            criterion="Login half card is displayed on the screen.",
        )

        decision = await checker.check(
            workflow_id="wf-1",
            sub_goal=sub_goal,
            observation=observation,
        )

        self.assertEqual(decision.verdict, CriterionVerdict.UNCLEAR)
        self.assertEqual(decision.source, CriterionSource.LLM)

    async def test_llm_exception_degrades_to_unclear(self) -> None:
        """
        Any exception from the LLM port is caught and degraded to UNCLEAR.
        """

        llm = AsyncMock(spec=LLMPort)
        llm.generate.side_effect = RuntimeError("provider down")
        checker = CriterionObserver(llm=llm)
        observation = _observation(texts=["Home"])
        sub_goal = _sub_goal(
            index=0,
            criterion="Login half card is displayed on the screen.",
        )

        decision = await checker.check(
            workflow_id="wf-1",
            sub_goal=sub_goal,
            observation=observation,
        )

        self.assertEqual(decision.verdict, CriterionVerdict.UNCLEAR)
        self.assertEqual(decision.source, CriterionSource.LLM)
        self.assertIn("RuntimeError", decision.notes or "")


class CriterionObserverCacheTest(unittest.IsolatedAsyncioTestCase):
    """
    Cache pins: positive-only cache. SATISFIED verdicts are cached;
    UNSATISFIED and UNCLEAR verdicts are not cached and re-invoke the LLM
    so a stuck loop cannot lock in a wrong verdict.
    """

    async def test_satisfied_verdict_is_cached(self) -> None:
        """
        Repeated check with SATISFIED first call hits cache on second call.
        """

        llm = _StubLLM(
            responses=[
                "SATISFIED — first call goes to LLM.",
                "UNSATISFIED — would change verdict if called.",
            ]
        )
        checker = CriterionObserver(llm=llm)
        observation = _observation(texts=["Home"], visual_hash="hX")
        sub_goal = _sub_goal(
            index=0,
            criterion="Login half card is displayed on the screen.",
        )

        first = await checker.check(workflow_id="wf-1", sub_goal=sub_goal, observation=observation)
        second = await checker.check(workflow_id="wf-1", sub_goal=sub_goal, observation=observation)

        self.assertEqual(first.verdict, CriterionVerdict.SATISFIED)
        self.assertEqual(first.source, CriterionSource.LLM)
        self.assertEqual(second.verdict, CriterionVerdict.SATISFIED)
        self.assertEqual(second.source, CriterionSource.CACHE)
        self.assertEqual(llm.calls, 1)

    async def test_unsatisfied_verdict_is_not_cached(self) -> None:
        """
        Repeated check with UNSATISFIED first call re-invokes the LLM
        — positive-only cache guarantees a stuck loop cannot lock in a
        wrong UNSATISFIED verdict against a screen the agent has actually
        transitioned past.
        """

        llm = _StubLLM(
            responses=[
                "UNSATISFIED — first call.",
                "SATISFIED — second call sees post-state.",
            ]
        )
        checker = CriterionObserver(llm=llm)
        observation = _observation(texts=["Home"], visual_hash="hX")
        sub_goal = _sub_goal(
            index=0,
            criterion="Login half card is displayed on the screen.",
        )

        first = await checker.check(workflow_id="wf-1", sub_goal=sub_goal, observation=observation)
        second = await checker.check(workflow_id="wf-1", sub_goal=sub_goal, observation=observation)

        self.assertEqual(first.verdict, CriterionVerdict.UNSATISFIED)
        self.assertEqual(first.source, CriterionSource.LLM)
        self.assertEqual(second.verdict, CriterionVerdict.SATISFIED)
        self.assertEqual(second.source, CriterionSource.LLM)
        self.assertEqual(llm.calls, 2)

    async def test_unclear_verdict_is_not_cached(self) -> None:
        """
        Repeated check with UNCLEAR first call re-invokes the LLM.
        """

        llm = _StubLLM(
            responses=[
                "UNCLEAR — not enough evidence on first call.",
                "SATISFIED — second call resolves.",
            ]
        )
        checker = CriterionObserver(llm=llm)
        observation = _observation(texts=["Home"], visual_hash="hX")
        sub_goal = _sub_goal(
            index=0,
            criterion="Login half card is displayed on the screen.",
        )

        first = await checker.check(workflow_id="wf-1", sub_goal=sub_goal, observation=observation)
        second = await checker.check(workflow_id="wf-1", sub_goal=sub_goal, observation=observation)

        self.assertEqual(first.verdict, CriterionVerdict.UNCLEAR)
        self.assertEqual(first.source, CriterionSource.LLM)
        self.assertEqual(second.verdict, CriterionVerdict.SATISFIED)
        self.assertEqual(second.source, CriterionSource.LLM)
        self.assertEqual(llm.calls, 2)

    async def test_distinct_screen_hash_bypasses_cache(self) -> None:
        """
        A different visual_hash forces a fresh check even for cached SATISFIED.
        """

        llm = _StubLLM(
            responses=[
                "SATISFIED — screen A.",
                "UNSATISFIED — screen B does not have it.",
            ]
        )
        checker = CriterionObserver(llm=llm)
        sub_goal = _sub_goal(
            index=0,
            criterion="Login half card is displayed on the screen.",
        )

        first = await checker.check(
            workflow_id="wf-1",
            sub_goal=sub_goal,
            observation=_observation(texts=["Home"], visual_hash="hA"),
        )
        second = await checker.check(
            workflow_id="wf-1",
            sub_goal=sub_goal,
            observation=_observation(texts=["Home"], visual_hash="hB"),
        )

        self.assertEqual(first.verdict, CriterionVerdict.SATISFIED)
        self.assertEqual(second.verdict, CriterionVerdict.UNSATISFIED)
        self.assertEqual(llm.calls, 2)


class CriterionObserverEmptyCriterionTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins for sub-goals lacking criterion text.
    """

    async def test_missing_criterion_returns_unclear(self) -> None:
        """
        Sub-goal with no criterion and no description text → UNCLEAR.
        """

        llm = _StubLLM(responses=[])
        checker = CriterionObserver(llm=llm)
        sub_goal = SubGoal(index=0, description="", criterion=None, directive=None)

        decision = await checker.check(
            workflow_id="wf-1",
            sub_goal=sub_goal,
            observation=_observation(texts=["Home"]),
        )

        self.assertEqual(decision.verdict, CriterionVerdict.UNCLEAR)
        self.assertEqual(llm.calls, 0)


if __name__ == "__main__":
    unittest.main()
