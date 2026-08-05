from __future__ import annotations

import unittest
from typing import Optional, Sequence

from fathom.constants.observation import KeyboardVisibility
from fathom.core.services.criterion import CriterionObserver
from fathom.interfaces.llm import LLMPort
from fathom.schemas.actions import Bounds
from fathom.schemas.criterion import CriterionAssessment, CriterionSource, CriterionVerdict
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    PerceivedElement,
    ScreenObservation,
)
from fathom.schemas.results import GenerateResult
from fathom.schemas.screens import ScreenHashBundle
from fathom.schemas.success import ObservationRequirement


class _StubLLM(LLMPort):
    """
    LLM port returning a queued CriterionAssessment JSON per call, or raising a queued error.
    """

    def __init__(self, *, verdicts: Sequence[object]) -> None:
        self.__verdicts = list(verdicts)
        self.calls: int = 0
        self.structured_payloads: list[object] = []

    @property
    def model_name(self) -> str:
        """
        Report a stub model name.
        """

        return "stub"

    async def generate(
        self,
        *,
        use_cache: bool,
        prompt,
        tools=None,
        structured_output: Optional[StructuredOutput] = None,
        system_instruction=None,
        conversation_history=None,
    ) -> GenerateResult:
        """
        Return the next queued verdict as structured JSON, or raise the queued exception.
        """

        _ = (use_cache, prompt, tools, system_instruction, conversation_history)
        self.structured_payloads.append(structured_output.payload if structured_output else None)
        item = self.__verdicts[self.calls]
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        content = CriterionAssessment(verdict=item, reason="stub reasoning").model_dump_json()
        return GenerateResult(content=content, tool_calls=[], metrics={})

    async def cleanup(self) -> None:
        """
        Release resources (no-op).
        """

        return None


def _observation(*, visual_hash: str = "h0") -> ScreenObservation:
    """
    Build a minimal ScreenObservation.
    """

    element = PerceivedElement(
        identifier="e0",
        bounds=Bounds(x=0, y=0, width=10, height=10),
        source=ElementSource.XML,
        role=ElementRole.TEXT,
        confidence=1.0,
        text="Home",
        tappable=False,
    )
    return ScreenObservation(
        activity="com.test.app",
        elements=(element,),
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


def _requirement(assertion: str = "the home screen is open") -> ObservationRequirement:
    return ObservationRequirement(assertion=assertion)


class CriterionObserverModelAdjudicationTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins model-based structured adjudication: the model's typed verdict drives the decision, never a
    token heuristic; failures degrade to UNCLEAR; only SATISFIED is cached.
    """

    async def test_satisfied_verdict_drives_decision_via_structured_output(self) -> None:
        """
        A satisfied model verdict yields SATISFIED with LLM source and clears the confidence floor.
        """

        llm = _StubLLM(verdicts=[CriterionVerdict.SATISFIED])
        checker = CriterionObserver(llm=llm)

        decision = await checker.check(
            workflow_id="wf",
            index=0,
            requirement=_requirement(),
            observation=_observation(),
        )

        self.assertEqual(decision.verdict, CriterionVerdict.SATISFIED)
        self.assertEqual(decision.source, CriterionSource.LLM)
        self.assertGreaterEqual(decision.confidence, 0.7)
        self.assertEqual(llm.structured_payloads[0], CriterionAssessment)

    async def test_unsatisfied_and_unclear_pass_through(self) -> None:
        """
        Unsatisfied is definitive; unclear stays below the confidence floor so it never advances.
        """

        checker = CriterionObserver(llm=_StubLLM(verdicts=[CriterionVerdict.UNSATISFIED]))
        unsat = await checker.check(
            workflow_id="wf", index=0, requirement=_requirement(), observation=_observation()
        )
        self.assertEqual(unsat.verdict, CriterionVerdict.UNSATISFIED)

        checker2 = CriterionObserver(llm=_StubLLM(verdicts=[CriterionVerdict.UNCLEAR]))
        unclear = await checker2.check(
            workflow_id="wf", index=0, requirement=_requirement(), observation=_observation()
        )
        self.assertEqual(unclear.verdict, CriterionVerdict.UNCLEAR)
        self.assertLess(unclear.confidence, 0.7)

    async def test_llm_failure_degrades_to_unclear(self) -> None:
        """
        An adjudication error degrades to UNCLEAR, never a false satisfaction.
        """

        checker = CriterionObserver(llm=_StubLLM(verdicts=[RuntimeError("provider down")]))
        decision = await checker.check(
            workflow_id="wf", index=0, requirement=_requirement(), observation=_observation()
        )
        self.assertEqual(decision.verdict, CriterionVerdict.UNCLEAR)
        self.assertEqual(decision.source, CriterionSource.LLM)

    async def test_satisfied_is_cached_unsatisfied_is_not(self) -> None:
        """
        A SATISFIED verdict serves from cache on the same screen; a non-satisfied one re-adjudicates.
        """

        satisfied = _StubLLM(verdicts=[CriterionVerdict.SATISFIED, CriterionVerdict.UNSATISFIED])
        checker = CriterionObserver(llm=satisfied)
        first = await checker.check(
            workflow_id="wf", index=0, requirement=_requirement(), observation=_observation()
        )
        second = await checker.check(
            workflow_id="wf", index=0, requirement=_requirement(), observation=_observation()
        )
        self.assertEqual(satisfied.calls, 1)
        self.assertEqual(first.source, CriterionSource.LLM)
        self.assertEqual(second.source, CriterionSource.CACHE)

        unsatisfied = _StubLLM(verdicts=[CriterionVerdict.UNSATISFIED, CriterionVerdict.SATISFIED])
        checker2 = CriterionObserver(llm=unsatisfied)
        await checker2.check(
            workflow_id="wf", index=0, requirement=_requirement(), observation=_observation()
        )
        again = await checker2.check(
            workflow_id="wf", index=0, requirement=_requirement(), observation=_observation()
        )
        self.assertEqual(unsatisfied.calls, 2)
        self.assertEqual(again.verdict, CriterionVerdict.SATISFIED)

    async def test_distinct_screen_hash_bypasses_cache(self) -> None:
        """
        A different visual hash forces a fresh adjudication even after a cached SATISFIED.
        """

        llm = _StubLLM(verdicts=[CriterionVerdict.SATISFIED, CriterionVerdict.UNSATISFIED])
        checker = CriterionObserver(llm=llm)
        await checker.check(
            workflow_id="wf",
            index=0,
            requirement=_requirement(),
            observation=_observation(visual_hash="hA"),
        )
        other = await checker.check(
            workflow_id="wf",
            index=0,
            requirement=_requirement(),
            observation=_observation(visual_hash="hB"),
        )
        self.assertEqual(llm.calls, 2)
        self.assertEqual(other.verdict, CriterionVerdict.UNSATISFIED)


if __name__ == "__main__":
    unittest.main()
