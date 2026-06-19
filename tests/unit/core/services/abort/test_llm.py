from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional, Sequence

from fathom.core.services.abort.llm import LLMAbortDetector
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.abort import (
    AbortConfidenceConfiguration,
    AbortDetectorConfiguration,
    AbortDetectorResponse,
)
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import GenerateResult


class _ScriptedLLM(LLMPort):
    """
    LLM port double returning a scripted response or raising a scripted error.
    """

    def __init__(
        self,
        *,
        content: Optional[str] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """
        Pre-seed the canned generate-result content or the exception to raise.
        """

        self.__error = error
        self.__content = content
        self.calls: List[Sequence[PromptPart]] = []
        self.structured_outputs: List[Optional[StructuredOutput]] = []

    @property
    def model_name(self) -> str:
        """
        Return the stub model name used in tests.
        """

        return "stub-model"

    async def generate(
        self,
        *,
        use_cache: bool,
        prompt: Sequence[PromptPart],
        tools: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        structured_output: Optional[StructuredOutput] = None,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
    ) -> GenerateResult:
        """
        Record the call and either raise the scripted error or return the canned content.
        """

        _ = use_cache, tools, system_instruction, conversation_history
        self.calls.append(prompt)
        self.structured_outputs.append(structured_output)

        if self.__error is not None:
            raise self.__error

        return GenerateResult(content=self.__content or "")

    async def cleanup(self) -> None:
        """
        No-op cleanup for the stub.
        """

        return


class LLMAbortDetectorAbortedTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins :meth:`LLMAbortDetector.aborted` for the happy primary-classifier path.
    """

    async def test_high_confidence_abort_is_honoured(self) -> None:
        """
        LLM verdict ``aborted=true`` above the floor flows through unchanged.
        """

        llm = _ScriptedLLM(content='{"aborted": true, "confidence": 0.95}')
        detector = LLMAbortDetector(llm=llm)

        decision = await detector.aborted(response="cancel the workflow")

        self.assertTrue(decision.aborted)
        self.assertFalse(decision.fallback)
        self.assertAlmostEqual(decision.confidence, 0.95)

    async def test_low_confidence_abort_is_gated_to_not_aborted(self) -> None:
        """
        LLM ``aborted=true`` below the floor is suppressed but raw confidence preserved.
        """

        llm = _ScriptedLLM(content='{"aborted": true, "confidence": 0.5}')
        detector = LLMAbortDetector(llm=llm)

        decision = await detector.aborted(response="hmm")

        self.assertFalse(decision.aborted)
        self.assertFalse(decision.fallback)
        self.assertAlmostEqual(decision.confidence, 0.5)

    async def test_aborted_call_passes_structured_output_contract(self) -> None:
        """
        Every classification call must pin :class:`AbortDetectorResponse` as
        the structured-output payload so the adapter constrains decoding.
        """

        llm = _ScriptedLLM(content='{"aborted": false, "confidence": 0.9}')
        detector = LLMAbortDetector(llm=llm)

        await detector.aborted(response="navigate to settings")

        self.assertEqual(len(llm.structured_outputs), 1)
        structured = llm.structured_outputs[0]
        self.assertIsInstance(structured, StructuredOutput)
        assert structured is not None
        self.assertIs(structured.payload, AbortDetectorResponse)

    async def test_negative_verdict_is_not_aborted(self) -> None:
        """
        Negative verdict from the LLM passes through unchanged.
        """

        llm = _ScriptedLLM(content='{"aborted": false, "confidence": 0.9}')
        detector = LLMAbortDetector(llm=llm)

        decision = await detector.aborted(response="navigate to settings")

        self.assertFalse(decision.aborted)
        self.assertFalse(decision.fallback)


class LLMAbortDetectorFailOpenTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins fail-open behavior for every supported failure mode.
    """

    async def test_empty_response_is_fail_open(self) -> None:
        """
        An empty operator response never calls the LLM and returns the safe default.
        """

        llm = _ScriptedLLM(content='{"aborted": true, "confidence": 1.0}')
        detector = LLMAbortDetector(llm=llm)

        decision = await detector.aborted(response="")

        self.assertEqual(llm.calls, [])
        self.assertFalse(decision.aborted)
        self.assertTrue(decision.fallback)

    async def test_llm_exception_is_fail_open(self) -> None:
        """
        Any LLM exception flips the decision to fallback so the composite can take over.
        """

        llm = _ScriptedLLM(error=RuntimeError("network down"))
        detector = LLMAbortDetector(llm=llm)

        decision = await detector.aborted(response="please stop")

        self.assertFalse(decision.aborted)
        self.assertTrue(decision.fallback)

    async def test_non_json_content_is_fail_open(self) -> None:
        """
        A non-JSON response is treated as classifier abstention.
        """

        llm = _ScriptedLLM(content="not json at all")
        detector = LLMAbortDetector(llm=llm)

        decision = await detector.aborted(response="cancel the run")

        self.assertFalse(decision.aborted)
        self.assertTrue(decision.fallback)

    async def test_schema_violation_is_fail_open(self) -> None:
        """
        JSON missing the required fields fails open.
        """

        llm = _ScriptedLLM(content='{"aborted": true}')
        detector = LLMAbortDetector(llm=llm)

        decision = await detector.aborted(response="please stop")

        self.assertFalse(decision.aborted)
        self.assertTrue(decision.fallback)

    async def test_confidence_floor_override_via_configuration(self) -> None:
        """
        Configuration override changes the floor applied to the LLM verdict.
        """

        llm = _ScriptedLLM(content='{"aborted": true, "confidence": 0.6}')
        detector = LLMAbortDetector(
            llm=llm,
            configuration=AbortDetectorConfiguration(
                confidence=AbortConfidenceConfiguration(floor=0.5),
            ),
        )

        decision = await detector.aborted(response="please stop")

        self.assertTrue(decision.aborted)


class LLMAbortDetectorWarmupTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the warmup path issues a tiny prompt against the underlying LLM.
    """

    async def test_warmup_invokes_llm_generate(self) -> None:
        """
        Warmup primes the model with a tiny prompt to minimise first-call latency.
        """

        llm = _ScriptedLLM(content='{"aborted": false, "confidence": 0.1}')
        detector = LLMAbortDetector(llm=llm)

        await detector.warmup()

        self.assertEqual(len(llm.calls), 1)

    async def test_warmup_swallows_llm_exception_silently(self) -> None:
        """
        Warmup failures are logged but never propagate to the caller.
        """

        llm = _ScriptedLLM(error=RuntimeError("timeout"))
        detector = LLMAbortDetector(llm=llm)

        await detector.warmup()

    async def test_warmup_pins_structured_output_contract(self) -> None:
        """
        Warmup must use the same structured-output payload as the live path so
        the provider primes the constrained-decoding plan.
        """

        llm = _ScriptedLLM(content='{"aborted": false, "confidence": 0.1}')
        detector = LLMAbortDetector(llm=llm)

        await detector.warmup()

        self.assertEqual(len(llm.structured_outputs), 1)
        structured = llm.structured_outputs[0]
        self.assertIsInstance(structured, StructuredOutput)
        assert structured is not None
        self.assertIs(structured.payload, AbortDetectorResponse)
