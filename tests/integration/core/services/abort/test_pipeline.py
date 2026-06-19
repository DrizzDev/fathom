from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional, Sequence

from fathom.core.services.abort.factory import AbortDetectorFactory
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.abort import AbortDetectorConfiguration
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
        Pre-seed the canned response content or the exception to raise on every call.
        """

        self.__error = error
        self.__content = content
        self.calls: List[Sequence[PromptPart]] = []

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

        _ = (
            tools,
            use_cache,
            structured_output,
            system_instruction,
            conversation_history,
        )
        self.calls.append(prompt)

        if self.__error is not None:
            raise self.__error

        return GenerateResult(content=self.__content or "")

    async def cleanup(self) -> None:
        """
        No-op cleanup for the stub.
        """

        return


class AbortDetectorPipelineHappyPathTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins composite + LLM pipeline behavior when the LLM returns a clean verdict.
    """

    async def test_primary_high_confidence_abort_is_honoured(self) -> None:
        """
        Composite delegates to LLM and returns the LLM verdict directly when confident.
        """

        llm = _ScriptedLLM(content='{"aborted": true, "confidence": 0.93}')
        detector = AbortDetectorFactory.build(llm=llm)

        decision = await detector.aborted(response="cancel the workflow")

        self.assertTrue(decision.aborted)
        self.assertFalse(decision.fallback)
        self.assertEqual(len(llm.calls), 1)

    async def test_primary_non_abort_is_honoured(self) -> None:
        """
        Composite trusts the LLM's negative verdict without consulting the fallback.
        """

        llm = _ScriptedLLM(content='{"aborted": false, "confidence": 0.9}')
        detector = AbortDetectorFactory.build(llm=llm)

        decision = await detector.aborted(response="navigate to settings")

        self.assertFalse(decision.aborted)
        self.assertFalse(decision.fallback)


class AbortDetectorPipelineFallbackTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins composite + heuristic pipeline when the LLM fails for any reason.
    """

    async def test_llm_outage_routes_to_heuristic_abort(self) -> None:
        """
        LLM exception forces composite to consult the heuristic, which detects the abort.
        """

        llm = _ScriptedLLM(error=RuntimeError("backend down"))
        detector = AbortDetectorFactory.build(llm=llm)

        decision = await detector.aborted(response="close the execution")

        self.assertTrue(decision.aborted)
        self.assertTrue(decision.fallback)

    async def test_llm_outage_with_ui_directive_response_is_safe(self) -> None:
        """
        Even when the LLM is down a UI directive like 'tap on stop' must not abort.
        """

        llm = _ScriptedLLM(error=RuntimeError("backend down"))
        detector = AbortDetectorFactory.build(llm=llm)

        decision = await detector.aborted(response="tap on stop")

        self.assertFalse(decision.aborted)

    async def test_unparseable_response_routes_to_heuristic(self) -> None:
        """
        Non-JSON response is treated as classifier abstention; heuristic adjudicates.
        """

        llm = _ScriptedLLM(content="oops not json")
        detector = AbortDetectorFactory.build(llm=llm)

        decision = await detector.aborted(response="please stop the workflow")

        self.assertTrue(decision.aborted)
        self.assertTrue(decision.fallback)

    async def test_low_confidence_abort_routes_to_heuristic(self) -> None:
        """
        LLM aborted=true below floor returns fallback=False but composite trusts that.
        """

        llm = _ScriptedLLM(content='{"aborted": true, "confidence": 0.4}')
        detector = AbortDetectorFactory.build(llm=llm)

        decision = await detector.aborted(response="please stop")

        self.assertFalse(decision.aborted)
        self.assertFalse(decision.fallback)


class AbortDetectorPipelineEdgeCaseTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins safe-default behavior for empty and unrelated input across the whole pipeline.
    """

    async def test_empty_response_never_aborts(self) -> None:
        """
        Empty input short-circuits in the LLM detector before any network call.
        """

        llm = _ScriptedLLM(content='{"aborted": true, "confidence": 1.0}')
        detector = AbortDetectorFactory.build(llm=llm)

        decision = await detector.aborted(response="")

        self.assertFalse(decision.aborted)
        self.assertEqual(len(llm.calls), 0)

    async def test_warmup_propagates_to_both_layers(self) -> None:
        """
        Warmup primes the LLM exactly once across the composite tree.
        """

        llm = _ScriptedLLM(content='{"aborted": false, "confidence": 0.1}')
        detector = AbortDetectorFactory.build(llm=llm)

        await detector.warmup()

        self.assertEqual(len(llm.calls), 1)

    async def test_configuration_propagates_to_primary(self) -> None:
        """
        Configuration override changes the confidence floor used by the primary detector.
        """

        llm = _ScriptedLLM(content='{"aborted": true, "confidence": 0.6}')
        configuration = AbortDetectorConfiguration.model_validate({"confidence": {"floor": 0.5}})
        detector = AbortDetectorFactory.build(llm=llm, configuration=configuration)

        decision = await detector.aborted(response="please stop")

        self.assertTrue(decision.aborted)
