from __future__ import annotations

import unittest
from typing import Any, Dict, Optional, Sequence

from fathom.core.services.abort.composite import CompositeAbortDetector
from fathom.core.services.abort.factory import AbortDetectorFactory
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import GenerateResult


class _NoopLLM(LLMPort):
    """
    Minimal LLM port double used purely for factory wiring assertions.
    """

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
        Return an empty result; factory tests never exercise this path.
        """

        _ = (
            tools,
            prompt,
            use_cache,
            structured_output,
            system_instruction,
            conversation_history,
        )
        return GenerateResult(content="")

    async def cleanup(self) -> None:
        """
        No-op cleanup for the stub.
        """

        return


class AbortDetectorFactoryTest(unittest.TestCase):
    """
    Pins the factory composition contract.
    """

    def test_build_returns_composite_detector(self) -> None:
        """
        Factory always returns the composite detector type.
        """

        detector = AbortDetectorFactory.build(llm=_NoopLLM())

        self.assertIsInstance(detector, CompositeAbortDetector)
