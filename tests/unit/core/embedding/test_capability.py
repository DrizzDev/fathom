"""Verify embedding-capability detection across LLM adapters."""

from __future__ import annotations

import unittest
from typing import Any, Dict, Optional, Sequence

from fathom.core.embedding.capability import EmbeddingCapability
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import GenerateResult


class _BareLLM(LLMPort):
    """
    LLM double that exposes no client, as a non-Gemini adapter does.
    """

    @property
    def model_name(self) -> str:
        """
        Model identifier this double reports.
        """

        return "bare-1"

    async def generate(
        self,
        *,
        use_cache: bool,
        prompt: Sequence[PromptPart],
        tools: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
        structured_output: Optional[StructuredOutput] = None,
    ) -> GenerateResult:
        """
        Unused by the capability check.
        """

        raise NotImplementedError

    async def cleanup(self) -> None:
        """
        Unused by the capability check.
        """


class _ClientLLM(_BareLLM):
    """
    LLM double that exposes a client attribute, initialized or not.
    """

    def __init__(self, *, client: Optional[object]) -> None:
        """
        Record the client this double exposes.
        """

        self.__client = client

    @property
    def client(self) -> Optional[object]:
        """
        Provider client a sibling embedding adapter would share.
        """

        return self.__client


class EmbeddingCapabilityTest(unittest.TestCase):
    """
    The embedding capability is decided by the typed port contract, never by attribute probing.
    """

    def test_adapter_without_a_client_is_not_capable(self) -> None:
        """
        An adapter that declares no client cannot back an embedding adapter.
        """

        self.assertFalse(EmbeddingCapability.available(llm=_BareLLM()))

    def test_adapter_with_an_initialized_client_is_capable(self) -> None:
        """
        An adapter exposing a live client can back an embedding adapter.
        """

        self.assertTrue(EmbeddingCapability.available(llm=_ClientLLM(client=object())))

    def test_adapter_whose_client_never_initialized_is_not_capable(self) -> None:
        """
        Declaring the attribute is not enough; a None client means construction never completed.
        """

        self.assertFalse(EmbeddingCapability.available(llm=_ClientLLM(client=None)))
