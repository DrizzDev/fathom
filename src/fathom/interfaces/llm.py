from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import GenerateResult

if TYPE_CHECKING:
    from fathom.schemas.configuration import LLMConfiguration

# Type alias for LLM prompt parts (text, images, or structured content)
PromptPart = Union[str, bytes, Dict[str, str]]


class LLMPort(ABC):
    """
    Abstract interface for language model interactions.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Name of the model being used.
        """

        raise NotImplementedError

    @abstractmethod
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
        Generate response from LLM.

        Args:
            prompt: List of text parts and image bytes
            system_instruction: Optional system prompt
            tools: Optional tool definitions
            conversation_history: Optional prior turns for multi-turn feedback loops.
                Each entry is a provider-neutral ConversationTurn.
                When provided, prompt is appended as the final user turn.
            structured_output: Optional vendor-neutral constrained-decoding contract.
                Adapters translate it into their provider's structured-output API.

        Returns:
            GenerateResult with content and tool calls
        """

        raise NotImplementedError

    async def prewarm(
        self,
        *,
        system_instruction: Optional[str],
        tools: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Prewarm provider-side prompt cache when supported.

        Default is a no-op. Providers that support cached content (e.g. Gemini)
        override this to pre-create the cache entry before the first generate call.
        """

        return  # noqa: PLE0101

    def derive(self, *, overrides: "LLMConfiguration") -> "LLMPort":
        """
        Spawn a sibling adapter sharing credentials and model, with only the explicitly-set
        fields of ``overrides`` applied on top.

        Default returns self; providers that support reconfiguration (e.g. Gemini) override
        this to build a genuinely independent sibling.
        """

        _ = overrides
        return self

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Release resources.
        """

        raise NotImplementedError
