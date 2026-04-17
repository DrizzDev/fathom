from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Union

from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.results import GenerateResult

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
        thinking_level: Optional[str] = None,
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
            thinking_level: Optional override for model thinking depth
                (e.g. "minimal", "low", "medium", "high"). When None, uses
                the configured default.

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

    def with_bucket(self, name: str) -> "LLMPort":
        """
        Return a variant of this port whose cached-content state is
        isolated under ``name``.

        Providers without caching, or without bucket-based isolation,
        return ``self``. Providers with bucketed caches (e.g. Gemini)
        return a clone that shares underlying resources but routes
        cache reads/writes through the named bucket, preventing one
        caller subsystem from evicting another's entries.
        """

        del name
        return self

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Release resources.
        """

        raise NotImplementedError
