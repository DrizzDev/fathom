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
        cache_bucket: str = "default",
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
            cache_bucket: Caller namespace that isolates cache eviction when
                ``use_cache`` is true. Callers from different subsystems
                should pass distinct bucket names so one caller's hash
                cannot evict another's. Defaults to ``"default"``.

        Returns:
            GenerateResult with content and tool calls
        """

        raise NotImplementedError

    async def prewarm(
        self,
        *,
        system_instruction: Optional[str],
        tools: Optional[Dict[str, Any]] = None,
        cache_bucket: str = "default",
    ) -> None:
        """
        Prewarm provider-side prompt cache when supported.

        Default is a no-op. Providers that support cached content (e.g. Gemini)
        override this to pre-create the cache entry before the first generate call.

        Args:
            cache_bucket: Caller namespace routed to the underlying cache
                implementation's bucket isolation, if any.
        """

        return  # noqa: PLE0101

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Release resources.
        """

        raise NotImplementedError
