"""LLM port interface for language model interactions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from fathom.schemas.results import GenerateResult


class LLMPort(ABC):
    """Abstract interface for language model interactions."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model being used."""
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        *,
        prompt: List[Any],
        system_instruction: Optional[str] = None,
        tools: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        """
        Generate response from LLM.

        Args:
            prompt: List of text parts and image bytes
            system_instruction: Optional system prompt
            tools: Optional tool definitions

        Returns:
            GenerateResult with content and tool calls
        """
        raise NotImplementedError

    @abstractmethod
    async def cleanup(self) -> None:
        """Release resources."""
        raise NotImplementedError
