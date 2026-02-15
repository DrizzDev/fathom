"""LLM port interface for language model interactions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from fathom.schemas.results import AnalysisResult


class LLMPort(ABC):
    """Abstract interface for language model interactions."""

    @abstractmethod
    async def analyze(
        self,
        *,
        system_instruction: str,
        user_content: List[Any],
        tools: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Analyze content with LLM.

        Args:
            system_instruction: System prompt
            user_content: List of text strings and image bytes
            tools: Optional tool definitions for function calling

        Returns:
            AnalysisResult with reasoning, action, and metrics
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Release resources."""
        pass
