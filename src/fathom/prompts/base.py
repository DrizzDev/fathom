from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class PromptBuilder(ABC):
    """
    Abstract base class for building model-specific system prompts.
    """

    @abstractmethod
    def build(
        self,
        mode: str = "default",
        intent: str = "",
        hints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Constructs the stable system instruction string (for caching).
        """

        raise NotImplementedError

    @abstractmethod
    def build_user_context(
        self,
        history: Optional[Any] = None,
        memory: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Constructs the dynamic user context string.
        """

        raise NotImplementedError

    @abstractmethod
    def build_task_instructions(
        self,
        intent: str,
        hints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Constructs the dynamic task instructions (for user message).
        """

        raise NotImplementedError
