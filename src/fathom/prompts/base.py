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
        intent: str,
        history: Optional[Any] = None,
        hints: Optional[Dict[str, Any]] = None,
        memory: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Constructs the final system instruction string.
        """

        raise NotImplementedError
