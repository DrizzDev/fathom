from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from fathom.schemas.tools import AllowedTools


class PromptBuilder(ABC):
    """Abstract base class for building model-specific system prompts."""

    @abstractmethod
    def build(self, *, tools: AllowedTools) -> str:
        """Construct the stable system instruction string scoped to the allowed tools."""

        raise NotImplementedError

    @abstractmethod
    def build_user_context(
        self,
        history: Optional[Any] = None,
        memory: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """Construct the dynamic user context string."""

        raise NotImplementedError
