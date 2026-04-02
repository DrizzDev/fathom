"""Port definitions for prompt builders."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence


class PromptBuilder(ABC):
    """
    Abstract base class for building model-specific system prompts.
    """

    @abstractmethod
    def build(self) -> str:
        """
        Constructs the stable system instruction string (for caching).
        """

        raise NotImplementedError

    @abstractmethod
    def build_user_context(
        self,
        history: Optional[Any] = None,
        memory: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Constructs the dynamic user context string.
        """

        raise NotImplementedError


class ExportPromptBuilder(ABC):
    """
    Abstract builder for script-export prompting.
    """

    @abstractmethod
    def build_system_instruction(self) -> str:
        """
        Build stable system instruction for export generation.
        """

        raise NotImplementedError

    @abstractmethod
    def build_user_prompt(
        self,
        *,
        intent: str,
        goal_state: str,
        package_name: str,
        trace_payload: Sequence[Dict[str, Any]],
        action_catalog_lines: Sequence[str],
    ) -> str:
        """
        Build dynamic user prompt for export generation.
        """

        raise NotImplementedError


class DecompositionPromptBuilder(ABC):
    """
    Abstract builder for intent decomposition prompting.
    """

    @abstractmethod
    def build_system_instruction(self) -> str:
        """
        Build stable system instruction for decomposition generation.
        """

        raise NotImplementedError

    @abstractmethod
    def build_user_prompt(self, *, intent: str) -> str:
        """
        Build dynamic user prompt for decomposing an intent.
        """

        raise NotImplementedError
