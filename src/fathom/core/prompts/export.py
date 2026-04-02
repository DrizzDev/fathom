from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence


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
