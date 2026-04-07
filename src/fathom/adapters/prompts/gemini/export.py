from __future__ import annotations

from typing import Any, Dict, Sequence

from fathom.core.prompts.export import (
    EXPORT_SYSTEM_INSTRUCTION,
    ExportPromptBuilder,
    build_export_user_prompt,
)


class GeminiExportPromptBuilder(ExportPromptBuilder):
    """
    Provider shim for script export prompting.

    Export rules (grammar, ordering, conditional handling, validation
    distribution, final-state phrasing) are provider-neutral product
    policy and live in ``fathom.core.prompts.export``. This class only
    exists so the PromptFactory can hand back an ``ExportPromptBuilder``
    for the Gemini provider key — adding new providers reuses the same
    policy.
    """

    def build_system_instruction(self) -> str:
        return EXPORT_SYSTEM_INSTRUCTION

    def build_user_prompt(
        self,
        *,
        intent: str,
        goal_state: str,
        package_name: str,
        trace_payload: Sequence[Dict[str, Any]],
        action_catalog_lines: Sequence[str],
    ) -> str:
        return build_export_user_prompt(
            intent=intent,
            goal_state=goal_state,
            package_name=package_name,
            trace_payload=trace_payload,
            action_catalog_lines=action_catalog_lines,
        )
