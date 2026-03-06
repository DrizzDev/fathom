from __future__ import annotations

import json
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
        baseline_script: str,
        trace_payload: Sequence[Dict[str, Any]],
    ) -> str:
        """
        Build dynamic user prompt for export generation.
        """

        raise NotImplementedError


class GeminiExportPromptBuilder(ExportPromptBuilder):
    """
    Gemini-focused prompt builder for script export composition.
    """

    def build_system_instruction(self) -> str:
        """
        Build stable system prompt for deterministic export scripts.
        """

        return (
            "You convert mobile UI execution traces into deterministic automation scripts.\n"
            "Output only script lines; no markdown, no commentary.\n"
            "Grammar:\n"
            "- OPEN_APP <package>\n"
            "- IF <condition> {\n"
            "- }\n"
            "- Action/validation lines such as Tap on..., Type '...' into..., Wait..., Scroll..., Swipe..., Validate..., Press...\n"
            "Rules:\n"
            "1) Preserve chronological order and user intent.\n"
            "2) Group consecutive condition-bound actions into the same IF block when they share the same intent guard.\n"
            "3) For intents like 'if cart is not empty clear cart then add item', include ALL cart-clearing actions inside IF cart-not-empty block, and keep remaining actions outside that block.\n"
            "4) Never invent new actions, screens, or targets. Use only the provided trace data.\n"
            "5) For repeatability, replace product-specific dynamic targets with generic references (e.g., 'the first search result') unless the user intent explicitly names that product.\n"
            "6) Do not include store-brand names in action/condition targets (e.g., Walmart, Costco); use generic functional targets instead.\n"
            "7) Keep OPEN_APP and final validations when supported by trace/baseline.\n"
            "8) The final non-structural line MUST be a goal validation statement that semantically matches the user intent (not a generic placeholder).\n"
            "9) Emit script via tool using schema-compliant plain text only (no markdown fences).\n"
            "10) Ensure balanced braces and executable plain-text script format."
        )

    def build_user_prompt(
        self,
        *,
        intent: str,
        goal_state: str,
        package_name: str,
        baseline_script: str,
        trace_payload: Sequence[Dict[str, Any]],
    ) -> str:
        """
        Build user payload for export generation.
        """

        return (
            f"User intent: {intent or goal_state or 'N/A'}\n"
            f"Goal state: {goal_state or intent or 'N/A'}\n"
            f"Package: {package_name or 'N/A'}\n\n"
            "Final-goal requirement:\n"
            "- End the script with one validation line that captures the exact user goal in natural language.\n"
            "- Avoid generic endings like 'Validate Goal State is visible'.\n\n"
            f"Baseline script:\n{baseline_script}\n"
            f"Execution trace JSON:\n{json.dumps(list(trace_payload), ensure_ascii=True, indent=2)}\n\n"
            "Generate the final script now."
        )
