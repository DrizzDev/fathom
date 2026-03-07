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
        action_catalog_lines: Sequence[str],
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
            "10) Executable actions must be copied exactly from allowed step-derived action lines. Do not invent or paraphrase action text like 'Clear all items'.\n"
            "11) When a package is provided, the first executable line MUST be exactly OPEN_APP <package>.\n"
            "12) If user intent includes conditional language (e.g., 'if', 'when', 'if cart is not empty'), you MUST represent that branch using IF block syntax.\n"
            "13) If the intent requests multiple checks (validate/verify/assert/check/confirm), you MUST distribute them across action-anchored validations in action_validations{}. NEVER collapse multiple checks into a single final validation line. Example: intent='Validate X, Validate Y, Validate Z' → action_validations must map at least 2 checks to different action IDs, preserving 1 for final_validation.\n"
            "14) Return structured tool args: conditional_blocks[].action_ids, remaining_action_ids[], action_validations{}, final_validation.\n"
            "15) action_validations keys must be action IDs from the catalog; values must start with 'Validate'.\n"
            "16) Use only action IDs from the provided action catalog; do not emit raw executable action text.\n"
            "17) CRITICAL: If the execution trace has intermediate points where user validations should occur (between actions), anchor each validation to the nearest preceding action ID in action_validations."
        )

    def build_user_prompt(
        self,
        *,
        intent: str,
        goal_state: str,
        package_name: str,
        baseline_script: str,
        trace_payload: Sequence[Dict[str, Any]],
        action_catalog_lines: Sequence[str],
    ) -> str:
        """
        Build user payload for export generation.
        """

        catalog_formatted = (
            "\n".join(f"- {line}" for line in list(action_catalog_lines)) or "- (none)"
        )

        return (
            f"User intent: {intent or goal_state or 'N/A'}\n"  # nosec B608
            f"Goal state: {goal_state or intent or 'N/A'}\n"
            f"Package: {package_name or 'N/A'}\n\n"
            "Opening-line constraint:\n"
            "- If package is provided, first executable line must be exactly: "
            f"OPEN_APP {package_name or '<package>'}\n\n"
            "Action catalog (STRICT, use IDs):\n"
            f"{catalog_formatted}\n\n"
            "Action constraints:\n"
            "- Select action IDs only from the action catalog.\n"
            "- Do not rewrite, summarize, or paraphrase executable actions.\n"
            "- Preserve chronological order from the trace when grouping into IF blocks.\n\n"
            "Conditional-block constraint:\n"
            "- If the intent has an 'if/when' condition, include at least one IF block and place condition-scoped steps inside it.\n\n"
            "Tool output format constraint:\n"
            "- Return structured tool args with keys: conditional_blocks, remaining_action_ids, action_validations, final_validation.\n"
            "- In conditional_blocks, use action_ids (not action text).\n"
            "- In action_validations, map 1+ action IDs to intermediate validation lines (must start with 'Validate'). CRITICAL: populate this field whenever intent has multiple validation requirements (e.g., 'Validate X', 'Validate Y', 'Verify Z'). Map each to a different action ID.\n"
            "- Do not return a free-form script string.\n\n"
            "Final-goal requirement:\n"
            "- End the script with one validation line that captures the exact user goal in natural language.\n"
            "- Avoid generic endings like 'Validate Goal State is visible'.\n\n"
            f"Baseline script:\n{baseline_script}\n"
            f"Execution trace JSON:\n{json.dumps(list(trace_payload), ensure_ascii=True, indent=2)}\n\n"
            "Generate the final script now."
        )
