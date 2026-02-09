from __future__ import annotations

from typing import Any, Dict, Optional

from fathom.prompts.base import PromptBuilder
from fathom.prompts.templates import COMMON_RULES, TOOL_GUIDANCE


class GeminiPromptBuilder(PromptBuilder):
    """
    Structured builder for Gemini system instructions.
    """

    def build(
        self,
        intent: str,
        history: Optional[Any] = None,
        hints: Optional[Dict[str, Any]] = None,
        memory: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Assembles the final prompt using a high-density format.
        """

        parts = [
            self.__get_persona(),
            self.__get_contextual_rules(intent=intent, hints=hints),
            TOOL_GUIDANCE,
            COMMON_RULES,
            self.__get_ledger_segment(memory=memory),
            self.__get_history_segment(history=history),
            f"GOAL: {intent}\n\nExecute next step via tool.",
        ]

        return "\n".join([part for part in parts if part])

    def __get_persona(self) -> str:
        """
        Core identity.
        """

        return "You are a Mobile UI Agent. Grounding via coordinates (0-1000)."

    def __get_contextual_rules(self, intent: str, hints: Optional[Dict[str, Any]]) -> str:
        """
        Intent-specific guidance.
        """

        rules = []

        if hints and hints.get("use_xml"):
            rules.append("- XML Grounding enabled.")

        if any(word in intent.lower() for word in ["every", "all"]):
            rules.append("- LOOP: Iterate untried elements.")

        if any(word in intent.lower() for word in ["type", "enter", "input"]):
            rules.append("- SEQ: Tap to focus, then type.")

        return "\nRULES:\n" + "\n".join(rules) if rules else ""

    def __get_ledger_segment(self, memory: Optional[Dict[str, str]]) -> str:
        """
        High-density ledger memory.
        """

        if not memory:
            return ""

        # Format: [KEY:VAL]
        items = [f"{key}:{value}" for key, value in memory.items()]
        return f"\nLEDGER: [{', '.join(items)}]"

    def __get_history_segment(self, history: Optional[Any]) -> str:
        """
        Symbolic history to minimize tokens.
        Format: [ACTION:TARGET:RESULT]
        """

        if isinstance(history, str):
            return history

        if not history or not isinstance(history, list):
            return ""

        # Last 5 steps should be sufficient for loop detection and save tokens
        steps = []
        recent = history[-5:]

        for item in recent:
            action = item.get("action_type", "tap").upper()[:3]
            target = item.get("element_text") or "UI"
            result = "✓" if item.get("success", True) else "✗"
            steps.append(f"{action}:{target}:{result}")

        return f"\nTRACE: [{' | '.join(steps)}]"
