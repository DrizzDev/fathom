from __future__ import annotations

from typing import Any, Dict, List, Optional

from fathom.prompts.base import PromptBuilder
from fathom.prompts.templates import COMMON_RULES, TOOL_GUIDANCE


class GeminiPromptBuilder(PromptBuilder):
    """
    Constructs optimized prompts for Gemini model series.
    Supports native tool calling and visual grounding.
    """

    def build(
        self,
        intent: str,
        history: Optional[Any] = None,
        hints: Optional[Dict[str, Any]] = None,
        memory: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Builds the complete system instruction for Gemini.
        """

        context = self.__format_history(history)

        if memory:
            ledger = "\n=== PERSISTENT MEMORY ===\n"
            for key, value in memory.items():
                ledger += f"- {key}: {value}\n"

            ledger += "=== END MEMORY ===\n"
            context = ledger + context

        base = self.__get_base(intent, hints)
        rules = COMMON_RULES

        return (
            f"{base}\n"
            f"{TOOL_GUIDANCE}\n"
            f"{rules}\n"
            f"{context}\n"
            f"User intent: {intent}\n\n"
            f"Use the execute_ui function to respond."
        )

    def __format_history(self, history: Optional[Any], limit: int = 10) -> str:
        """
        Formats interaction history into a readable block.
        """

        if isinstance(history, str):
            return history

        if not history or not isinstance(history, list):
            return ""

        recent = history[-limit:]
        formatted = "\n\n=== INTERACTION HISTORY ===\n"
        formatted += "Already interacted elements (avoid these):\n\n"

        elements = []

        for index, item in enumerate(recent, 1):
            action = item.get("action_type", "tap")
            text = item.get("element_text") or item.get("rationale", "unknown")

            formatted += f"{index}. {text} ({action})\n"
            if text and text != "unknown":
                elements.append(text)

        formatted += "\nMatch by text/labels AND visual characteristics. Select DIFFERENT element of SAME TYPE.\n"
        if elements:
            formatted += f"Avoid: {', '.join([f'\"{t}\"' for t in elements[:5]])}\n"

        formatted += "=== END ===\n"
        return formatted

    def __get_base(self, intent: str, hints: Optional[Dict[str, Any]]) -> str:
        """
        Constructs the foundation of the prompt with dynamic hints.
        """

        base = "You are a Mobile UI expert. Map user intent to screen actions using tools.\n"
        
        if hints:
            if hints.get("use_xml"):
                base += "Use NUMERIC LABELS from the XML hierarchy for grounding.\n"
            
        # Add dynamic instructions based on intent
        dynamic = []

        if "every" in intent.lower() or "all" in intent.lower():
            dynamic.append("- REPEAT UNTIL ALL: Identify ALL matching elements. Select NEXT untapped element.")
        
        if any(kw in intent.lower() for kw in ["type", "enter", "input"]):
            dynamic.append("- SEQUENTIAL: For typing, tap field to focus, then type.")

        if dynamic:
            base += "\nCONTEXTUAL RULES:\n" + "\n".join(dynamic) + "\n"

        return base