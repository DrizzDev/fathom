"""
Structured Gemini prompt builder using GCC-inspired context tiers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fathom.core.prompts.base import PromptBuilder
from fathom.core.prompts.templates import (
    COMMON_RULES,
    CONFIDENCE_RULES,
    COORD_RULES,
    TOOL_GUIDANCE,
)


class GeminiPromptBuilder(PromptBuilder):
    """
    Structured Gemini prompt builder that formats hierarchical context.
    """

    def build(
        self,
        intent: str,
        hints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build stable system prompt for tool-based UI execution.
        """

        contextual_rules = self.__get_contextual_rules(intent=intent, hints=hints)
        conditional_notes = self.__get_conditional_notes(intent=intent, hints=hints)

        parts = [
            self.__get_persona(),
            TOOL_GUIDANCE,
            COMMON_RULES,
            contextual_rules,
            conditional_notes,
            (
                "OUTPUT REQUIREMENTS:\n"
                f"- {COORD_RULES}\n"
                f"- {CONFIDENCE_RULES}\n"
                "- Return tool call(s) only, with schema-valid fields.\n"
                f"\nGOAL: {intent}\nExecute next best step via tool."
            ),
        ]
        return "\n\n".join([part for part in parts if part.strip()])

    def build_user_context(
        self,
        context: Dict[str, Any],
        memory: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Build dynamic user context string from GCC-inspired tiers.
        """

        parts = []

        # 1. Priority Guidance (HITL)
        guidance = context.get("guidance", [])
        if guidance:
            instructions = [f"- {item}" for item in guidance]
            parts.append("USER INSTRUCTIONS (PRIORITY):\n" + "\n".join(instructions))

        # 2. Roadmap & Milestones
        milestones = context.get("milestones", [])
        if milestones:
            parts.append("COMPLETED MILESTONES:\n" + "\n".join(f"- {m}" for m in milestones))

        # 3. Memory Ledger
        ledger = self.__get_ledger_segment(memory=memory)
        if ledger:
            parts.append(ledger)

        # 4. Execution Trace (Interaction History)
        trace = context.get("trace", [])
        interaction_context = self.__format_trace(trace=trace)
        if interaction_context:
            parts.append(interaction_context)

        return "\n\n".join(parts)

    def __get_persona(self) -> str:
        """Core identity."""
        return (
            "You are a Mobile UI expert agent. "
            "Ground all interactions using normalized coordinates (0-1000)."
        )

    def __get_contextual_rules(self, intent: str, hints: Optional[Dict[str, Any]]) -> str:
        """High-priority contextual rules."""
        rules: List[str] = []

        if hints and hints.get("use_xml"):
            rules.append("- XML Grounding enabled.")

        if any(word in intent.lower() for word in ["every", "all"]):
            rules.append("- LOOP: Iterate untried matching elements. Avoid repeats.")

        if any(word in intent.lower() for word in ["type", "enter", "input"]):
            rules.append(
                "- CRITICAL SEQ: Use 'tap' to gain focus on the input field, followed by 'type'."
            )

        return "RULES:\n" + "\n".join(rules) if rules else ""

    def __get_conditional_notes(self, intent: str, hints: Optional[Dict[str, Any]]) -> str:
        """Add concise behavior notes."""
        notes: List[str] = []
        intent_lower = intent.lower()

        if hints and hints.get("typing_text"):
            text = str(hints["typing_text"])
            notes.append(f"- TYPING INTENT: Use literal text_to_type='{text}'.")

        if "search" in intent_lower and any(k in intent_lower for k in ["tap", "select", "click"]):
            notes.append("- SEARCH FLOW: If suggestions are visible, type then tap suggestion.")

        notes.append("- COMPLETE CHECK: If goal appears fully achieved, verify goal explicitly.")

        return "NOTES:\n" + "\n".join(notes)

    def __get_ledger_segment(self, memory: Optional[Dict[str, str]]) -> str:
        """High-density ledger memory."""
        if not memory:
            return ""
        items = [f"{key}:{value}" for key, value in memory.items()]
        return f"LEDGER: [{', '.join(items)}]"

    def __format_trace(self, trace: List[Dict[str, Any]]) -> str:
        """Formats the GCC trace into a readable interaction history."""
        if not trace:
            return ""

        recent = trace[-8:]
        lines = []
        avoided = []

        for index, entry in enumerate(recent, 1):
            observation = entry.get("observation", "Unknown screen")
            action = entry.get("action", {})

            # Action might be dict or object
            if isinstance(action, dict):
                desc = action.get("target", "unknown")
                type_ = action.get("action_type", "tap")
            else:
                desc = getattr(action, "target", "unknown")
                type_ = getattr(action, "action_type", "tap")
                if hasattr(type_, "value"):
                    type_ = type_.value

            lines.append(f"{index}. {observation} -> {type_.upper()}:{desc}")
            if desc != "unknown":
                avoided.append(desc)

        block = "INTERACTION HISTORY:\n" + "\n".join(lines)
        if avoided:
            block += f"\nAvoid repeats when possible: {', '.join(avoided[:6])}"

        return block
