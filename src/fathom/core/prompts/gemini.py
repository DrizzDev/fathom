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
        history: Optional[Any] = None,
        memory: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Build dynamic user context string from GCC-inspired tiers.
        """

        # Map history to context for internal consistency with GCC terminology
        context = history if isinstance(history, dict) else {}
        tracking_note: Optional[str] = kwargs.get("tracking_note")

        parts = []

        # 0. Interaction Cadence (Deterministic Repetition Tracking)
        if tracking_note:
            parts.append(f"<CADENCE_NOTE>\n{tracking_note}\n</CADENCE_NOTE>")

        # 1. Memory Ledger (Factual Memory)
        ledger = self.__get_ledger_segment(memory=memory)
        if ledger:
            parts.append(f"<MEMORY_LEDGER>\n{ledger}\n</MEMORY_LEDGER>")

        # 2. Roadmap & Milestones (Tier 2 Context)
        milestones = context.get("milestones", [])
        if milestones:
            parts.append(
                "<MILESTONES>\n" + "\n".join(f"- {m}" for m in milestones) + "\n</MILESTONES>"
            )

        # 3. Execution Trace (Tier 3 Context - The Hot Suffix)
        trace = context.get("trace", [])
        interaction_context = self.__format_trace(trace=trace)
        if interaction_context:
            parts.append(f"<CURRENT_TRACE>\n{interaction_context}\n</CURRENT_TRACE>")

        # 4. Priority Guidance (HITL) - The "System Override"
        # Placed LAST to ensure maximum recency bias and adherence
        guidance = context.get("guidance", [])
        if guidance:
            instructions = [f"- {item}" for item in guidance]
            parts.append(
                "<SYSTEM_OVERRIDE>\n"
                "  <INSTRUCTION>\n" + "\n".join(f"    {inst}" for inst in instructions) + "\n"
                "  </INSTRUCTION>\n"
                "  <WARNING>\n"
                "    This is a meta-instruction for the agent's behavior.\n"
                "    Do NOT treat this as content to be typed or searched.\n"
                "    You MUST adjust your plan to comply with this override.\n"
                "  </WARNING>\n"
                "</SYSTEM_OVERRIDE>"
            )

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
        return f"[{', '.join(items)}]"

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
