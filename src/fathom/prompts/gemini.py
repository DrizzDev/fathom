from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fathom.prompts.base import PromptBuilder
from fathom.prompts.templates import (
    ACTION_RULES,
    COMMON_RULES,
    CONFIDENCE_RULES,
    COORD_RULES,
    PRECISION_RULES,
    TOOL_GUIDANCE,
    UI_RULES,
)


class GeminiPromptBuilder(PromptBuilder):
    """
    Structured Gemini prompt builder.
    """

    def build(
        self,
        intent: str,
        history: Optional[Any] = None,
        hints: Optional[Dict[str, Any]] = None,
        memory: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Build a high-signal system prompt for tool-based UI execution.
        """

        interaction_context = self.__format_interaction_history(history=history)
        contextual_rules = self.__get_contextual_rules(intent=intent, hints=hints)
        conditional_notes = self.__get_conditional_notes(intent=intent, hints=hints)

        parts = [
            self.__get_persona(),
            TOOL_GUIDANCE,
            COMMON_RULES,
            contextual_rules,
            conditional_notes,
            self.__get_ledger_segment(memory=memory),
            interaction_context,
            (
                "OUTPUT REQUIREMENTS:\n"
                f"- {COORD_RULES}\n"
                f"- {CONFIDENCE_RULES}\n"
                "- Return tool call(s) only, with schema-valid fields.\n"
                f"\nGOAL: {intent}\nExecute next best step via tool."
            ),
        ]
        return "\n\n".join([part for part in parts if part.strip()])

    def __get_persona(self) -> str:
        """
        Core identity.
        """

        return (
            "You are a Mobile UI expert agent. "
            "Ground all interactions using normalized coordinates (0-1000)."
        )

    def __get_contextual_rules(self, intent: str, hints: Optional[Dict[str, Any]]) -> str:
        """
        High-priority contextual rules.
        """

        rules: List[str] = []

        if hints and hints.get("use_xml"):
            rules.append("- XML Grounding enabled.")

        if any(word in intent.lower() for word in ["every", "all"]):
            rules.append("- LOOP: Iterate untried matching elements. Avoid repeats.")

        if any(word in intent.lower() for word in ["type", "enter", "input"]):
            rules.append(
                "- CRITICAL SEQ: Use 'tap' to gain focus on the input field, followed by 'type'."
            )

        if hints and hints.get("needs_navigation"):
            target = str(hints.get("target_screen", "target screen"))
            rules.append(
                f"- NAVIGATION: Move toward '{target}' before actioning intent-specific UI."
            )

        return "RULES:\n" + "\n".join(rules) if rules else ""

    def __get_conditional_notes(self, intent: str, hints: Optional[Dict[str, Any]]) -> str:
        """
        Add concise behavior notes derived from intent/hints.
        """

        notes: List[str] = []
        intent_lower = intent.lower()

        if hints and hints.get("typing_text"):
            text = str(hints["typing_text"])
            notes.append(f"- TYPING INTENT: Use literal text_to_type='{text}'.")

        if "search" in intent_lower and any(k in intent_lower for k in ["tap", "select", "click"]):
            notes.append("- SEARCH FLOW: If suggestions are visible, type then tap suggestion.")

        if not hints or not hints.get("requires_repeat_all"):
            notes.append(
                "- COMPLETE CHECK: If goal appears fully achieved, verify goal explicitly."
            )

        if not notes:
            return ""

        return "NOTES:\n" + "\n".join(notes)

    def __get_ledger_segment(self, memory: Optional[Dict[str, str]]) -> str:
        """
        High-density ledger memory.
        """

        if not memory:
            return ""

        # Format: [KEY:VAL]
        items = [f"{key}:{value}" for key, value in memory.items()]
        return f"LEDGER: [{', '.join(items)}]"

    def __format_interaction_history(self, history: Optional[Any]) -> str:
        """
        Compact interaction history to reduce repeated actions.
        """

        if isinstance(history, str):
            return history

        if not history or not isinstance(history, list):
            return ""

        recent = history[-8:]
        lines: List[str] = []
        avoided: List[str] = []

        for index, item in enumerate(recent, 1):
            action = str(item.get("action_type", "tap"))
            target = str(item.get("element_text") or item.get("target") or "unknown")
            success = bool(item.get("success", True))
            lines.append(f"{index}. {target} ({action}) -> {'ok' if success else 'failed'}")
            if target != "unknown":
                avoided.append(target)

        block = "INTERACTION HISTORY:\n" + "\n".join(lines)
        if avoided:
            block += f"\nAvoid repeats when possible: {', '.join(avoided[:6])}"

        return block

    def build_next_step_prompt(
        self, user_intent: str, interaction_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build focused prompt for stuck/next-step recovery.
        """

        interaction_context = self.__format_interaction_history(history=interaction_history)

        return (
            "Task: Intent appears stuck. Propose ONE best action to progress.\n"
            f"User intent: {user_intent}\n"
            f"{interaction_context}\n\n"
            "RULES:\n"
            f"- {COORD_RULES}\n"
            f"- {CONFIDENCE_RULES}\n"
            f"- {UI_RULES['dropdown']}\n"
            f"- {UI_RULES['goal_lock']}\n"
            f"- {PRECISION_RULES['input']}\n"
            f"- {PRECISION_RULES['text']}\n"
            f"- {ACTION_RULES['scroll']}\n"
            f"- {ACTION_RULES['wait']}\n"
            "Respond using tool schema only."
        )

    def build_action_verification_prompt(self, intent: str) -> str:
        """
        Build strict action+screen alignment verification prompt.
        """

        return (
            "Task: Verify action alignment and screen context.\n"
            f"User Intent: {intent}\n"
            "Check: (1) screen is on-intent, (2) action marker targets correct UI.\n"
            "If blocker overlays appear (permissions, cookie prompts, login wall, updates), mark misaligned.\n"
            "Use validate_state to report evidence."
        )

    def build_intent_generation_prompt(
        self, image_size: Tuple[int, int], app_context: Optional[str] = None
    ) -> str:
        """
        Build prompt for exploration-style screen intent generation.
        """

        width, height = image_size
        context = f"App context: {app_context}\n" if app_context else ""

        return (
            "Task: Generate intents for all interactive elements on this screen.\n"
            f"{context}Screen size: {width}x{height}\n"
            "Output each intent with normalized bbox and predicted next state."
        )
