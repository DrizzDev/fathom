from __future__ import annotations

from typing import Any, Dict, List, Optional

from fathom.prompts.base import PromptBuilder
from fathom.prompts.modes import PromptMode
from fathom.prompts.templates import (
    COORD_RULES,
    EXPLORATION_ACTION_PALETTE,
    EXPLORATION_ELEMENT_CATEGORIES,
    EXPLORATION_EXHAUSTION_RULES,
    EXPLORATION_FOCUS_DIRECTIVE,
    EXPLORATION_LIST_SAMPLING,
    EXPLORATION_MENTAL_MODEL,
    EXPLORATION_OVERLAY_RULES,
    EXPLORATION_PERSONA,
    EXPLORATION_PRIORITY,
    EXPLORATION_REGION_GUIDE,
    EXPLORATION_RESPONSE_DIRECTIVE,
    EXPLORATION_SCAN_STRATEGY,
    EXPLORATION_SCREEN_DESCRIPTION_GUIDE,
)


class GeminiPromptBuilder(PromptBuilder):
    """
    Structured Gemini prompt builder for exploration workflows.
    """

    def build(
        self,
        mode: str = "exploration",
        intent: str = "",
        hints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build stable system prompt for tool-based UI exploration.
        EXCLUDES dynamic history/memory to enable Context Caching.
        """

        if mode == PromptMode.EXPLORATION.value:
            return self.__build_exploration_prompt(intent=intent)

        if mode == PromptMode.SCREEN_TRANSLATION.value:
            return self.__build_screen_translation_prompt()

        # Fallback to exploration prompt for unknown modes
        return self.__build_exploration_prompt(intent=intent)

    def build_task_instructions(
        self,
        intent: str,
        hints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build dynamic task instructions for the User Message.
        Only includes per-step dynamic hints (stuck detection, delta signals).
        The stable GOAL is now part of the cached system instruction.
        """

        parts: List[str] = []

        # Stuck detection hints (used by exploration orchestrator)
        if hints and hints.get("is_stuck"):
            last_action = hints.get("last_action", "")
            if last_action in ["scroll", "swipe_left", "swipe_right", "swipe_up", "swipe_down"]:
                parts.append(
                    "RULES:\n"
                    "- CRITICAL: SCREEN UNCHANGED AFTER SCROLL. "
                    "Set 'content_exhausted' to true."
                )
            elif last_action in ["tap", "type", "enter"]:
                parts.append(
                    "RULES:\n"
                    "- CRITICAL: LOOP DETECTED (action had no effect). "
                    "Try a different element or go BACK."
                )
            else:
                parts.append(
                    "RULES:\n"
                    "- CRITICAL: LOOP DETECTED. You are stuck. "
                    "Use 'back', 'scroll', or try a different element."
                )

        if hints and int(hints.get("delta_low_streak", 0)) >= 3:
            parts.append(
                "- DELTA SIGNAL: recent actions produced little/no screen change. "
                "Prefer strategy shift (back, different target, or content_exhausted if all tried)."
            )

        return "\n\n".join([part for part in parts if part.strip()])

    def build_user_context(
        self,
        history: Optional[Any] = None,
        memory: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Build dynamic user context string.
        """

        ledger = self.__get_ledger_segment(memory=memory)
        interaction_context = self.__format_interaction_history(history=history)

        parts = []

        if ledger:
            parts.append(ledger)

        if interaction_context:
            parts.append(interaction_context)

        return "\n".join(parts)

    def __get_ledger_segment(self, memory: Optional[Dict[str, str]]) -> str:
        """
        High-density ledger memory.
        """

        if not memory:
            return ""

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

        for index, item in enumerate(recent, 1):
            action = str(item.get("action_type", "tap"))
            target = str(item.get("target") or "unknown")
            success = bool(item.get("success", True))
            lines.append(f"{index}. {target} ({action}) -> {'ok' if success else 'fail'}")

        return "ACTIONS:\n" + "\n".join(lines)

    def __build_exploration_prompt(self, intent: str = "") -> str:
        """
        Exploration Mode: systematic app mapping (depth-first).

        Prompt structure follows the Six Elements framework:
        - Role/Persona (EXPLORATION_PERSONA)
        - Task (EXPLORATION_MENTAL_MODEL)
        - Context (SCAN_STRATEGY, ELEMENT_CATEGORIES)
        - Format (explore_ui tool contract — via function_declarations)
        - Examples (inline in SCREEN_DESCRIPTION_GUIDE)
        - Constraints (PRIORITY, OVERLAY, EXHAUSTION rules + COORD_RULES)

        Section order uses anchoring: critical rules at beginning and end.
        The GOAL is embedded here so it becomes part of the cached system
        instruction — it is constant for the entire exploration session.
        """
        parts = [
            EXPLORATION_PERSONA,
        ]

        if intent:
            parts.append(f"GOAL: {intent}")

        parts.extend(
            [
                EXPLORATION_MENTAL_MODEL,
                EXPLORATION_SCAN_STRATEGY,
                EXPLORATION_ELEMENT_CATEGORIES,
                EXPLORATION_PRIORITY,
                EXPLORATION_FOCUS_DIRECTIVE,
                EXPLORATION_ACTION_PALETTE,
                EXPLORATION_LIST_SAMPLING,
                EXPLORATION_REGION_GUIDE,
                EXPLORATION_SCREEN_DESCRIPTION_GUIDE,
                EXPLORATION_OVERLAY_RULES,
                EXPLORATION_EXHAUSTION_RULES,
                COORD_RULES,
                EXPLORATION_RESPONSE_DIRECTIVE,
            ]
        )
        return "\n\n".join(parts)

    def __build_screen_translation_prompt(self) -> str:
        """
        Screen Translation Mode: a rich functional description of an activity
        screen — what is on it, what each element does, and what a user can
        achieve here.

        Uses the ``describe_screen`` tool to return structured sections.
        """
        return (
            "You are a mobile app analyst. Given a screenshot, describe what is on the "
            "screen so a reader understands it without seeing it: each element, what it "
            "does, and what a user can achieve here.\n\n"
            "CRITICAL: Use STABLE labels, not volatile data.\n"
            "- Capture meaningful labels: button/tab/section names, what a card represents.\n"
            "- Do NOT include volatile runtime content: specific prices, individual item "
            "names ('99 Slice Pizza', '₹717'). Describe the element TYPE and its function.\n\n"
            "You MUST call the describe_screen tool with:\n\n"
            "- activity_name: The Android activity this screen belongs to.\n"
            "- screen_purpose: 1-2 sentences on what this screen is for and the primary "
            "tasks available here.\n"
            "- elements: Every element, one per line, grouped by region — what it is, its "
            "stable label, and what it does or where it leads.\n"
            "- achievable_actions: The concrete things a user can accomplish on this screen, "
            "one per line.\n\n"
            "Be exhaustive on elements — every icon, tab, field, card type, and button."
        )
