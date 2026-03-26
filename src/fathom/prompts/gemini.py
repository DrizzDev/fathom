from __future__ import annotations

from typing import Any, Dict, List, Optional

from fathom.prompts.base import PromptBuilder
from fathom.prompts.modes import PromptMode
from fathom.prompts.templates import (
    COORD_RULES,
    EXPLORATION_ELEMENT_CATEGORIES,
    EXPLORATION_EXHAUSTION_RULES,
    EXPLORATION_MENTAL_MODEL,
    EXPLORATION_OVERLAY_RULES,
    EXPLORATION_PERSONA,
    EXPLORATION_PRIORITY,
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
            return self.__build_exploration_prompt()

        if mode == PromptMode.SCREEN_TRANSLATION.value:
            return self.__build_screen_translation_prompt()

        # Fallback to exploration prompt for unknown modes
        return self.__build_exploration_prompt()

    def build_task_instructions(
        self,
        intent: str,
        hints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build dynamic task instructions for the User Message.
        For exploration, returns the goal + stuck-detection hints.
        """

        parts = [f"GOAL: {intent}"]

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
            target = str(item.get("element_text") or item.get("target") or "unknown")
            success = bool(item.get("success", True))
            lines.append(f"{index}. {target} ({action}) -> {'ok' if success else 'fail'}")

        return "ACTIONS:\n" + "\n".join(lines)

    def __build_exploration_prompt(self) -> str:
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
        """
        parts = [
            EXPLORATION_PERSONA,
            EXPLORATION_MENTAL_MODEL,
            EXPLORATION_SCAN_STRATEGY,
            EXPLORATION_ELEMENT_CATEGORIES,
            EXPLORATION_PRIORITY,
            EXPLORATION_SCREEN_DESCRIPTION_GUIDE,
            EXPLORATION_OVERLAY_RULES,
            EXPLORATION_EXHAUSTION_RULES,
            COORD_RULES,
            EXPLORATION_RESPONSE_DIRECTIVE,
        ]
        return "\n\n".join(parts)

    def __build_screen_translation_prompt(self) -> str:
        """
        Screen Translation Mode: thorough rich-text description of all
        visible designs, features, and content on a mobile app screen.

        Uses the ``describe_screen`` tool to return structured sections.
        """
        return (
            "You are a meticulous UI design analyst. Given a screenshot of a mobile app screen, "
            "produce a thorough textual translation of EVERYTHING visible.\n\n"
            "You MUST call the describe_screen tool with detailed content for each section:\n\n"
            "- layout_and_structure: Describe the overall page layout — regions, columns, cards, "
            "spacing, visual hierarchy, header/body/footer arrangement, and how content is organized.\n"
            "- navigation: Describe all navigation elements — top/bottom bars, tabs, menus, "
            "breadcrumbs, back buttons, hamburger icons, sidebars, and their labels.\n"
            "- content: Describe ALL visible text, headings, subheadings, body text, labels, "
            "images, icons, media, cards, lists, badges, tags, and data displayed.\n"
            "- interactive_elements: Describe every interactive control — buttons (with labels), "
            "text inputs, search bars, toggles, switches, checkboxes, radio buttons, dropdowns, "
            "sliders, links, FABs, and their current states (enabled/disabled/selected).\n"
            "- visual_design: Describe colors (background, text, accents), typography (font sizes, "
            "weights, styles), iconography, branding elements, shadows, borders, rounded corners, "
            "and overall design language/theme.\n"
            "- summary: A 2-4 sentence flowing prose paragraph summarizing the screen's purpose, "
            "the primary user task it supports, and the overall user experience.\n\n"
            "Be comprehensive — capture every visible element, label, icon, and design detail. "
            "Do not omit anything visible on screen."
        )
