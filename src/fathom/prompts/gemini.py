from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fathom.prompts.base import PromptBuilder
from fathom.prompts.modes import PromptMode
from fathom.prompts.templates import (
    ACTION_RULES,
    COMMON_RULES,
    CONFIDENCE_RULES,
    COORD_RULES,
    PRECISION_RULES,
    RESPONSE_DIRECTIVE,
    TOOL_GUIDANCE,
    UI_RULES,
)


class GeminiPromptBuilder(PromptBuilder):
    """
    Structured Gemini prompt builder.
    """

    def build(
        self,
        mode: str = "default",
        intent: str = "",
        hints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build stable system prompt for tool-based UI execution.
        EXCLUDES dynamic history/memory to enable Context Caching.
        Ignores intent/hints to ensure static cache key.
        Uses 'mode' to select specialized prompt.
        """

        if mode == PromptMode.DISCOVERY.value:
            return self.__build_discovery_prompt()

        if mode == PromptMode.VERIFICATION.value:
            return self.__build_verification_prompt()

        if mode == PromptMode.EXPLORATION.value:
            return self.__build_exploration_prompt()

        parts = [
            self.__get_persona(),
            TOOL_GUIDANCE,
            COMMON_RULES,
            RESPONSE_DIRECTIVE,
        ]
        return "\n\n".join([part for part in parts if part.strip()])

    def build_task_instructions(
        self,
        intent: str,
        hints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build dynamic task instructions for the User Message.
        Note: Static scaffolding (tool response format, error recovery) is now in cached system prompt.
        """

        contextual_rules = self.__get_contextual_rules(intent=intent, hints=hints)
        conditional_notes = self.__get_conditional_notes(intent=intent, hints=hints)

        parts = [
            f"GOAL: {intent}",
            contextual_rules,
            conditional_notes,
        ]
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

    def __get_persona(self) -> str:
        """
        Core identity.
        """

        return "You are a Mobile UI expert agent."

    def __get_contextual_rules(self, intent: str, hints: Optional[Dict[str, Any]]) -> str:
        """
        High-priority contextual rules.
        """

        rules: List[str] = []

        if hints and hints.get("is_stuck"):
            last_action = hints.get("last_action", "")
            if last_action in ["scroll", "swipe_left", "swipe_right", "swipe_up", "swipe_down"]:
                rules.append(
                    "- CRITICAL: SCREEN UNCHANGED AFTER SCROLL. Compare the last visible item on this screen with the previous screen. If they are the same, you have reached the end. Set 'content_exhausted' to true."
                )
            elif last_action in ["tap", "type", "enter"]:
                rules.append(
                    "- CRITICAL: LOOP DETECTED (action had no effect). Target may be non-interactive or loading. Try a different strategy or go BACK."
                )
            else:
                rules.append(
                    "- CRITICAL: LOOP DETECTED. You are stuck (screen repeating). You MUST break this loop. Use 'back', 'scroll', or 'home'."
                )

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
                "- COMPLETE CHECK: If goal appears fully achieved, call complete_goal with evidence."
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

        for index, item in enumerate(recent, 1):
            action = str(item.get("action_type", "tap"))
            target = str(item.get("element_text") or item.get("target") or "unknown")
            success = bool(item.get("success", True))
            lines.append(f"{index}. {target} ({action}) -> {'ok' if success else 'fail'}")

        return "ACTIONS:\n" + "\n".join(lines)

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
            "You MUST respond with a tool call only. Use execute_ui with required fields: "
            "assistant_message (str), action (object with {action_type, rationale, is_valid}). "
            "Never output plain text."
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
            f"Screen Analysis ({width}x{height})\n"
            f"{context}"
            "Describe the screen content and suggest likely user intents."
        )

    def __build_discovery_prompt(self) -> str:
        """
        Discovery Mode: Focused on finding elements.
        """
        parts = [
            self.__get_persona(),
            TOOL_GUIDANCE,  # Keep standard guidance for now
            COMMON_RULES,  # Keep standard rules for now
            (
                "MODE: DISCOVERY (Navigation & Scanning)\n"
                "Prioritize scrolling and swiping to find elements."
            ),
            RESPONSE_DIRECTIVE,
        ]
        return "\n\n".join([part for part in parts if part.strip()])

    def __build_verification_prompt(self) -> str:
        """
        Verification Mode: Focused on assertions.
        """
        parts = [
            self.__get_persona(),
            TOOL_GUIDANCE,
            COMMON_RULES,
            (
                "MODE: VERIFICATION (Strict Checking)\n"
                "Use execute_ui with action_type='validate' (or verify_goal for final completion checks).\n"
                "Be extremely strict with evidence."
            ),
            RESPONSE_DIRECTIVE,
        ]
        return "\n\n".join([part for part in parts if part.strip()])

    def __build_exploration_prompt(self) -> str:
        """
        Exploration Mode: systematic app mapping (depth-first).

        The VLM identifies ONE untried interactive element per call.
        Uses exploration context (already-tried actions from the KG) to avoid
        repeats; signals content_exhausted when all visible interactive
        elements on the current screen have been tried.
        """
        return (
            "Mobile App Explorer: discover all screens and features.\n\n"
            "TASK: Identify ONE interactive element not yet tried (see context). "
            "Tap it via execute_ui.\n\n"
            "RULES:\n"
            "- Coordinates: normalized (0-1000). bbox x,y = TOP-LEFT corner of element.\n"
            "- Prioritize: buttons, tabs, links, icons, menu items.\n"
            "- Avoid: labels, images, dividers.\n"
            "- Set screen_description: brief (1-2 sentence) screen summary.\n"
            "- If every visible interactive element is in 'ALREADY TRIED FROM THIS SCREEN', "
            "set content_exhausted=true and describe the screen. Do NOT repeat or invent actions.\n"
            "- Prefer elements that navigate to new screens (buttons, tabs, links).\n\n"
            "OUTPUT: ONE execute_ui call (one untried element) or content_exhausted=true if none remain.\n\n"
            + RESPONSE_DIRECTIVE
        )
