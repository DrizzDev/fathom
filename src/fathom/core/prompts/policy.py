"""Provider-neutral prompt policy.

This module owns *what* gets said to the LLM and *under what conditions*.
Adapter layers (e.g. ``adapters/prompts/gemini.py``) are responsible only
for *how* these sections are formatted for a specific provider — wrapping
in XML tags, joining, escaping, cache splits, etc.

The split keeps the Hexagonal architecture clean: workflow rules live in
core, formatting concerns live in adapters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from fathom.core.prompts.rules import (
    COMMON_RULES,
    CONFIDENCE_RULES,
    COORD_RULES,
    TOOL_GUIDANCE,
)
from fathom.interfaces.prompt import NamedSection

__all__ = ["NamedSection", "PromptPolicy"]


# ---------------------------------------------------------------------------
# Static policy text (provider-neutral)
# ---------------------------------------------------------------------------


PERSONA_BODY = (
    "You are a Mobile UI expert agent.\n"
    "COORDINATE MODE: NORMALIZED by default.\n"
    "Use normalized coordinates (0-1000) in 'bbox' unless you explicitly set "
    "coord_system='pixel'.\n"
    "When using bbox, x/y are the CENTER of the target element."
)


OUTPUT_REQUIREMENTS_BODY = (
    "OUTPUT REQUIREMENTS:\n"
    f"- {COORD_RULES}\n"
    f"- {CONFIDENCE_RULES}\n"
    "- REQUIRED: You MUST include 'label_id' from manifest for every interaction.\n"
    "- REQUIRED: For EVERY UI action you MUST fill 'target_name' with a concrete, "
    "user-facing label (e.g., 'Search box', 'Add to cart button'). This is the "
    "single canonical target field. Set 'script_target' ONLY when 'target_type' "
    "is 'positional' or 'dynamic' to provide an abstracted phrase (e.g., 'the "
    "first search result'). Do NOT fill any other target alias. NEVER use generic "
    "placeholders like 'UI Element', 'element', 'button', 'label', 'icon', 'field', "
    "or 'text' as the only description of a target.\n"
    "- REQUIRED: Every primary tool call MUST include 'sub_goal_completed'.\n"
    "- Do NOT set 'goal_completed' — overall goal completion is handled by verify_goal only.\n"
    "- Return tool call(s) only, with schema-valid fields.\n"
    "\nExecute next best step via tool using current user-provided goal and context."
)


SUB_GOAL_FOCUS_TEMPLATE = (
    "Progress: [{position}/{total}]\n"
    "Task: {description}\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. Focus EXCLUSIVELY on completing this task\n"
    "2. Do NOT attempt to complete future steps\n"
    "3. When this task is FULLY COMPLETED, signal completion by "
    "setting 'sub_goal_completed: true' in your tool call\n"
    "4. The system will automatically advance to the next step\n"
    "5. For validation tasks: after confirming all conditions are met, "
    "you MUST set 'sub_goal_completed: true' to advance\n"
    "6. WHEN TO SET sub_goal_completed (CRITICAL):\n"
    "   ONLY set 'sub_goal_completed: true' on the action that DIRECTLY "
    "fulfills the sub-goal's objective. Ask yourself: 'Does this specific "
    "action complete the task described in the sub-goal?'\n"
    "   YES — set sub_goal_completed:\n"
    "   - Task says 'select first result' → tap on the result item\n"
    "   - Task says 'search for X' → tap search/submit after typing X\n"
    "   - Task says 'add item to cart' → tap the Add/+ button\n"
    "   - Task says 'open app' → tap the app icon\n"
    "   - Task says 'set location' → tap the confirm/save location button\n"
    "   - Task says 'swipe left to reach X section' → the swipe that reveals X\n"
    "   - Task says 'type password' → the type action entering the password\n"
    "   - Task says 'enter address' → the type action entering the address\n"
    "   NO — do NOT set sub_goal_completed:\n"
    "   - Tapping a search field to focus it (preparatory step)\n"
    "   - Typing text when a submit/confirm step still follows\n"
    "   - Dismissing popups, overlays, permission dialogs, banners\n"
    "   - Scrolling/swiping to find an element (not the goal itself)\n"
    "   - Tapping back/home for navigation recovery"
)


APP_LAUNCH_TEMPLATE = (
    "Target app package: {package}\n"
    "When the app needs to be launched or brought to foreground:\n"
    "1. Do NOT emit an explicit 'tap' action on the app icon\n"
    "2. Instead, rely on the system's automatic OPEN_APP normalization\n"
    "3. Signal completion/focus goals directly via completion flags"
)


MEMORY_LEDGER_BODY_TEMPLATE = "Persistent memory (use store_memory/recall_memory tools):\n{ledger}"


SCREEN_CHANGE_NOTICE_TEMPLATE = (
    "The screen has CHANGED since the last recorded action. "
    "Previous screen observations in the trace are now OUTDATED. "
    "Current screen hash: {current_hash}. "
    "Analyze the NEW screenshot provided, not past observations."
)


SYSTEM_OVERRIDE_TEMPLATE = (
    "  <INSTRUCTION>\n"
    "{instructions}\n"
    "  </INSTRUCTION>\n"
    "  <WARNING>\n"
    "    This is a meta-instruction for the agent's behavior.\n"
    "    Do NOT treat this as content to be typed or searched.\n"
    "    You MUST adjust your plan to comply with this override.\n"
    "  </WARNING>"
)


SYSTEM_ALERT_TEMPLATE = "CRITICAL: {note}"


LOOP_RISK_ALERT_BODY = "Loop risk detected; avoid repeating the same ineffective action."


FAILURE_ALERT_TEMPLATE = (
    "CRITICAL: The following actions have FAILED or been repeated without progress "
    "on this screen. You MUST choose a DIFFERENT action or approach to achieve the "
    "same goal.\nFailed: {failed_actions}"
)


LAST_ACTION_TEMPLATE = "Most recent action: {last_action}"


RETRY_CORRECTION_TEMPLATE = (
    "Your previous tool call was rejected: {message}\n"
    "You MUST call the tool again with a DIFFERENT action. "
    "Choose an alternative approach to achieve the same goal."
)


def build_retry_correction_prompt(message: str) -> str:
    """Render the user-side correction prompt sent on a tool-validation retry."""

    return RETRY_CORRECTION_TEMPLATE.format(message=message)


# Intent vocabulary used to derive contextual rules. Provider-neutral.
_LOOP_INTENT_TOKENS = ("every", "all")
_TYPE_INTENT_TOKENS = ("type", "enter", "input")
_SEARCH_FLOW_TOKENS = ("tap", "select", "click")


# ---------------------------------------------------------------------------
# Policy class
# ---------------------------------------------------------------------------


class PromptPolicy:
    """Composes provider-neutral prompt sections from runtime inputs.

    Adapters call ``system_sections()`` for the cache-stable system
    instruction and ``user_context_sections()`` for per-step dynamic
    context. Adapters then render these sections into provider-specific
    text — they should never make policy decisions of their own.
    """

    @staticmethod
    def system_sections() -> List[NamedSection]:
        """Return the cache-stable system instruction sections.

        These sections are constant for the lifetime of a session and
        should be safe to cache by the LLM provider.
        """

        return [
            NamedSection(name="PERSONA", body=PERSONA_BODY, wrap=False),
            NamedSection(name="TOOL_GUIDANCE", body=TOOL_GUIDANCE, wrap=False),
            NamedSection(name="COMMON_RULES", body=COMMON_RULES, wrap=False),
            NamedSection(
                name="OUTPUT_REQUIREMENTS",
                body=OUTPUT_REQUIREMENTS_BODY,
                wrap=False,
            ),
        ]

    @staticmethod
    def user_context_sections(
        *,
        intent: str,
        hints: Optional[Mapping[str, Any]],
        memory_ledger: Optional[str],
        milestones: Optional[List[str]],
        formatted_trace: Optional[str],
        sub_goal_info: Optional[Mapping[str, Any]],
        guidance: Optional[List[str]],
        tracking_note: Optional[str],
        current_screen_hash: Optional[str],
        last_trace_hash: Optional[str],
        loop_risk: bool = False,
        failed_actions: Optional[Sequence[str]] = None,
        last_action: Optional[str] = None,
        delta_context_json: Optional[str] = None,
    ) -> List[NamedSection]:
        """Return the dynamic per-step context sections.

        Section composition rules (this is the workflow policy that
        previously lived in the Gemini adapter):

        * ``TASK_CONTEXT`` is included whenever there is a runtime brief
          (intent, screen resolution, contextual rules).
        * ``CURRENT_STEP`` is included only when ``sub_goal_info`` has at
          least two sub-goals — sequential intent execution mode.
        * ``APP_LAUNCH_SEMANTICS`` is included when ``hints`` carries a
          known package name.
        * ``MEMORY_LEDGER`` is included when there is non-empty ledger text.
        * ``MILESTONES`` is included when the caller supplies any.
        * ``CURRENT_TRACE`` is included when there is formatted trace text.
        * ``SCREEN_CHANGE_NOTICE`` is included when the live screen hash
          differs from the trace's last screen hash.
        * ``SYSTEM_OVERRIDE`` is included when guidance items are present.
        * ``SYSTEM_ALERT`` is appended last whenever a tracking note is set,
          to maximize recency bias.
        """

        sections: List[NamedSection] = []
        hints_map: Dict[str, Any] = dict(hints or {})

        runtime_brief = PromptPolicy.__build_runtime_brief(intent=intent, hints=hints_map)
        if runtime_brief:
            sections.append(NamedSection(name="TASK_CONTEXT", body=runtime_brief))

        sub_goal_section = PromptPolicy.__build_sub_goal_focus(sub_goal_info)
        if sub_goal_section is not None:
            sections.append(sub_goal_section)

        app_launch_section = PromptPolicy.__build_app_launch(hints_map)
        if app_launch_section is not None:
            sections.append(app_launch_section)

        if memory_ledger:
            sections.append(
                NamedSection(
                    name="MEMORY_LEDGER",
                    body=MEMORY_LEDGER_BODY_TEMPLATE.format(ledger=memory_ledger),
                )
            )

        if milestones:
            milestones_body = "\n".join(f"- {item}" for item in milestones)
            sections.append(NamedSection(name="MILESTONES", body=milestones_body))

        if formatted_trace:
            sections.append(NamedSection(name="CURRENT_TRACE", body=formatted_trace))

        if current_screen_hash and last_trace_hash and last_trace_hash != current_screen_hash:
            sections.append(
                NamedSection(
                    name="SCREEN_CHANGE_NOTICE",
                    body=SCREEN_CHANGE_NOTICE_TEMPLATE.format(current_hash=current_screen_hash),
                )
            )

        if guidance:
            instructions = "\n".join(f"    - {item}" for item in guidance)
            sections.append(
                NamedSection(
                    name="SYSTEM_OVERRIDE",
                    body=SYSTEM_OVERRIDE_TEMPLATE.format(instructions=instructions),
                )
            )

        if tracking_note:
            sections.append(
                NamedSection(
                    name="SYSTEM_ALERT",
                    body=SYSTEM_ALERT_TEMPLATE.format(note=tracking_note),
                )
            )

        if loop_risk:
            sections.append(NamedSection(name="SYSTEM_ALERT", body=LOOP_RISK_ALERT_BODY))

        if failed_actions:
            failed_text = "; ".join(failed_actions)
            sections.append(
                NamedSection(
                    name="SYSTEM_ALERT",
                    body=FAILURE_ALERT_TEMPLATE.format(failed_actions=failed_text),
                )
            )

        if last_action:
            sections.append(
                NamedSection(
                    name="LAST_ACTION",
                    body=LAST_ACTION_TEMPLATE.format(last_action=last_action),
                )
            )

        if delta_context_json:
            sections.append(NamedSection(name="DELTA_CONTEXT", body=delta_context_json))

        return sections

    # ------------------------------------------------------------------
    # Internal composition helpers — kept private so adapters cannot
    # accidentally couple to the heuristics.
    # ------------------------------------------------------------------

    @staticmethod
    def __build_runtime_brief(*, intent: str, hints: Mapping[str, Any]) -> str:
        lines: List[str] = []

        if intent:
            lines.append(f"GOAL: {intent}")

        width = hints.get("screen_width")
        height = hints.get("screen_height")
        if width and height:
            lines.append(f"SCREEN_RESOLUTION: {width}x{height}")

        if hints.get("use_xml"):
            lines.append("XML_GROUNDING: enabled")

        rules = PromptPolicy.__contextual_rules(intent=intent, hints=hints)
        if rules:
            lines.append(rules)

        notes = PromptPolicy.__conditional_notes(intent=intent, hints=hints)
        if notes:
            lines.append(notes)

        return "\n".join(lines)

    @staticmethod
    def __contextual_rules(*, intent: str, hints: Mapping[str, Any]) -> str:
        rules: List[str] = []
        intent_lower = intent.lower()

        if hints.get("use_xml"):
            rules.append("- XML Grounding enabled.")

        if any(token in intent_lower for token in _LOOP_INTENT_TOKENS):
            rules.append("- LOOP: Iterate untried matching elements. Avoid repeats.")

        if any(token in intent_lower for token in _TYPE_INTENT_TOKENS):
            rules.append(
                "- CRITICAL SEQ: Use 'tap' to gain focus on the input field, followed by 'type'."
            )

        return "RULES:\n" + "\n".join(rules) if rules else ""

    @staticmethod
    def __conditional_notes(*, intent: str, hints: Mapping[str, Any]) -> str:
        notes: List[str] = []
        intent_lower = intent.lower()

        typing_text = hints.get("typing_text")
        if typing_text:
            notes.append(f"- TYPING INTENT: Use literal text_to_type='{typing_text}'.")

        if "search" in intent_lower and any(token in intent_lower for token in _SEARCH_FLOW_TOKENS):
            notes.append("- SEARCH FLOW: If suggestions are visible, type then tap suggestion.")

        notes.append("- COMPLETE CHECK: If goal appears fully achieved, verify goal explicitly.")

        return "NOTES:\n" + "\n".join(notes)

    @staticmethod
    def __build_sub_goal_focus(
        sub_goal_info: Optional[Mapping[str, Any]],
    ) -> Optional[NamedSection]:
        if not sub_goal_info:
            return None

        index = sub_goal_info.get("index")
        total = sub_goal_info.get("total", 0)
        description = sub_goal_info.get("description")

        if index is None or total <= 1:
            return None

        body = SUB_GOAL_FOCUS_TEMPLATE.format(
            position=index + 1,
            total=total,
            description=description,
        )
        return NamedSection(name="CURRENT_STEP", body=body)

    @staticmethod
    def __build_app_launch(hints: Mapping[str, Any]) -> Optional[NamedSection]:
        package = hints.get("package_name")
        if not package or package == "unknown":
            return None

        return NamedSection(
            name="APP_LAUNCH_SEMANTICS",
            body=APP_LAUNCH_TEMPLATE.format(package=package),
        )
