from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast

from fathom.core.prompts.base import PromptBuilder
from fathom.core.prompts.templates import (
    COMMON_RULES,
    CONFIDENCE_RULES,
    COORD_RULES,
    TOOL_GUIDANCE,
)

logger = logging.getLogger(__name__)


class GeminiPromptBuilder(PromptBuilder):
    """
    Structured Gemini prompt builder that formats hierarchical context.
    """

    def build(self) -> str:
        """
        Build stable system prompt for tool-based UI execution.
        """

        parts = [
            self.__get_persona(),
            TOOL_GUIDANCE,
            COMMON_RULES,
            (
                "OUTPUT REQUIREMENTS:\n"
                f"- {COORD_RULES}\n"
                f"- {CONFIDENCE_RULES}\n"
                "- REQUIRED: You MUST include 'label_id' from manifest for every interaction.\n"
                "- REQUIRED: Every primary tool call MUST include BOTH boolean flags: "
                "'goal_completed' and 'sub_goal_completed'.\n"
                "- REQUIRED: If action_type is 'complete', set BOTH flags to true.\n"
                "- Return tool call(s) only, with schema-valid fields.\n"
                "\nExecute next best step via tool using current user-provided goal and context."
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
        intent = str(kwargs.get("intent") or "").strip()
        hints_raw = kwargs.get("hints")
        hints: Dict[str, Any] = (
            cast("Dict[str, Any]", hints_raw) if isinstance(hints_raw, dict) else {}
        )

        parts = []

        # Runtime task metadata previously embedded in system prompt (kept dynamic for cache stability).
        if runtime_brief := self.__build_runtime_brief(intent=intent, hints=hints):
            parts.append(runtime_brief)

        # 1a. Sub-goal Progress (if sequential intent execution is active) - SINGLE FOCUS MODE
        # Only pass current sub-goal, no remaining steps list
        sub_goal_info = kwargs.get("sub_goal_info")
        if sub_goal_info and isinstance(sub_goal_info, dict):
            index = sub_goal_info.get("index")
            total = sub_goal_info.get("total", 0)
            description = sub_goal_info.get("description")

            if index is not None and total > 1:
                progress_text = f"[{index + 1}/{total}]"

                parts.append(
                    f"<CURRENT_STEP>\n"
                    f"Progress: {progress_text}\n"
                    f"Task: {description}\n\n"
                    f"CRITICAL INSTRUCTIONS:\n"
                    f"1. Focus EXCLUSIVELY on completing this task\n"
                    f"2. Do NOT attempt to complete future steps\n"
                    f"3. When this task is FULLY COMPLETED, signal completion by:\n"
                    f"   - Setting 'is_goal_complete: true' in your response, OR\n"
                    f"   - Returning a COMPLETE action\n"
                    f"4. The system will automatically advance to the next step\n"
                    f"</CURRENT_STEP>"
                )
                logger.debug(
                    f"[H3] Single Sub-goal Focus | step={index + 1}/{total} | "
                    f"task={(description or '')[:50]}"
                )

        # 1a-bis. App Launch Semantics (when package is known)
        if hints and hints.get("package_name") and hints.get("package_name") != "unknown":
            pkg = hints.get("package_name")
            parts.append(
                f"<APP_LAUNCH_SEMANTICS>\n"
                f"Target app package: {pkg}\n"
                f"When the app needs to be launched or brought to foreground:\n"
                f"1. Do NOT emit an explicit 'tap' action on the app icon\n"
                f"2. Instead, rely on the system's automatic OPEN_APP normalization\n"
                f"3. Signal completion/focus goals directly via completion flags\n"
                f"</APP_LAUNCH_SEMANTICS>"
            )
            logger.debug(f"[H3] App Launch Semantics | package={pkg}")

        # 1. Memory Ledger (Factual Memory - PERSISTENT ACROSS SCREENS)
        if ledger := self.__get_ledger_segment(memory=memory):
            parts.append(
                f"<MEMORY_LEDGER>\n"
                f"Persistent memory (use store_memory/recall_memory tools):\n"
                f"{ledger}\n"
                f"</MEMORY_LEDGER>"
            )
            logger.debug(f"[H3] Memory Ledger Added | ledger_length={len(ledger)}")
        else:
            logger.debug("[H3] No Memory Ledger | memory is empty or None")

        # 2. Roadmap & Milestones (Tier 2 Context)
        if milestones := context.get("milestones", []):
            parts.append(
                "<MILESTONES>\n" + "\n".join(f"- {text}" for text in milestones) + "\n</MILESTONES>"
            )

        # 3. Execution Trace (Tier 3 Context - The Hot Suffix)
        trace = context.get("trace", [])

        if interaction_context := self.__format_trace(trace=trace):
            parts.append(f"<CURRENT_TRACE>\n{interaction_context}\n</CURRENT_TRACE>")

        # 4. Priority Guidance (HITL) - The "System Override"
        if guidance := context.get("guidance", []):
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

        # 5. Interaction Cadence (Deterministic Repetition Tracking)
        # Placed LAST to ensure maximum recency bias and adherence when stuck
        if tracking_note:
            parts.append(f"<SYSTEM_ALERT>\nCRITICAL: {tracking_note}\n</SYSTEM_ALERT>")

        return "\n\n".join(parts)

    def __get_persona(self) -> str:
        """
        Core identity with a cache-stable instruction footprint.
        """

        return (
            "You are a Mobile UI expert agent.\n"
            "COORDINATE MODE: NORMALIZED by default.\n"
            "Use normalized coordinates (0-1000) in 'bbox' unless you explicitly set "
            "coord_system='pixel'.\n"
            "When using bbox, x/y are TOP-LEFT and width/height extend right/down."
        )

    def __build_runtime_brief(self, intent: str, hints: Dict[str, Any]) -> str:
        """
        Build dynamic task guidance that should live in user payload.
        """

        runtime_lines: List[str] = []
        if intent:
            runtime_lines.append(f"GOAL: {intent}")

        if (w := hints.get("screen_width")) and (h := hints.get("screen_height")):
            runtime_lines.append(f"SCREEN_RESOLUTION: {w}x{h}")

        if hints.get("use_xml"):
            runtime_lines.append("XML_GROUNDING: enabled")

        contextual_rules = self.__get_contextual_rules(intent=intent, hints=hints)
        if contextual_rules:
            runtime_lines.append(contextual_rules)

        conditional_notes = self.__get_conditional_notes(intent=intent, hints=hints)
        if conditional_notes:
            runtime_lines.append(conditional_notes)

        if not runtime_lines:
            return ""

        return "<TASK_CONTEXT>\n" + "\n".join(runtime_lines) + "\n</TASK_CONTEXT>"

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

        return "RULES:\n" + "\n".join(rules) if rules else ""

    def __get_conditional_notes(self, intent: str, hints: Optional[Dict[str, Any]]) -> str:
        """
        Add concise behavior notes.
        """

        notes: List[str] = []
        intent_lower = intent.lower()

        if hints and hints.get("typing_text"):
            text = str(hints["typing_text"])
            notes.append(f"- TYPING INTENT: Use literal text_to_type='{text}'.")

        if "search" in intent_lower and any(k in intent_lower for k in ["tap", "select", "click"]):
            notes.append("- SEARCH FLOW: If suggestions are visible, type then tap suggestion.")

        notes.append("- COMPLETE CHECK: If goal appears fully achieved, verify goal explicitly.")
        # notes.append(
        #     "- DISABLED ELEMENTS: Do NOT interact with elements marked as '[DISABLED]' in the manifest."
        # )

        return "NOTES:\n" + "\n".join(notes)

    def __get_ledger_segment(self, memory: Optional[Dict[str, str]]) -> str:
        """
        High-density ledger memory.
        """

        if not memory:
            return ""

        # Filter out internal system keys to prevent context explosion
        # KEEP: user_guidance, task state, user preferences
        items = [
            f"{key}:{value}"
            for key, value in memory.items()
            if not key.startswith(("context:", "ctx_v3:", "ctx_"))
        ]

        if not items:
            return ""

        return f"[{', '.join(items)}]"

    def __format_trace(self, trace: List[Dict[str, Any]]) -> str:
        """
        Formats the GCC trace into a readable interaction history.
        """

        if not trace:
            return ""

        lines = []
        avoided = []
        recent = trace[-50:]

        for index, entry in enumerate(recent, 1):
            action = entry.get("action", {})
            observation = entry.get("observation", "Unknown screen")

            # Action might be dict or object
            if isinstance(action, dict):
                desc = action.get("target", "unknown")
                type_ = action.get("action_type", "tap")
            else:
                desc = getattr(action, "target", "unknown")
                type_ = getattr(action, "action_type", "tap")

            # Convert enum to string if needed
            type_str = (
                type_.value
                if hasattr(type_, "value") and not isinstance(type_, str)
                else str(type_)
            )

            lines.append(f"{index}. {observation} -> {type_str.upper()}:{desc}")
            if desc != "unknown":
                avoided.append(desc)

        block = "INTERACTION HISTORY:\n" + "\n".join(lines)
        if avoided:
            block += f"\nAvoid repeats when possible: {', '.join(avoided[:20])}"

        return block
