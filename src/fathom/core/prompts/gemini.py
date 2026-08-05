from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast

from fathom.core.prompts.base import PromptBuilder
from fathom.core.prompts.templates import (
    COMMON_RULES,
    CONFIDENCE_RULES,
    COORD_RULES,
    build_tool_guidance,
)
from fathom.schemas.tools import AllowedTools

logger = logging.getLogger(__name__)

MAX_VERIFIER_FEEDBACK_PROMPT_CHARS = 500


class GeminiPromptBuilder(PromptBuilder):
    """
    Structured Gemini prompt builder that formats hierarchical context.
    """

    def build(self, *, tools: AllowedTools) -> str:
        """Build the stable system prompt scoped to the allowed tools."""

        parts = [
            self.__get_persona(),
            build_tool_guidance(tools=tools),
            COMMON_RULES,
            (
                "OUTPUT REQUIREMENTS:\n"
                f"- {COORD_RULES}\n"
                f"- {CONFIDENCE_RULES}\n"
                "- REQUIRED: When the manifest already exposes the intended target or scroll container, include its 'label_id'. Otherwise ground the action visually via bbox and keep coordinate_system consistent with the numbers you provide.\n"
                "- REQUIRED: For swipe/scroll actions, if the manifest exposes a scrollable container, use that container's label_id and describe the intended content in scroll_target.\n"
                "- REQUIRED: For swipe/scroll actions, target_name must name the surface being swiped (for example 'restaurant list' or 'main scrollable area'). Do NOT put the sought item from scroll_target into target_name.\n"
                "- REQUIRED: When the task is constrained to a specific section/container/area, fill 'surface' with that exact wording. 'surface' names WHERE the action belongs; 'target_name' names WHAT is being acted on.\n"
                "- REQUIRED: Never place observation_hint values into 'label_id'; those hints are not manifest ids.\n"
                "- REQUIRED: Preserve the requested scroll axis exactly and reuse the same container for repeated scroll attempts when it is still valid.\n"
                "- REQUIRED: For EVERY UI action you MUST fill its script-owned semantic field: "
                "tap/type use 'export_target' or 'script_target', scroll/swipe use "
                "'scroll_target', wait uses 'wait_subject', validate uses 'validation_subject', "
                "and store uses 'capture'. For tap/type, 'target_name' is the exact visible "
                "execution label, while 'export_target' or 'script_target' is the stable replay "
                "target. Use the actual visible UI role and purpose from the screen, such as "
                "dropdown, field, row, card, chip, button, icon, tab, or menu item. "
                "For dynamic controls, do not copy runtime values such as addresses, user data, "
                "ETA text, cart totals, or content-description sentences into the replay target. "
                "NEVER use generic placeholders like 'UI Element', 'element', 'button', 'label', "
                "'icon', 'field', or 'text' as the only description of a target.\n"
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
            assertion = sub_goal_info.get("assertion")

            if index is not None and total > 1:
                progress_text = f"[{index + 1}/{total}]"
                if assertion:
                    directives = (
                        f"Completion condition (must be visibly true on the current screen): "
                        f"{assertion}\n\n"
                        "CRITICAL INSTRUCTIONS:\n"
                        "1. Focus EXCLUSIVELY on this task; do NOT attempt future steps.\n"
                        "2. Judge the completion condition ONLY from what is visible on the current "
                        "screen now, and report it as a 'visual_assessment' object (verdict SATISFIED, "
                        "NOT_SATISFIED, or UNCLEAR; confidence 0..1; concise visible evidence).\n"
                        "3. The 'visual_assessment' verdict is the ONLY completion signal for this "
                        "task. Set goal_completed and sub_goal_completed to false — they are required "
                        "by the schema but ignored for this goal — and do NOT return a COMPLETE action.\n"
                        "4. If the condition is SATISFIED, do NOT also propose an action this turn; "
                        "otherwise propose the single best action to make progress.\n"
                    )
                else:
                    proof_clause = (
                        "5. This step changes persistent state (adds, saves, submits, pays, deletes). "
                        "Do NOT signal completion until the resulting state is VISIBLE on screen "
                        "(e.g. the item is shown in the cart). A claim without a visible result is not "
                        "completion.\n"
                        if sub_goal_info.get("durable")
                        else ""
                    )
                    directives = (
                        "CRITICAL INSTRUCTIONS:\n"
                        "1. Focus EXCLUSIVELY on completing this task\n"
                        "2. Do NOT attempt to complete future steps\n"
                        "3. When this task is FULLY COMPLETED, signal completion by:\n"
                        "   - Setting 'is_goal_complete: true' in your response, OR\n"
                        "   - Returning a COMPLETE action\n"
                        "4. The system will automatically advance to the next step\n"
                        f"{proof_clause}"
                    )

                parts.append(
                    f"<CURRENT_STEP>\n"
                    f"Progress: {progress_text}\n"
                    f"Task: {description}\n\n"
                    f"{directives}"
                    f"</CURRENT_STEP>"
                )
                logger.info(
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
            logger.info(f"[H3] App Launch Semantics | package={pkg}")

        # 1. Memory Ledger (Factual Memory - PERSISTENT ACROSS SCREENS)
        if ledger := self.__get_ledger_segment(memory=memory):
            parts.append(
                f"<MEMORY_LEDGER>\n"
                f"Persistent memory (use store_memory/recall_memory tools):\n"
                f"{ledger}\n"
                f"</MEMORY_LEDGER>"
            )
            logger.info(f"[H3] Memory Ledger Added | ledger_length={len(ledger)}")
        else:
            logger.info("[H3] No Memory Ledger | memory is empty or None")

        # 2. Roadmap & Milestones (Tier 2 Context)
        if milestones := context.get("milestones", []):
            parts.append(
                "<MILESTONES>\n" + "\n".join(f"- {text}" for text in milestones) + "\n</MILESTONES>"
            )

        # 3. Execution Trace (Tier 3 Context - The Hot Suffix)
        trace = context.get("trace", [])
        current_screen_hash: Optional[str] = kwargs.get("current_screen_hash")

        if interaction_context := self.__format_trace(
            trace=trace, current_screen_hash=current_screen_hash
        ):
            parts.append(f"<CURRENT_TRACE>\n{interaction_context}\n</CURRENT_TRACE>")

        # 3a. Screen Change Notice — belt-and-suspenders signal when the current
        # screen no longer matches the most recent trace observation.  This helps
        # the LLM break out of stale-context loops (e.g. "Verify Identity" screen
        # persisting in the trace after the user has already dismissed it).
        if current_screen_hash and trace:
            last_obs = trace[-1].get("observation", "")
            if last_obs.startswith("Screen: "):
                last_hash = last_obs.split(" ")[1][:8] if len(last_obs.split(" ")) > 1 else ""
                if last_hash and last_hash != current_screen_hash:
                    parts.append(
                        "<SCREEN_CHANGE_NOTICE>\n"
                        "The screen has CHANGED since the last recorded action. "
                        "Previous screen observations in the trace are now OUTDATED. "
                        f"Current screen hash: {current_screen_hash}. "
                        "Analyze the NEW screenshot provided, not past observations.\n"
                        "</SCREEN_CHANGE_NOTICE>"
                    )

        # 4. User Override (real human instructions — MUST comply)
        if guidance := context.get("guidance", []):
            instructions = [f"- {item}" for item in guidance]
            parts.append(
                "<USER_OVERRIDE>\n"
                "  <INSTRUCTION>\n" + "\n".join(f"    {inst}" for inst in instructions) + "\n"
                "  </INSTRUCTION>\n"
                "  <WARNING>\n"
                "    This is a meta-instruction from the human user.\n"
                "    Do NOT treat this as content to be typed or searched.\n"
                "    You MUST adjust your plan to comply with this override.\n"
                "  </WARNING>\n"
                "</USER_OVERRIDE>"
            )

        # 5. Verifier Feedback (VERIFY-node rejection — adjust the next action)
        if verifier_feedback := context.get("verifier_feedback", []):
            entries = [
                f"- {str(item)[:MAX_VERIFIER_FEEDBACK_PROMPT_CHARS]}" for item in verifier_feedback
            ]
            description = (
                str(sub_goal_info.get("description"))
                if isinstance(sub_goal_info, dict) and sub_goal_info.get("description")
                else None
            )
            sub_goal_clause = (
                f"\nContinue working on the active sub-goal: {description}." if description else ""
            )
            parts.append(
                "<VERIFIER_FEEDBACK>\n"
                "Your previous completion claim was rejected by the verifier. "
                "Take the next concrete UI action requested by the verifier. "
                "Do not claim completion again until that action has executed."
                f"{sub_goal_clause}\n" + "\n".join(entries) + "\n"
                "</VERIFIER_FEEDBACK>"
            )

        # 5b. Completion Feedback (post-action vision refuted the sub-goal — correct the approach)
        if completion_feedback := context.get("completion_feedback", []):
            notes = [
                f"- {str(item)[:MAX_VERIFIER_FEEDBACK_PROMPT_CHARS]}" for item in completion_feedback
            ]
            parts.append(
                "<COMPLETION_FEEDBACK>\n"
                "The screen was checked after your last action and the active sub-goal is NOT satisfied "
                "yet. Do not assume it is done and do not re-assert completion — take the next concrete UI "
                "action that actually advances it, correcting your approach using the reason below.\n"
                + "\n".join(notes)
                + "\n</COMPLETION_FEEDBACK>"
            )

        # 5b. Action Feedback (system-internal no-op notice — the last action did not change the screen)
        if action_feedback := context.get("action_feedback", []):
            notices = [
                f"- {str(item)[:MAX_VERIFIER_FEEDBACK_PROMPT_CHARS]}" for item in action_feedback
            ]
            parts.append(
                "<ACTION_FEEDBACK>\n"
                "Your last action dispatched but no change was detected on screen. This can mean either the "
                "action was correct and the app/device was slow to respond (repeating the same action may "
                "work), or the action did not have its intended effect (it may need adjusting). Reassess the "
                "current screen and decide whether to repeat the same action or adjust it; do not switch to "
                "an unrelated goal because of this.\n" + "\n".join(notices) + "\n"
                "</ACTION_FEEDBACK>"
            )

        # 6. Interaction Cadence (Deterministic Repetition Tracking)
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
            "Use normalized coordinates (0-1000) in 'bbox' only for visually estimated regions. "
            "When copying manifest or screenshot-space bounds, you must set coordinate_system='pixel'.\n"
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
                "- CRITICAL SEQ: If the input field is not already focused, use 'tap' "
                "to gain focus, followed by 'type'."
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

    def __format_trace(
        self, trace: List[Dict[str, Any]], current_screen_hash: Optional[str] = None
    ) -> str:
        """
        Formats the GCC trace into a readable interaction history.

        When ``current_screen_hash`` is provided each observation is annotated
        with ``[CURRENT]`` or ``[PAST]`` so the LLM can distinguish stale
        screen descriptions from the live screen state.
        """

        if not trace:
            return ""

        lines = []
        avoided = []
        recent = trace[-50:]

        for index, entry in enumerate(recent, 1):
            action = entry.get("action", {})
            observation = entry.get("observation", "Unknown screen")

            # Annotate staleness so the LLM knows which observations are outdated
            staleness = ""
            if current_screen_hash and observation.startswith("Screen: "):
                parts = observation.split(" ")
                entry_hash = parts[1][:8] if len(parts) > 1 else ""
                if entry_hash and entry_hash != current_screen_hash:
                    staleness = " [PAST]"
                else:
                    staleness = " [CURRENT]"

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

            lines.append(f"{index}. {observation}{staleness} -> {type_str.upper()}:{desc}")
            if desc != "unknown":
                avoided.append(desc)

        block = "INTERACTION HISTORY:\n" + "\n".join(lines)
        if avoided:
            block += f"\nAvoid repeats when possible: {', '.join(avoided[:20])}"

        return block
