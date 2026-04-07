from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

from fathom.core.prompts.policy import NamedSection, PromptPolicy
from fathom.interfaces.prompt import PromptBuilder, PromptUserContext

logger = logging.getLogger(__name__)


class GeminiPromptBuilder(PromptBuilder):
    """
    Provider-specific Gemini prompt renderer.

    All workflow policy (sub-goal completion rules, app-launch semantics,
    output requirements, intent heuristics) lives in
    ``fathom.core.prompts.policy``. This adapter is intentionally thin: it
    only translates provider-neutral ``NamedSection`` objects into Gemini's
    XML-tagged format and stitches them together.
    """

    def build(self) -> str:
        """Render the cache-stable system instruction."""

        sections = PromptPolicy.system_sections()
        return self.__render_sections(sections)

    def build_user_context(self, context: PromptUserContext) -> str:
        """Render the dynamic per-step user context from a typed contract."""

        memory_ledger = self.__format_ledger(memory=context.memory)
        formatted_trace = self.__format_trace(
            trace=context.trace,
            current_screen_hash=context.current_screen_hash,
        )
        last_trace_hash = self.__last_trace_hash(trace=context.trace)

        # Build the provider-neutral hints map expected by PromptPolicy.
        hints: Dict[str, Any] = {
            "use_xml": context.use_xml,
        }
        if context.screen_width is not None:
            hints["screen_width"] = context.screen_width
        if context.screen_height is not None:
            hints["screen_height"] = context.screen_height
        if context.package_name:
            hints["package_name"] = context.package_name
        if context.typing_text is not None:
            hints["typing_text"] = context.typing_text

        sub_goal_map: Optional[Dict[str, Any]] = None
        if context.sub_goal_info is not None:
            sub_goal_map = {
                "index": context.sub_goal_info.index,
                "total": context.sub_goal_info.total,
                "description": context.sub_goal_info.description,
            }

        sections = PromptPolicy.user_context_sections(
            intent=context.intent,
            hints=hints,
            memory_ledger=memory_ledger,
            milestones=list(context.milestones) if context.milestones else None,
            formatted_trace=formatted_trace,
            sub_goal_info=sub_goal_map,
            guidance=list(context.guidance) if context.guidance else None,
            tracking_note=context.tracking_note,
            current_screen_hash=context.current_screen_hash,
            last_trace_hash=last_trace_hash,
        )

        for section in sections:
            if section.name == "CURRENT_STEP" and context.sub_goal_info is not None:
                description = context.sub_goal_info.description[:50]
                logger.debug(
                    "[H3] Single Sub-goal Focus | step=%s/%s | task=%s",
                    context.sub_goal_info.index + 1,
                    context.sub_goal_info.total,
                    description,
                )
            elif section.name == "APP_LAUNCH_SEMANTICS":
                logger.debug("[H3] App Launch Semantics | package=%s", context.package_name)
            elif section.name == "MEMORY_LEDGER" and memory_ledger:
                logger.debug("[H3] Memory Ledger Added | ledger_length=%d", len(memory_ledger))

        if memory_ledger is None:
            logger.debug("[H3] No Memory Ledger | memory is empty or None")

        return self.__render_sections(sections)

    # ------------------------------------------------------------------
    # Provider-specific rendering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def __render_sections(sections: List[NamedSection]) -> str:
        """Wrap each section in Gemini's XML-style tags and join with blank lines."""

        rendered: List[str] = []
        for section in sections:
            body = section.body
            if not body or not body.strip():
                continue
            if section.wrap:
                rendered.append(f"<{section.name}>\n{body}\n</{section.name}>")
            else:
                rendered.append(body)
        return "\n\n".join(rendered)

    @staticmethod
    def __format_ledger(memory: Mapping[str, str]) -> Optional[str]:
        """Render persistent memory as a compact ``[k:v, ...]`` block."""

        if not memory:
            return None

        items = [
            f"{key}:{value}"
            for key, value in memory.items()
            if not key.startswith(("context:", "ctx_v3:", "ctx_"))
        ]
        if not items:
            return None
        return f"[{', '.join(items)}]"

    @staticmethod
    def __last_trace_hash(trace: Sequence[Mapping[str, Any]]) -> Optional[str]:
        """Extract the most recent screen hash from the trace, if any."""

        if not trace:
            return None
        last_obs = trace[-1].get("observation", "")
        if not isinstance(last_obs, str) or not last_obs.startswith("Screen: "):
            return None
        parts = last_obs.split(" ")
        return parts[1][:8] if len(parts) > 1 else None

    def __format_trace(
        self,
        trace: Sequence[Mapping[str, Any]],
        current_screen_hash: Optional[str] = None,
    ) -> str:
        """Format the trace as numbered interaction history with [PAST]/[CURRENT] tags."""

        if not trace:
            return ""

        lines: List[str] = []
        avoided: List[str] = []
        recent = list(trace)[-50:]

        for index, entry in enumerate(recent, 1):
            action = entry.get("action", {})
            observation = entry.get("observation", "Unknown screen")

            staleness = ""
            if current_screen_hash and observation.startswith("Screen: "):
                parts = observation.split(" ")
                entry_hash = parts[1][:8] if len(parts) > 1 else ""
                if entry_hash and entry_hash != current_screen_hash:
                    staleness = " [PAST]"
                else:
                    staleness = " [CURRENT]"

            if isinstance(action, dict):
                desc = action.get("target", "unknown")
                type_ = action.get("action_type", "tap")
            else:
                desc = getattr(action, "target", "unknown")
                type_ = getattr(action, "action_type", "tap")

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

    @staticmethod
    def resolve_version(model_name: str, use_xml: bool) -> str:
        """Determines the optimal prompt version based on Gemini model capabilities."""

        is_flash = "flash" in model_name.lower()
        tier = "flash" if is_flash else "pro"
        strategy = "xml" if use_xml else "vision"
        return f"{tier}_{strategy}"
