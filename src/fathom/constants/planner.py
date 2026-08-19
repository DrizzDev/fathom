from __future__ import annotations

from enum import StrEnum


class PlannerEventCategory(StrEnum):
    """
    Discriminates the family of a planner observability event surfaced to the graph boundary.
    """

    ESCALATION = "ESCALATION"
    GUARD = "GUARD"
    TOOL_SCOPE = "TOOL_SCOPE"
    COMMAND_REJECTED = "COMMAND_REJECTED"


class PlannerEventKind(StrEnum):
    """
    Names the specific escalation or guard event within its family.
    """

    ESCALATION_DETECTED = "ESCALATION_DETECTED"
    ESCALATION_ALLOWED = "ESCALATION_ALLOWED"
    ESCALATION_DEFERRED = "ESCALATION_DEFERRED"
    ASK_USER_EMITTED = "ASK_USER_EMITTED"
    ACTION_BLOCKED = "ACTION_BLOCKED"
    GUARD_BYPASSED = "GUARD_BYPASSED"


class EscalationPath(StrEnum):
    """
    Identifies which planner path produced an escalation event.
    """

    PLANNER_SYNTHESIZED = "PLANNER_SYNTHESIZED"
    LLM_TOOL = "LLM_TOOL"


class GuardOutcome(StrEnum):
    """
    The verdict of the action guard for a proposed action.
    """

    ALLOW = "ALLOW"
    SILENT_REJECTION = "SILENT_REJECTION"
    CURRENT_SCREEN_REPEAT = "CURRENT_SCREEN_REPEAT"
    REPEATING_ON_SCREEN = "REPEATING_ON_SCREEN"
