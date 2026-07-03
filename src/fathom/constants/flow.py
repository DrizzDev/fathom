from __future__ import annotations

from enum import IntEnum, StrEnum


class NodeKind(StrEnum):
    """
    Semantic category of a target-neutral flow node.
    """

    TAP = "TAP"
    TYPE = "TYPE"
    SCROLL = "SCROLL"
    SCROLL_UNTIL = "SCROLL_UNTIL"

    WAIT = "WAIT"
    BACK = "BACK"
    KILL = "KILL"
    CLEAR = "CLEAR"
    LAUNCH = "LAUNCH"
    MINIMIZE = "MINIMIZE"

    MAP = "MAP"
    LOCATION = "LOCATION"

    STORE = "STORE"
    CHECK = "CHECK"
    BRANCH = "BRANCH"


class LaunchProvenance(StrEnum):
    """
    Why a launch was synthesised; room to add EXTERNAL_APP_LAUNCH (deep link/share) later.
    """

    LAUNCHER_TRANSITION = "LAUNCHER_TRANSITION"
    SYNTHETIC_WARM_START = "SYNTHETIC_WARM_START"


class CheckKind(StrEnum):
    """
    Category of a validation assertion supported by the renderer.
    """

    VISIBLE = "VISIBLE"
    PRESENT = "PRESENT"
    ENABLED = "ENABLED"
    DISABLED = "DISABLED"


class ScrollDirection(StrEnum):
    """
    Direction of a scroll gesture.
    """

    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class EvidenceMarker(StrEnum):
    """
    Recorded marker values; lowercase to match the planner's recorded condition data.
    """

    RECOVERY = "recovery"
    LOOP_RATIONALE = "Loop detected"


class IssueCode(StrEnum):
    """
    Identifier for a problem raised by a validation gate.
    """

    SYNTAX_ERROR = "SYNTAX_ERROR"
    EMPTY_SCRIPT = "EMPTY_SCRIPT"
    BASELINE_UNAVAILABLE = "BASELINE_UNAVAILABLE"

    REDUNDANT_WAIT = "REDUNDANT_WAIT"
    REDUNDANT_BRANCH = "REDUNDANT_BRANCH"
    UNRENDERABLE_VALUE = "UNRENDERABLE_VALUE"
    ROUND_TRIP_MISMATCH = "ROUND_TRIP_MISMATCH"

    STRAY_LAUNCH = "STRAY_LAUNCH"
    MISSING_LAUNCH = "MISSING_LAUNCH"
    LAUNCH_MISMATCH = "LAUNCH_MISMATCH"
    UNGROUNDED_LAUNCH = "UNGROUNDED_LAUNCH"

    REDUNDANT_SCROLL = "REDUNDANT_SCROLL"
    UNGROUNDED_STORE = "UNGROUNDED_STORE"
    UNGROUNDED_SCROLL = "UNGROUNDED_SCROLL"
    UNGROUNDED_CONDITION = "UNGROUNDED_CONDITION"
    UNGUARDED_CONDITIONAL = "UNGUARDED_CONDITIONAL"

    TAP_TARGET_MISMATCH = "TAP_TARGET_MISMATCH"
    TYPE_CONTENT_MISMATCH = "TYPE_CONTENT_MISMATCH"
    WAIT_SUBJECT_MISMATCH = "WAIT_SUBJECT_MISMATCH"
    SCROLL_DIRECTION_MISMATCH = "SCROLL_DIRECTION_MISMATCH"
    VALIDATION_SUBJECT_MISMATCH = "VALIDATION_SUBJECT_MISMATCH"

    RECOVERY_NODE = "RECOVERY_NODE"
    DANGLING_PROVENANCE = "DANGLING_PROVENANCE"

    MISSING_PARTIAL = "MISSING_PARTIAL"
    INVENTED_VALIDATION = "INVENTED_VALIDATION"
    MISSING_GOAL_VALIDATION = "MISSING_GOAL_VALIDATION"


class Limit(IntEnum):
    """
    Bounded numeric limits for generation control flow.
    """

    MAX_REPAIR_ATTEMPTS = 2
