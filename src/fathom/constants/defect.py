"""
Constants for exploration defect detection.
"""

from __future__ import annotations

from enum import StrEnum


class DefectKind(StrEnum):
    """
    Broad category a detected defect belongs to.

    UI         - A visual or layout flaw the user can see (overlap, clipping, contrast).
    FUNCTIONAL - Broken behaviour the user can hit (dead tap, crash, error dialog).
    CONTENT    - Wrong or unfinished content (placeholder copy, broken image).
    """

    UI = "ui"
    FUNCTIONAL = "functional"
    CONTENT = "content"


class DefectSeverity(StrEnum):
    """
    How badly a defect degrades the user experience.
    """

    BLOCKER = "blocker"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"

    @property
    def rank(self) -> int:
        """
        Sort rank; lower sorts first, so the most severe defects lead the report.
        """

        return _SEVERITY_RANK[self]


# Ascending rank for sorting: blocker leads, info trails.
_SEVERITY_RANK: dict[DefectSeverity, int] = {
    DefectSeverity.BLOCKER: 0,
    DefectSeverity.MAJOR: 1,
    DefectSeverity.MINOR: 2,
    DefectSeverity.INFO: 3,
}


class DefectSource(StrEnum):
    """
    Stage of the run that produced a defect.

    INLINE   - Flagged live during the crawl from a step's runtime signals.
    POST_RUN - Flagged afterward by analysing a captured screen.
    """

    INLINE = "inline"
    POST_RUN = "post_run"


class DefectVerification(StrEnum):
    """
    How much the pipeline trusts a defect, gating whether it leads the report.

    CONFIRMED    - Corroborated enough to surface in the headline defect list.
    NEEDS_REVIEW - An uncorroborated signal held back for manual triage so it
                   does not inflate the headline count.
    """

    CONFIRMED = "confirmed"
    NEEDS_REVIEW = "needs_review"


class DefectSignal(StrEnum):
    """
    The specific observation that evidences a defect.
    """

    DEAD_TAP = "dead_tap"
    CRASH = "crash"
    LEFT_PACKAGE = "left_package"
    BLANK_CAPTURE = "blank_capture"
    ERROR_DIALOG = "error_dialog"
    INFINITE_SPINNER = "infinite_spinner"
    BROKEN_NAVIGATION = "broken_navigation"
    PLACEHOLDER_TEXT = "placeholder_text"
    LOREM_IPSUM = "lorem_ipsum"
    TODO_TEXT = "todo_text"
    BROKEN_IMAGE = "broken_image"
    UNTRANSLATED_STRING = "untranslated_string"
    EMPTY_STATE = "empty_state"
    OVERLAP_CLIPPING = "overlap_clipping"
    CONTRAST = "contrast"
    SPELLING = "spelling"

    @property
    def kind(self) -> DefectKind:
        """
        Broad category this signal evidences.
        """

        return _SIGNAL_KIND[self]

    @property
    def default_severity(self) -> DefectSeverity:
        """
        Severity assigned when a detector does not override it.
        """

        return _SIGNAL_SEVERITY[self]


# Every signal maps to exactly one kind; the kind property raises on omission.
_SIGNAL_KIND: dict[DefectSignal, DefectKind] = {
    DefectSignal.DEAD_TAP: DefectKind.FUNCTIONAL,
    DefectSignal.CRASH: DefectKind.FUNCTIONAL,
    DefectSignal.LEFT_PACKAGE: DefectKind.FUNCTIONAL,
    DefectSignal.BLANK_CAPTURE: DefectKind.FUNCTIONAL,
    DefectSignal.ERROR_DIALOG: DefectKind.FUNCTIONAL,
    DefectSignal.INFINITE_SPINNER: DefectKind.FUNCTIONAL,
    DefectSignal.BROKEN_NAVIGATION: DefectKind.FUNCTIONAL,
    DefectSignal.PLACEHOLDER_TEXT: DefectKind.CONTENT,
    DefectSignal.LOREM_IPSUM: DefectKind.CONTENT,
    DefectSignal.TODO_TEXT: DefectKind.CONTENT,
    DefectSignal.BROKEN_IMAGE: DefectKind.CONTENT,
    DefectSignal.UNTRANSLATED_STRING: DefectKind.CONTENT,
    DefectSignal.EMPTY_STATE: DefectKind.CONTENT,
    DefectSignal.SPELLING: DefectKind.CONTENT,
    DefectSignal.OVERLAP_CLIPPING: DefectKind.UI,
    DefectSignal.CONTRAST: DefectKind.UI,
}


# Default severity per signal; a detector may override when context warrants.
_SIGNAL_SEVERITY: dict[DefectSignal, DefectSeverity] = {
    DefectSignal.DEAD_TAP: DefectSeverity.MINOR,
    DefectSignal.CRASH: DefectSeverity.BLOCKER,
    DefectSignal.LEFT_PACKAGE: DefectSeverity.MAJOR,
    DefectSignal.BLANK_CAPTURE: DefectSeverity.MAJOR,
    DefectSignal.ERROR_DIALOG: DefectSeverity.MAJOR,
    DefectSignal.INFINITE_SPINNER: DefectSeverity.MAJOR,
    DefectSignal.BROKEN_NAVIGATION: DefectSeverity.MAJOR,
    DefectSignal.PLACEHOLDER_TEXT: DefectSeverity.MINOR,
    DefectSignal.LOREM_IPSUM: DefectSeverity.MAJOR,
    DefectSignal.TODO_TEXT: DefectSeverity.MINOR,
    DefectSignal.BROKEN_IMAGE: DefectSeverity.MAJOR,
    DefectSignal.UNTRANSLATED_STRING: DefectSeverity.MINOR,
    DefectSignal.EMPTY_STATE: DefectSeverity.MINOR,
    DefectSignal.SPELLING: DefectSeverity.MINOR,
    DefectSignal.OVERLAP_CLIPPING: DefectSeverity.MINOR,
    DefectSignal.CONTRAST: DefectSeverity.MINOR,
}


# Markers of unfinished copy reliable enough to flag from a screen's text, mapped
# to the signal each raises (matched on word boundaries). Words like "placeholder",
# "skeleton", "dummy", and "todo" are deliberately excluded: they appear in
# legitimate UI descriptions (loading skeletons, search-field placeholder hints)
# and produced false positives. The vision detector catches placeholder/TODO copy
# from the real screenshot instead.
PLACEHOLDER_SIGNALS: dict[str, DefectSignal] = {
    "lorem ipsum": DefectSignal.LOREM_IPSUM,
    "lorem": DefectSignal.LOREM_IPSUM,
}


# Planner-confidence floor below which a dead tap is only strong enough for
# needs-review, not the headline: a low-confidence tap that changed nothing is
# more likely a grounding miss than a genuinely inert control.
DEAD_TAP_MIN_CONFIDENCE: float = 0.5


# Telemetry event name carrying a single detected defect; live observers (TUI,
# console) surface it as it is found, mirroring EXPLORATION_PROGRESS_EVENT.
DEFECT_DETECTED_EVENT: str = "exploration.defect"

# Tool the vision detector asks the model to call with the defects it sees.
DETECT_DEFECTS_TOOL: str = "detect_defects"

# Defect signals a single screenshot can evidence; constrains the vision tool's
# enum and the parser so the model cannot emit a runtime-only signal (dead tap,
# crash, blank capture) that only the inline detector can observe.
VISION_DEFECT_SIGNALS: tuple[DefectSignal, ...] = (
    DefectSignal.OVERLAP_CLIPPING,
    DefectSignal.CONTRAST,
    DefectSignal.BROKEN_IMAGE,
    DefectSignal.LOREM_IPSUM,
    DefectSignal.PLACEHOLDER_TEXT,
    DefectSignal.TODO_TEXT,
    DefectSignal.UNTRANSLATED_STRING,
    DefectSignal.EMPTY_STATE,
    DefectSignal.ERROR_DIALOG,
    DefectSignal.INFINITE_SPINNER,
    DefectSignal.SPELLING,
)
