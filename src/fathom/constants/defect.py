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


# Case-insensitive markers of unfinished copy, mapped to the signal each raises.
# The content detector matches these on word boundaries to avoid false hits.
PLACEHOLDER_SIGNALS: dict[str, DefectSignal] = {
    "lorem ipsum": DefectSignal.LOREM_IPSUM,
    "lorem": DefectSignal.LOREM_IPSUM,
    "todo": DefectSignal.TODO_TEXT,
    "fixme": DefectSignal.TODO_TEXT,
    "placeholder": DefectSignal.PLACEHOLDER_TEXT,
    "dummy text": DefectSignal.PLACEHOLDER_TEXT,
    "sample text": DefectSignal.PLACEHOLDER_TEXT,
}
