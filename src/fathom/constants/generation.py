from __future__ import annotations

from enum import StrEnum
from typing import Final

BASELINE_SCRIPT_FILENAME: Final[str] = "script.baseline.txt"
BASELINE_METADATA_FILENAME: Final[str] = "script.baseline.meta.json"
COMPLETION_ASSERTIONS_FILENAME: Final[str] = "completion.assertions.json"


class ScriptArtifactScope(StrEnum):
    """
    Filename scopes for script artifacts that are not tied to an app package.
    """

    EXECUTION = "execution"


class ScriptSource(StrEnum):
    """
    Which generation path produced a script artifact.
    """

    QUALITY = "QUALITY"
    BASELINE = "BASELINE"


class ScriptStatus(StrEnum):
    """
    Whether a script artifact was produced or generation failed.
    """

    FAILED = "FAILED"
    GENERATED = "GENERATED"


class ScriptArtifactMode(StrEnum):
    """
    Controls whether script generation writes only production artifacts or also debug artifacts.
    """

    NORMAL = "NORMAL"
    DEBUG = "DEBUG"


class SkipReason(StrEnum):
    """
    Why the deterministic projector dropped an evidence step from the baseline flow.
    """

    FAILED = "FAILED"
    RECOVERY = "RECOVERY"
    UNSUPPORTED = "UNSUPPORTED"

    MISSING_TEXT = "MISSING_TEXT"
    MISSING_TARGET = "MISSING_TARGET"
    MISSING_CAPTURE = "MISSING_CAPTURE"
    MISSING_WAIT_SUBJECT = "MISSING_WAIT_SUBJECT"
