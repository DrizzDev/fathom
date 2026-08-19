from __future__ import annotations

from enum import StrEnum


class SuccessKind(StrEnum):
    """
    Discriminates how a sub-goal's success is defined and proven.
    """

    COMMAND = "COMMAND"
    CAPTURE = "CAPTURE"
    OBSERVED = "OBSERVED"


class CaptureNameProvenance(StrEnum):
    """
    Records whether a capture variable name was supplied by the user or proposed by the model.
    """

    USER = "USER"
    MODEL = "MODEL"
