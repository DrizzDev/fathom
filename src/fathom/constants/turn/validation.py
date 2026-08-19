from __future__ import annotations

from enum import StrEnum


class ValidationSource(StrEnum):
    """
    Door a validation assertion entered through.
    """

    GOAL = "GOAL"
    STATE = "STATE"
    COMMAND = "COMMAND"
