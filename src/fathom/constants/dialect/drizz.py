from __future__ import annotations

from enum import StrEnum


class Keyword(StrEnum):
    """
    Canonical Drizz command keywords (literal on-device syntax).
    """

    TAP = "Tap"
    TYPE = "Type"
    SCROLL = "Scroll"
    VALIDATE = "Validate"

    IF = "IF"
    WAIT = "Wait"
    STORE = "Store"
    SET_GPS = "SET_GPS"
    MAP_ACTION = "MAP_ACTION"

    OPEN_APP = "OPEN_APP"
    KILL_APP = "KILL_APP"
    CLEAR_APP = "CLEAR_APP"
    MINIMISE_APP = "MINIMISE_APP"
    PRESS_DEVICE_BACK_BUTTON = "PRESS_DEVICE_BACK_BUTTON"


class Direction(StrEnum):
    """
    Rendered scroll direction tokens (literal Drizz syntax).
    """

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class State(StrEnum):
    """
    Rendered single-assertion validation-state phrases (literal Drizz syntax).
    """

    VISIBLE = "is visible"
    PRESENT = "is present"
    ENABLED = "is enabled"
    DISABLED = "is disabled"


class GroupState(StrEnum):
    """
    Bare state words used in grouped validations (literal Drizz syntax).
    """

    VISIBLE = "visible"
    PRESENT = "present"
    ENABLED = "enabled"
    DISABLED = "disabled"


class Syntax(StrEnum):
    """
    Literal Drizz syntax tokens.
    """

    OPEN_APP_SEPARATOR = ": "
    BRACE_OPEN = "{"
    BRACE_CLOSE = "}"
    INDENT = "    "


class Quote(StrEnum):
    """
    Drizz string delimiters, in canonical preference order (double preferred).
    """

    DOUBLE = '"'
    SINGLE = "'"
    BACKTICK = "`"


class Phrase(StrEnum):
    """
    Connector phrases used when rendering Drizz commands (literal syntax).
    """

    ON = "on"
    IN = "in"
    INTO = "into"
    BY = "by"
    INSIDE = "inside"
    FOR = "for"
    AS = "as"
    THE = "the"
    UNTIL = "until"
    UNDER = "under"
    SECONDS = "seconds"
    FOLLOWING = "the following are"
