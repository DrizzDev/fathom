from __future__ import annotations

from typing import Dict, List, assert_never

from fathom.constants.dialect.drizz import (
    Direction,
    GroupState,
    Keyword,
    Phrase,
    State,
    Syntax,
)
from fathom.constants.flow import CheckKind, ScrollDirection
from fathom.core.dialect.drizz.quote import Quoting
from fathom.core.exceptions import InvariantViolation
from fathom.schemas.dialect.drizz.command import (
    BackCommand,
    ClearCommand,
    DrizzCommand,
    IfCommand,
    KillCommand,
    LeafCommand,
    MapActionCommand,
    MinimizeCommand,
    OpenAppCommand,
    ScrollCommand,
    SetGpsCommand,
    StoreCommand,
    TapCommand,
    TypeCommand,
    ValidateCommand,
    WaitCommand,
)
from fathom.schemas.dialect.drizz.script import DrizzScript
from fathom.schemas.dialect.drizz.target import Assertion, Target


class CanonicalPrinter:
    """
    Prints a typed Drizz AST back to canonical script text for round-trip checking.
    """

    __STATES: Dict[CheckKind, State] = {
        CheckKind.VISIBLE: State.VISIBLE,
        CheckKind.PRESENT: State.PRESENT,
        CheckKind.ENABLED: State.ENABLED,
        CheckKind.DISABLED: State.DISABLED,
    }
    __GROUP_STATES: Dict[CheckKind, GroupState] = {
        CheckKind.VISIBLE: GroupState.VISIBLE,
        CheckKind.PRESENT: GroupState.PRESENT,
        CheckKind.ENABLED: GroupState.ENABLED,
        CheckKind.DISABLED: GroupState.DISABLED,
    }
    __DIRECTIONS: Dict[ScrollDirection, Direction] = {
        ScrollDirection.UP: Direction.UP,
        ScrollDirection.DOWN: Direction.DOWN,
        ScrollDirection.LEFT: Direction.LEFT,
        ScrollDirection.RIGHT: Direction.RIGHT,
    }
    __quoting = Quoting()

    def emit(self, *, script: DrizzScript) -> str:
        """
        Print the script into canonical Drizz text.
        """

        out: List[str] = []

        for command in script.commands:
            out.extend(self.__command(command=command, indent=0))

        return "\n".join(out) + "\n"

    def __command(self, *, command: DrizzCommand, indent: int) -> List[str]:
        """
        Print a single command, recursing into IF bodies with indentation.
        """

        prefix = str(Syntax.INDENT) * indent

        if isinstance(command, IfCommand):
            return self.__branch(command=command, indent=indent)

        return [f"{prefix}{self.__line(command=command)}"]

    def __branch(self, *, command: IfCommand, indent: int) -> List[str]:
        """
        Print an IF block with its indented leaf body.
        """

        prefix = str(Syntax.INDENT) * indent
        out = [
            f"{prefix}{Keyword.IF} {self.__line_text(value=command.condition)}",
            f"{prefix}{Syntax.BRACE_OPEN}",
        ]

        for child in command.body:
            out.extend(self.__command(command=child, indent=indent + 1))

        out.append(f"{prefix}{Syntax.BRACE_CLOSE}")

        return out

    @staticmethod
    def __line_text(*, value: str) -> str:
        """
        Collapse line-breaking whitespace so header text stays inside one Drizz statement.
        """

        return " ".join(value.split())

    def __line(self, *, command: LeafCommand) -> str:
        """
        Print a single leaf command to its canonical Drizz line.
        """

        if isinstance(command, OpenAppCommand):
            return f"{Keyword.OPEN_APP}{Syntax.OPEN_APP_SEPARATOR}{command.package}"

        if isinstance(command, TapCommand):
            return f"{Keyword.TAP} {Phrase.ON} {self.__target(target=command.target)}"

        if isinstance(command, TypeCommand):
            return (
                f"{Keyword.TYPE} {self.__quoting.wrap(value=command.value)} "
                f"{Phrase.INTO} {self.__target(target=command.field)}"
            )

        if isinstance(command, ScrollCommand):
            return self.__scroll(command=command)

        if isinstance(command, WaitCommand):
            return self.__wait(command=command)

        if isinstance(command, BackCommand):
            return str(Keyword.PRESS_DEVICE_BACK_BUTTON)

        if isinstance(command, KillCommand):
            return str(Keyword.KILL_APP)

        if isinstance(command, ClearCommand):
            return str(Keyword.CLEAR_APP)

        if isinstance(command, MinimizeCommand):
            return str(Keyword.MINIMISE_APP)

        if isinstance(command, SetGpsCommand):
            return f"{Keyword.SET_GPS}(latitude={command.latitude}, longitude={command.longitude})"

        if isinstance(command, StoreCommand):
            return (
                f"{Keyword.STORE} {self.__quoting.conditional(value=command.value)} "
                f"{Phrase.AS} {command.name}"
            )

        if isinstance(command, MapActionCommand):
            return (
                f"{Keyword.MAP_ACTION} {Keyword.TAP} {Phrase.ON} "
                f"{self.__target(target=command.target)}"
            )

        if isinstance(command, ValidateCommand):
            return self.__validate(command=command)

        assert_never(command)

    def __target(self, *, target: Target) -> str:
        """
        Print a target as a quoted or positional phrase with an optional container.
        """

        core = target.text

        if target.position:
            core = f"{Phrase.THE} {target.position} {target.text}"

        if target.container:
            core = f"{core} {Phrase.UNDER} {target.container}"

        return self.__quoting.conditional(value=core)

    def __scroll(self, *, command: ScrollCommand) -> str:
        """
        Print a directional scroll, optionally bounded by a quoted target.
        """

        direction = self.__DIRECTIONS[command.direction]
        core = f"{Keyword.SCROLL} {direction}"

        if command.container:
            core = f"{core} {Phrase.INSIDE} {command.container}"

        if command.until is not None:
            return f"{core} {Phrase.UNTIL} {self.__quoting.wrap(value=command.until.text)}"

        if command.percentage is not None:
            return f"{core} {Phrase.BY} {command.percentage}%"

        return core

    def __wait(self, *, command: WaitCommand) -> str:
        """
        Print a wait bounded by a duration, a quoted subject, or both combined.
        """

        if command.duration is not None and command.subject is not None:
            return (
                f"{Keyword.WAIT} {command.duration} {Phrase.SECONDS} {Phrase.FOR} {command.subject}"
            )

        if command.duration is not None:
            return f"{Keyword.WAIT} {Phrase.FOR} {command.duration} {Phrase.SECONDS}"

        if command.subject is None:
            raise InvariantViolation("Wait command has neither duration nor subject.")

        return f"{Keyword.WAIT} {Phrase.UNTIL} {self.__quoting.wrap(value=command.subject)}"

    def __validate(self, *, command: ValidateCommand) -> str:
        """
        Print a validation as one assertion or a numbered list.
        """

        if len(command.assertions) == 1:
            return f"{Keyword.VALIDATE} {self.__assertion(assertion=command.assertions[0])}"

        word = self.__GROUP_STATES[command.assertions[0].state]
        numbered = " ".join(
            f"{index}. {self.__quoting.wrap(value=assertion.subject)}"
            for index, assertion in enumerate(command.assertions, start=1)
        )
        return f"{Keyword.VALIDATE} {Phrase.FOLLOWING} {word}: {numbered}"

    def __assertion(self, *, assertion: Assertion) -> str:
        """
        Print one assertion as 'subject state'.
        """

        return f"{self.__quoting.conditional(value=assertion.subject)} {self.__STATES[assertion.state]}"
