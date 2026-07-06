from __future__ import annotations

from typing import Dict, Tuple

from lark import Token, Transformer, v_args

from fathom.constants.dialect.drizz import Direction, GroupState, Keyword, State
from fathom.constants.flow import CheckKind, ScrollDirection
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


@v_args(inline=True)
class DrizzTransformer(Transformer[Token, object]):
    """
    Transforms a Lark parse tree of rendered Drizz into the typed command AST.
    """

    __ESCAPE = "\\"

    __DIRECTIONS: Dict[str, ScrollDirection] = {
        Direction.UP: ScrollDirection.UP,
        Direction.DOWN: ScrollDirection.DOWN,
        Direction.LEFT: ScrollDirection.LEFT,
        Direction.RIGHT: ScrollDirection.RIGHT,
    }
    __STATES: Dict[str, CheckKind] = {
        State.VISIBLE: CheckKind.VISIBLE,
        State.PRESENT: CheckKind.PRESENT,
        State.ENABLED: CheckKind.ENABLED,
        State.DISABLED: CheckKind.DISABLED,
    }
    __GROUP_STATES: Dict[str, CheckKind] = {
        GroupState.VISIBLE: CheckKind.VISIBLE,
        GroupState.PRESENT: CheckKind.PRESENT,
        GroupState.ENABLED: CheckKind.ENABLED,
        GroupState.DISABLED: CheckKind.DISABLED,
    }

    def start(self, *commands: DrizzCommand) -> DrizzScript:
        """
        Collect parsed commands into an ordered script.
        """

        return DrizzScript(commands=tuple(commands))

    def if_block(self, header: Token, *commands: LeafCommand) -> IfCommand:
        """
        Build a conditional block from its header line and leaf body.
        """

        condition = str(header).split(f"{Keyword.IF} ", 1)[1]
        return IfCommand(condition=condition, body=tuple(commands))

    def open_app(self, package: Token) -> OpenAppCommand:
        """
        Build a launch command from its package token.
        """

        return OpenAppCommand(package=str(package))

    def tap(self, target: Target) -> TapCommand:
        """
        Build a tap command from its target.
        """

        return TapCommand(target=target)

    def text(self, value: Token, field: Target) -> TypeCommand:
        """
        Build a type command from its quoted value and field target.
        """

        return TypeCommand(value=self.__unquote(token=value), field=field)

    def scroll_plain(self, direction: Token) -> ScrollCommand:
        """
        Build a directional scroll command.
        """

        return ScrollCommand(direction=self.__DIRECTIONS[str(direction)])

    def scroll_percent(self, direction: Token, percentage: Token) -> ScrollCommand:
        """
        Build a scroll command bounded by a percentage of the view.
        """

        return ScrollCommand(
            direction=self.__DIRECTIONS[str(direction)], percentage=int(percentage)
        )

    def scroll_inside(self, direction: Token, container: Target) -> ScrollCommand:
        """
        Build a scroll command scoped to a container.
        """

        return ScrollCommand(direction=self.__DIRECTIONS[str(direction)], container=container.text)

    def scroll_inside_until(
        self, direction: Token, container: Target, target: Token
    ) -> ScrollCommand:
        """
        Build a scroll command scoped to a container and bounded by a target.
        """

        return ScrollCommand(
            container=container.text,
            until=Target(text=self.__unquote(token=target)),
            direction=self.__DIRECTIONS[str(direction)],
        )

    def scroll_until(self, direction: Token, target: Token) -> ScrollCommand:
        """
        Build a scroll-until command from its direction and quoted target.
        """

        return ScrollCommand(
            until=Target(text=self.__unquote(token=target)),
            direction=self.__DIRECTIONS[str(direction)],
        )

    def wait_full(self, seconds: Token, subject: str) -> WaitCommand:
        """
        Build a wait command bounded by both a duration and a subject.
        """

        return WaitCommand(duration=int(seconds), subject=subject)

    def wait_duration(self, seconds: Token) -> WaitCommand:
        """
        Build a wait command bounded by a duration in seconds.
        """

        return WaitCommand(duration=int(seconds))

    def wait_subject(self, subject: Token) -> WaitCommand:
        """
        Build a wait command bounded by a quoted subject.
        """

        return WaitCommand(subject=self.__unquote(token=subject))

    def back(self) -> BackCommand:
        """
        Build a device-back command.
        """

        return BackCommand()

    def kill(self) -> KillCommand:
        """
        Build a kill-app command.
        """

        return KillCommand()

    def clear(self) -> ClearCommand:
        """
        Build a clear-app command.
        """

        return ClearCommand()

    def minimise(self, _keyword: Token) -> MinimizeCommand:
        """
        Build a minimise-app command.
        """

        _ = _keyword

        return MinimizeCommand()

    def set_gps(self, latitude: Token, longitude: Token) -> SetGpsCommand:
        """
        Build a GPS command from its latitude and longitude.
        """

        return SetGpsCommand(latitude=float(latitude), longitude=float(longitude))

    def store(self, capture: str, name: Token) -> StoreCommand:
        """
        Build a store command from its captured value and variable name.
        """

        return StoreCommand(value=capture, name=str(name))

    def map_action(self, target: Target) -> MapActionCommand:
        """
        Build a map-surface tap command.
        """

        return MapActionCommand(target=target)

    def validate_one(self, subject: str, state: Token) -> ValidateCommand:
        """
        Build a single-assertion validation.
        """

        return ValidateCommand(
            assertions=(Assertion(subject=subject, state=self.__STATES[str(state)]),)
        )

    def validate_many(
        self, _following: Token, group_state: Token, *subjects: str
    ) -> ValidateCommand:
        """
        Build a grouped validation sharing one state across quoted subjects.
        """

        _ = _following

        state = self.__GROUP_STATES[str(group_state)]
        return ValidateCommand(
            assertions=tuple(Assertion(subject=subject, state=state) for subject in subjects)
        )

    def numbered(self, _index: Token, value: Token) -> str:
        """
        Return the unquoted subject of one grouped validation item.
        """

        _ = _index

        return self.__unquote(token=value)

    def nl_target(self, *words: Token) -> Target:
        """
        Build a natural-language target from a word run or a single quoted string.
        """

        return Target(text=self.__phrase(words=words))

    def capture(self, *words: Token) -> str:
        """
        Resolve a stored subject from a word run or a single quoted string.
        """

        return self.__phrase(words=words)

    def subject(self, *words: Token) -> str:
        """
        Resolve a validation subject from a word run or a single quoted string.
        """

        return self.__phrase(words=words)

    def __phrase(self, *, words: Tuple[Token, ...]) -> str:
        """
        Unquote a single string token, else join the free-text words.
        """

        if len(words) == 1 and words[0].type == "STRING":
            return self.__unquote(token=words[0])

        return " ".join(str(word) for word in words)

    def __unquote(self, *, token: Token) -> str:
        """
        Strip the surrounding quote delimiter and decode Drizz string escapes.
        """

        return self.__unescape(value=str(token)[1:-1])

    def __unescape(self, *, value: str) -> str:
        """
        Decode escaped delimiters, backslashes, and line-breaking characters.
        """

        out = []
        escaping = False

        for character in value:
            if escaping:
                out.append(self.__escaped(character=character))
                escaping = False
                continue

            if character == self.__ESCAPE:
                escaping = True
                continue

            out.append(character)

        if escaping:
            out.append(self.__ESCAPE)

        return "".join(out)

    @classmethod
    def __escaped(cls, *, character: str) -> str:
        """
        Return the decoded value for one escaped character.
        """

        if character == "n":
            return "\n"

        if character == "r":
            return "\r"

        if character == "t":
            return "\t"

        return character
