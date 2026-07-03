from __future__ import annotations

from typing import Dict, List, Optional, assert_never

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
from fathom.interfaces.renderer import Renderer as RenderPort
from fathom.schemas.flow import (
    BackNode,
    BranchNode,
    Check,
    CheckNode,
    ClearNode,
    Flow,
    FlowNode,
    KillNode,
    LaunchNode,
    LeafNode,
    LocationNode,
    MapNode,
    MinimizeNode,
    ScrollNode,
    ScrollUntilNode,
    Selector,
    StoreNode,
    TapNode,
    TypeNode,
    WaitNode,
)


class Renderer(RenderPort):
    """
    Renders a target-neutral flow into idiomatic Drizz script text.
    """

    __quoting = Quoting()

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

    def render(self, *, flow: Flow) -> str:
        """
        Render the flow into idiomatic Drizz script text.
        """

        out: List[str] = []

        for node in flow.nodes:
            out.extend(self.__render(node=node, indent=0))

        return "\n".join(out) + "\n"

    def __render(self, *, node: FlowNode, indent: int) -> List[str]:
        """
        Render a single node, recursing into branch bodies with indentation.
        """

        prefix = str(Syntax.INDENT) * indent

        if isinstance(node, BranchNode):
            return self.__branch(node=node, indent=indent)

        if isinstance(node, CheckNode) and self.__mixed(node=node):
            return [
                f"{prefix}{Keyword.VALIDATE} {self.__assertion(check=check)}"
                for check in node.checks
            ]

        return [f"{prefix}{self.__line(node=node)}"]

    def __mixed(self, *, node: CheckNode) -> bool:
        """
        Report whether a check groups assertions of more than one state.
        """

        return len({check.kind for check in node.checks}) > 1

    def __branch(self, *, node: BranchNode, indent: int) -> List[str]:
        """
        Render an IF branch with its indented body.
        """

        prefix = str(Syntax.INDENT) * indent
        out = [f"{prefix}{Keyword.IF} {node.guard.condition}", f"{prefix}{Syntax.BRACE_OPEN}"]

        for child in node.body:
            out.extend(self.__render(node=child, indent=indent + 1))

        out.append(f"{prefix}{Syntax.BRACE_CLOSE}")

        return out

    def __line(self, *, node: LeafNode) -> str:
        """
        Render a single leaf node to its Drizz command line.
        """

        if isinstance(node, LaunchNode):
            return f"{Keyword.OPEN_APP}{Syntax.OPEN_APP_SEPARATOR}{node.package}"

        if isinstance(node, TapNode):
            return f"{Keyword.TAP} {Phrase.ON} {self.__selector(selector=node.selector)}"

        if isinstance(node, TypeNode):
            return self.__type(node=node)

        if isinstance(node, ScrollNode):
            return self.__scroll(node=node)

        if isinstance(node, ScrollUntilNode):
            return self.__scroll_until(node=node)

        if isinstance(node, WaitNode):
            return self.__wait(node=node)

        if isinstance(node, BackNode):
            return str(Keyword.PRESS_DEVICE_BACK_BUTTON)

        if isinstance(node, KillNode):
            return str(Keyword.KILL_APP)

        if isinstance(node, ClearNode):
            return str(Keyword.CLEAR_APP)

        if isinstance(node, MinimizeNode):
            return str(Keyword.MINIMISE_APP)

        if isinstance(node, LocationNode):
            return f"{Keyword.SET_GPS}(latitude={node.latitude}, longitude={node.longitude})"

        if isinstance(node, StoreNode):
            return f"{Keyword.STORE} {self.__quoting.conditional(value=node.value)} {Phrase.AS} {node.name}"

        if isinstance(node, MapNode):
            return f"{Keyword.MAP_ACTION} {Keyword.TAP} {Phrase.ON} {self.__selector(selector=node.selector)}"

        if isinstance(node, CheckNode):
            return self.__check(node=node)

        assert_never(node)

    def __selector(self, *, selector: Selector) -> str:
        """
        Render a selector as an unquoted natural-language Drizz target phrase.
        """

        core = self.__positioned(text=selector.text, position=selector.position)

        if selector.container:
            core = f"{core} {Phrase.UNDER} {selector.container}"

        return self.__quoting.conditional(value=core)

    def __positioned(self, *, text: str, position: Optional[str]) -> str:
        """
        Prepend the ordinal, skipping it when the text already leads with that ordinal.
        """

        if not position:
            return text

        lowered = text.lower()
        ordinal = position.lower()

        if lowered == ordinal or lowered.startswith(f"{ordinal} "):
            return f"{Phrase.THE} {text}"

        if lowered == f"{Phrase.THE} {ordinal}" or lowered.startswith(f"{Phrase.THE} {ordinal} "):
            return text

        return f"{Phrase.THE} {position} {text}"

    def __type(self, *, node: TypeNode) -> str:
        """
        Render a typed literal value into an unquoted field.
        """

        return (
            f"{Keyword.TYPE} {self.__quoting.wrap(value=node.text)} "
            f"{Phrase.INTO} {self.__selector(selector=node.field)}"
        )

    def __scroll(self, *, node: ScrollNode) -> str:
        """
        Render a scroll by direction, by a percentage, or inside a container.
        """

        direction = self.__DIRECTIONS[node.direction]

        if node.percentage is not None:
            return f"{Keyword.SCROLL} {direction} {Phrase.BY} {node.percentage}%"

        if node.container:
            return f"{Keyword.SCROLL} {direction} {Phrase.INSIDE} {node.container}"

        return f"{Keyword.SCROLL} {direction}"

    def __scroll_until(self, *, node: ScrollUntilNode) -> str:
        """
        Render a scroll until a quoted target, optionally scoped to a container.
        """

        core = f"{Keyword.SCROLL} {self.__DIRECTIONS[node.direction]}"

        if node.container:
            core = f"{core} {Phrase.INSIDE} {node.container}"

        return f"{core} {Phrase.UNTIL} {self.__quoting.wrap(value=node.target)}"

    def __wait(self, *, node: WaitNode) -> str:
        """
        Render a wait by duration, until a quoted subject, or both combined.
        """

        if node.duration is not None and node.subject is not None:
            return f"{Keyword.WAIT} {node.duration} {Phrase.SECONDS} {Phrase.FOR} {node.subject}"

        if node.duration is not None:
            return f"{Keyword.WAIT} {Phrase.FOR} {node.duration} {Phrase.SECONDS}"

        if node.subject is None:
            raise InvariantViolation("Wait node has neither duration nor subject.")

        return f"{Keyword.WAIT} {Phrase.UNTIL} {self.__quoting.wrap(value=node.subject)}"

    def __check(self, *, node: CheckNode) -> str:
        """
        Render a validation as one assertion or a grouped, state-once numbered list.
        """

        if len(node.checks) == 1:
            return f"{Keyword.VALIDATE} {self.__assertion(check=node.checks[0])}"

        word = self.__GROUP_STATES[node.checks[0].kind]
        numbered = " ".join(
            f"{index}. {self.__quoting.wrap(value=check.subject)}"
            for index, check in enumerate(node.checks, start=1)
        )
        return f"{Keyword.VALIDATE} {Phrase.FOLLOWING} {word}: {numbered}"

    def __assertion(self, *, check: Check) -> str:
        """
        Render one assertion as 'subject state'.
        """

        return f"{self.__quoting.conditional(value=check.subject)} {self.__STATES[check.kind]}"
