from __future__ import annotations

from typing import FrozenSet, Optional, Tuple

from fathom.constants.tools import TurnMode
from fathom.core.agent.state import AgentState
from fathom.core.agent.tools import ToolScope
from fathom.schemas.planner import GoalRef, ToolScopeEvent
from fathom.schemas.subgoal import GoalState
from fathom.schemas.tools import AllowedTools, ToolPolicyContext


class ToolScopeResolver:
    """
    Resolves the turn's allowed tools from the goal context and reports the resolution as an event.
    """

    def __init__(self, *, tool_scope: ToolScope) -> None:
        """
        Bind the tool-scope policy the resolver consults.
        """

        self.__tool_scope = tool_scope

    def resolve(
        self, *, state: AgentState, current_sub_goal: Optional[GoalState]
    ) -> Tuple[AllowedTools, ToolScopeEvent]:
        """
        Return the allowed tools for the turn and the event describing the resolution.

        Base UI tools remain available during any active goal; VERIFY scope is reserved for the
        terminal, no-active-goal verification phase. Planner tactics are never inferred from a
        goal's success kind.
        """

        modes: FrozenSet[TurnMode] = (
            frozenset() if state.has_sub_goals() else frozenset({TurnMode.VERIFY})
        )
        allowed = self.__tool_scope.compute(
            context=ToolPolicyContext(capabilities=state.capabilities, modes=modes)
        )
        event = ToolScopeEvent(
            modes=tuple(sorted(modes, key=lambda mode: mode.value)),
            tools=tuple(sorted(allowed.names, key=lambda name: name.value)),
            goal=GoalRef(index=current_sub_goal.index) if current_sub_goal is not None else None,
        )
        return allowed, event
