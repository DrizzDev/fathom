from __future__ import annotations

from typing import Tuple

from fathom.constants.tools import BASE_TOOLS, ToolName
from fathom.core.agent.tools.policy import ToolPolicy
from fathom.core.exceptions import InvariantViolation
from fathom.schemas.tools import AllowedTools, ToolPolicyContext


class ToolScope:
    """
    Assembles the per-turn allowed tool set by composing :class:`ToolPolicy` rules.
    """

    def __init__(self, *, policies: Tuple[ToolPolicy, ...]) -> None:
        """
        Bind the scope to an explicit, ordered tuple of policies.
        """

        self.__policies = policies

    def compute(self, *, context: ToolPolicyContext) -> AllowedTools:
        """
        Resolve the allowed tool set by applying every bound policy to the context.
        """

        names: set[ToolName] = set(BASE_TOOLS)
        for policy in self.__policies:
            if policy.applies(context=context):
                names.add(policy.tool)

        self.__assert_liveness(names=names)
        return AllowedTools(names=frozenset(names))

    @staticmethod
    def __assert_liveness(*, names: set[ToolName]) -> None:
        """
        Guarantee the resolved tool set always carries the action tool the model needs.
        """

        if ToolName.EXECUTE_UI not in names:
            raise InvariantViolation(
                "ToolScope produced a tool set without EXECUTE_UI; liveness invariant broken."
            )
