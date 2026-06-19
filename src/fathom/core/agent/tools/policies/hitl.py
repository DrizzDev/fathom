from __future__ import annotations

from fathom.constants.tools import ToolName
from fathom.core.agent.tools.policy import ToolPolicy
from fathom.schemas.tools import ToolPolicyContext


class HitlToolPolicy(ToolPolicy):
    """
    Exposes a tool only when the runtime declares HITL availability.
    """

    def __init__(self, *, tool: ToolName) -> None:
        """
        Bind the policy to the tool it gates.
        """

        self.__tool = tool

    @property
    def tool(self) -> ToolName:
        """
        Tool this policy gates.
        """

        return self.__tool

    def applies(self, *, context: ToolPolicyContext) -> bool:
        """
        Allow the tool when HITL is enabled in the runtime capabilities.
        """

        return context.capabilities.hitl.enabled
