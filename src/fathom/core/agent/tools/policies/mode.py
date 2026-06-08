from __future__ import annotations

from fathom.constants.tools import ToolName, TurnMode
from fathom.core.agent.tools.policy import ToolPolicy
from fathom.schemas.tools import ToolPolicyContext


class TurnModeToolPolicy(ToolPolicy):
    """
    Exposes a tool when its required :class:`TurnMode` flag is active for the turn.
    """

    def __init__(self, *, tool: ToolName, required_mode: TurnMode) -> None:
        """
        Bind the policy to its target tool and the mode flag that enables it.
        """

        self.__tool = tool
        self.__required_mode = required_mode

    @property
    def tool(self) -> ToolName:
        """
        Tool this policy gates.
        """

        return self.__tool

    def applies(self, *, context: ToolPolicyContext) -> bool:
        """
        Allow the tool when the required mode flag is present in the turn context.
        """

        return self.__required_mode in context.modes
