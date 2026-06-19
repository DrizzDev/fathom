from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.constants.tools import ToolName
from fathom.schemas.tools import ToolPolicyContext


class ToolPolicy(ABC):
    """
    Decides whether one tool is exposed for the given turn context.
    """

    @property
    @abstractmethod
    def tool(self) -> ToolName:
        """
        Tool this policy gates.
        """

        raise NotImplementedError

    @abstractmethod
    def applies(self, *, context: ToolPolicyContext) -> bool:
        """
        Return True when the tool should be exposed this turn.
        """

        raise NotImplementedError
