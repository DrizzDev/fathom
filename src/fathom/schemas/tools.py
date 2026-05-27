from __future__ import annotations

from typing import FrozenSet

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.tools import ToolName


class AllowedTools(BaseModel):
    """
    Tools the language model may invoke for a single analysis turn.
    """

    model_config = ConfigDict(frozen=True)

    names: FrozenSet[ToolName] = Field(description="Permitted tool identifiers.")

    def contains(self, *, name: ToolName) -> bool:
        """
        Return whether the tool is permitted.
        """

        return name in self.names
