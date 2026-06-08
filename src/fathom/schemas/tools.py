from __future__ import annotations

from typing import Any, FrozenSet

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fathom.constants.tools import ToolName, TurnMode
from fathom.schemas.capabilities import RuntimeCapabilities


class AllowedTools(BaseModel):
    """
    Tools the language model may invoke for a single analysis turn.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    names: FrozenSet[ToolName] = Field(description="Permitted tool identifiers.")

    def contains(self, *, name: ToolName) -> bool:
        """
        Return whether the tool is permitted.
        """

        return name in self.names


class ToolPolicyContext(BaseModel):
    """
    Per-turn signals consumed by every tool-inclusion policy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    capabilities: RuntimeCapabilities = Field(description="Runtime capabilities for this turn.")
    modes: FrozenSet[TurnMode] = Field(
        default_factory=frozenset,
        description="Active per-turn mode flags; empty means only base tools are exposed.",
    )

    @field_validator("modes", mode="before")
    @classmethod
    def __coerce_modes(cls, value: Any) -> FrozenSet[TurnMode]:
        """
        Accept any iterable of :class:`TurnMode` from callers and normalise to a frozen set.
        """

        if isinstance(value, frozenset):
            return value
        return frozenset(value)
