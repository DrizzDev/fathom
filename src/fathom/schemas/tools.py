from __future__ import annotations

from typing import FrozenSet, Iterable, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

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
    def __coerce_modes(cls, value: object) -> FrozenSet[TurnMode]:
        """
        Accept any iterable of :class:`TurnMode` from callers and normalize to a frozen set.
        """

        if value is None:
            return frozenset()

        if isinstance(value, TurnMode):
            return frozenset({value})

        if isinstance(value, frozenset):
            return value

        return frozenset(cast("Iterable[TurnMode]", value))


class ToolScopeMatrixExpansion(BaseModel):
    """
    One boot-time tool-scope expansion for observability.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    modes: FrozenSet[TurnMode] = Field(description="Active mode flags for this expansion.")
    hitl: bool = Field(description="Whether HITL is available for this expansion.")
    tools_allowed: FrozenSet[ToolName] = Field(description="Tools exposed for this expansion.")

    @field_serializer("modes")
    def __serialize_modes(self, value: FrozenSet[TurnMode]) -> Sequence[str]:
        """
        Serialize mode flags in deterministic order.
        """

        return sorted(mode.value for mode in value)

    @field_serializer("tools_allowed")
    def __serialize_tools(self, value: FrozenSet[ToolName]) -> Sequence[str]:
        """
        Serialize tool names in deterministic order.
        """

        return sorted(name.value for name in value)
