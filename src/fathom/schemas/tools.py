from __future__ import annotations

from typing import FrozenSet, Iterable, Optional, Sequence, Tuple, cast

from pydantic import Field, JsonValue, field_serializer, field_validator

from fathom.constants import ActionType
from fathom.constants.tools import DiagnosticSeverity, StateNamespace, ToolName, TurnMode
from fathom.schemas.base import SealedModel
from fathom.schemas.capabilities import RuntimeCapabilities
from fathom.schemas.gemini_tools import ExecuteAction


class AllowedTools(SealedModel):
    """
    Tools the language model may invoke for a single analysis turn.
    """

    names: FrozenSet[ToolName] = Field(description="Permitted tool identifiers.")

    def contains(self, *, name: ToolName) -> bool:
        """
        Return whether the tool is permitted.
        """

        return name in self.names


class ToolCommand(SealedModel):
    """
    Pre-action Fathom command requested by a model tool.
    """

    action_type: ActionType = Field(description="Requested Fathom command.")
    payload: ExecuteAction = Field(description="Strict execute_ui payload.")


class AcceptedCommand(SealedModel):
    """
    Tool command accepted by the command catalog gate.
    """

    action_type: ActionType = Field(description="Accepted Fathom command.")
    payload: ExecuteAction = Field(description="Payload accepted by the command gate.")


class StateUpdate(SealedModel):
    """
    Runtime state mutation requested by a model tool.
    """

    namespace: StateNamespace = Field(description="Runtime state namespace to update.")
    key: str = Field(min_length=1, description="State key.")
    value: str = Field(min_length=1, description="State value, never logged.")


class ToolData(SealedModel):
    """
    Data returned by a model tool for later reasoning.
    """

    name: str = Field(min_length=1, description="Data name.")
    value: JsonValue = Field(description="JSON-compatible data value.")


class ToolArtifact(SealedModel):
    """
    Artifact produced or referenced by a model tool.
    """

    name: str = Field(min_length=1, description="Artifact name.")
    reference: str = Field(min_length=1, description="Artifact reference or id.")


class ToolDiagnostic(SealedModel):
    """
    Diagnostic returned by a model tool.
    """

    severity: DiagnosticSeverity = Field(description="Diagnostic severity.")
    message: str = Field(min_length=1, description="Diagnostic message.")
    code: Optional[str] = Field(default=None, min_length=1, description="Stable diagnostic code.")


class ToolResponse(SealedModel):
    """
    Parsed model-tool response for one planner turn.
    """

    command: Optional[ToolCommand] = Field(default=None, description="Requested command.")
    updates: Tuple[StateUpdate, ...] = Field(
        default_factory=tuple, description="Runtime updates requested by tools."
    )
    data: Tuple[ToolData, ...] = Field(default_factory=tuple, description="Data returned by tools.")
    artifacts: Tuple[ToolArtifact, ...] = Field(
        default_factory=tuple, description="Artifacts returned by tools."
    )
    diagnostics: Tuple[ToolDiagnostic, ...] = Field(
        default_factory=tuple, description="Diagnostics returned by tools."
    )

    @property
    def has_non_command_parts(self) -> bool:
        """
        Return whether the response contains routable non-command parts.
        """

        return bool(self.updates or self.data or self.artifacts or self.diagnostics)


class ToolPolicyContext(SealedModel):
    """
    Per-turn signals consumed by every tool-inclusion policy.
    """

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


class ToolScopeMatrixExpansion(SealedModel):
    """
    One boot-time tool-scope expansion for observability.
    """

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
