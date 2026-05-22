from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.command import CommandScopeKind
from fathom.schemas.actions import Bounds, ExecutionRegion


class CommandAnchor(BaseModel):
    """
    Stable command anchor resolved from planner output.
    """

    model_config = ConfigDict(frozen=True)

    manifest_label_id: Optional[str] = Field(
        default=None,
        description="Resolved manifest label identifier when present.",
    )
    observation_region_id: Optional[str] = Field(
        default=None,
        description="Resolved observation-only region identifier when present.",
    )
    target: Optional[str] = Field(default=None, description="Human-readable target phrase when present.")
    bounds: Optional[Bounds] = Field(default=None, description="Planner-provided target bounds when present.")


class CommandScope(BaseModel):
    """
    Resolved execution scope for one command.
    """

    model_config = ConfigDict(frozen=True)

    identifier: str = Field(description="Stable scope identifier within the current screen.")
    kind: CommandScopeKind = Field(description="Coarse scope category.")
    bounds: Bounds = Field(description="Capture-space bounds of the resolved scope.")
    region: ExecutionRegion = Field(description="Logical execution region of the resolved scope.")
    axis: str = Field(description="Primary movement axis allowed inside the scope.")
    confidence: float = Field(ge=0.0, le=1.0, description="Resolution confidence in [0, 1].")


class CommandPolicy(BaseModel):
    """
    Shared bounded retry policy for scoped commands.
    """

    model_config = ConfigDict(frozen=True)

    attempts: int = Field(ge=1, description="Maximum attempts allowed for one supervised command run.")
    budget: int = Field(ge=1, description="Maximum wall time allowed for one supervised command run.")
