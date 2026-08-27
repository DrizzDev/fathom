"""Define immutable model-role manifest records."""

from __future__ import annotations

from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.llm import ModelProvider
from fathom.constants.manifest import MANIFEST_VERSION, ModelRole, RoleBinding
from fathom.schemas.llm import ModelIdentity


class RequestedModel(BaseModel):
    """
    Provider and model a role's configuration asked for, before composition resolved it.
    """

    model_config = ConfigDict(frozen=True)

    provider: Optional[ModelProvider] = Field(
        default=None,
        description="Provider the role's configuration declared, when it declares one.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model identifier the role's configuration asked for.",
    )


class RoleResolution(BaseModel):
    """
    How one model-consuming role resolved for a run.
    """

    model_config = ConfigDict(frozen=True)

    role: ModelRole = Field(description="The model-consuming role this entry describes.")
    enabled: bool = Field(description="Whether the role runs at all for this run.")
    binding: RoleBinding = Field(
        description="Whether the role owns its adapter, derives one, shares another role's, or has none.",
    )
    source: Optional[ModelRole] = Field(
        default=None,
        description="Role this one derives from or shares an adapter with.",
    )
    requested: Optional[RequestedModel] = Field(
        default=None,
        description="What configuration asked for; absent when the role declares no model of its own.",
    )
    resolved: Optional[ModelIdentity] = Field(
        default=None,
        description="What the instantiated adapter reports; absent when the role is disabled.",
    )
    note: Optional[str] = Field(
        default=None,
        description="Why requested and resolved disagree, when composition ignores the request.",
    )


class RuntimeManifest(BaseModel):
    """
    Every model-consuming role Fathom resolved for one run, with the step budget it runs under.
    """

    model_config = ConfigDict(frozen=True)

    version: int = Field(
        default=MANIFEST_VERSION,
        description="Manifest schema version, so a stored manifest outlives the code that wrote it.",
    )
    steps: int = Field(
        ge=0,
        description="Effective maximum steps for the run, after request and configuration defaults resolve.",
    )
    roles: Tuple[RoleResolution, ...] = Field(
        description="One entry per model-consuming role, enabled or not.",
    )

    def entry(self, *, role: ModelRole) -> Optional[RoleResolution]:
        """
        Return the resolution recorded for one role, or None when the role is absent.
        """

        return next((entry for entry in self.roles if entry.role == role), None)

    def providers(self) -> Tuple[ModelProvider, ...]:
        """
        Return the distinct providers every enabled role resolved to, in role order.
        """

        seen: list[ModelProvider] = []

        for entry in self.roles:
            if entry.resolved is not None and entry.resolved.provider not in seen:
                seen.append(entry.resolved.provider)

        return tuple(seen)
