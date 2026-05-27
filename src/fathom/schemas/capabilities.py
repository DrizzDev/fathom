from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HITLCapability(BaseModel):
    """Capability flags governing human-in-the-loop interactions."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(description="Whether a human operator is available.")


class RuntimeCapabilities(BaseModel):
    """Runtime capability flags injected at composition root."""

    model_config = ConfigDict(frozen=True)

    hitl: HITLCapability = Field(description="Human-in-the-loop capability flags.")
