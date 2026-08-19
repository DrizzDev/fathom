from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SealedModel(BaseModel):
    """
    Base model that is immutable and closed to unknown fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class ThresholdConfiguration(BaseModel):
    """
    Generic threshold values for bounded adaptive policies.
    """

    model_config = ConfigDict(extra="forbid")

    failures: int = Field(
        ge=1,
        description="Failure count threshold.",
    )
    slows: int = Field(
        ge=1,
        description="Slow-success count threshold.",
    )
    latency: float = Field(
        gt=0.0,
        description="Latency threshold.",
    )
    recovery: int = Field(
        ge=1,
        description="Healthy success count threshold.",
    )
