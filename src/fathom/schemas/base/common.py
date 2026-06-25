from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
