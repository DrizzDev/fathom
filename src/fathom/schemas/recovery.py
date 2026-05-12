from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class RecoveryPolicy(BaseModel):
    """
    Configuration for the stuck-loop recovery coordinator.
    Ships as part of ``InteractionConfiguration`` so each run can independently tune or disable recovery without touching core code.
    """

    enabled: bool = Field(
        default=False, description="Master kill switch for the recovery coordinator"
    )

    strategies: List[str] = Field(
        default_factory=lambda: ["replan"],
        description="Strategy names in priority order; empty list disables recovery",
    )
    verify_threshold: int = Field(
        default=3, ge=1, description="Consecutive VERIFY rejections per sub-goal before dispatch"
    )
    plan_threshold: int = Field(
        default=3,
        ge=1,
        description="Consecutive ACTION_BLOCKED emissions per sub-goal before dispatch",
    )
    recent_window: int = Field(
        default=10, ge=1, description="Most-recent action descriptors handed to strategies"
    )

    model_config = ConfigDict(frozen=True)
