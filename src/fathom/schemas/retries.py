from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.retries import (
    DEFAULT_PLANNER_RETRY_LIMIT,
    RetryBranch,
    RetryKind,
)


class RetryAttempt(BaseModel):
    """
    Diagnostic snapshot of one should_retry=True slot consumed against a budget.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: RetryKind = Field(description="Branch classification driving this retry.")
    branch: RetryBranch = Field(description="Source identifier the planner stamped on the return.")

    action: Optional[str] = Field(
        default=None,
        description="Action.to_description() of the rejected action; None when not applicable.",
    )
    screen: Optional[str] = Field(
        default=None,
        description="Visual hash of the screen the rejection happened on; None if unavailable.",
    )
    activity: Optional[str] = Field(
        default=None,
        description="Activity identifier the rejection happened on; None if unavailable.",
    )


class RetryCounter(BaseModel):
    """
    Frozen count and cap pair shared by every retry-budget kind.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int = Field(
        ge=0,
        default=0,
        description="Consecutive should_retry=True slots since the last reset.",
    )
    cap: int = Field(
        ge=1,
        description="Ceiling for ``count``; the budget is exhausted when count reaches cap.",
    )

    @property
    def exhausted(self) -> bool:
        """
        Return whether the budget has been exhausted.
        """

        return self.count >= self.cap


class RetryState(BaseModel):
    """
    Aggregate of per-kind retry counters managed by AgentState.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    planner: RetryCounter = Field(
        description="Planner should_retry=True budget for the current step."
    )


class RetryLimits(BaseModel):
    """
    Per-kind retry caps wired into IntentConfiguration and resolved into RetryState.
    """

    model_config = ConfigDict(extra="forbid")

    planner: int = Field(
        ge=1,
        default=DEFAULT_PLANNER_RETRY_LIMIT,
        description="Per-step cap on consecutive planner should_retry=True returns since the last successful dispatch.",
    )
