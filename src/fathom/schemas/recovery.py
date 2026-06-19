from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RecoveryDecisionKind(StrEnum):
    """
    Autonomous recovery control directive.
    """

    ALLOW = "allow"
    REPLAN = "replan"


class RecoveryReason(StrEnum):
    """
    Stable reason for an autonomous recovery decision.
    """

    SAFE = "safe"
    ACTIVE_NO_PROGRESS = "active_no_progress"


class RecoveryDecision(BaseModel):
    """
    Typed decision returned by autonomous recovery policy.
    """

    model_config = ConfigDict(frozen=True)

    kind: RecoveryDecisionKind = Field(description="Directive for autonomous recovery.")
    reason: RecoveryReason = Field(description="Stable explanation for the directive.")

    @property
    def allowed(self) -> bool:
        """
        Return whether blind mechanical recovery may run.
        """

        return self.kind is RecoveryDecisionKind.ALLOW
