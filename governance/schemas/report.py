from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field

from governance.constants import GovernanceMode
from governance.schemas.debt import DebtRecord
from governance.schemas.finding import Violation


class Report(BaseModel):
    """
    The reconciled result of a fitness run: findings and debt records partitioned by disposition.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    new: List[Violation] = Field(
        default_factory=list, description="Findings with no accepting debt record."
    )
    known: List[Violation] = Field(
        default_factory=list, description="Findings accepted one-to-one by a waivable debt record."
    )
    stale: List[DebtRecord] = Field(
        default_factory=list, description="Debt records with no matching finding; must be removed."
    )
    expiring: List[DebtRecord] = Field(
        default_factory=list, description="Debt records inside their expiry warning window."
    )
    expired: List[DebtRecord] = Field(
        default_factory=list, description="Debt records whose exception has lapsed."
    )
    duplicate: List[DebtRecord] = Field(
        default_factory=list, description="Debt records that accept more than one finding."
    )
    invalid: List[DebtRecord] = Field(
        default_factory=list,
        description="Debt records failing governance: unowned, non-waivable, or baseline in ratchet mode.",
    )

    def blocking(self) -> bool:
        """
        Whether the run holds any disposition that must fail the ratchet; expiring is a warning only.
        """

        return bool(self.new or self.stale or self.expired or self.duplicate or self.invalid)


class Audit(BaseModel):
    """
    A completed fitness run: the active mode, the reconciled report, and taxonomy readiness.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: GovernanceMode = Field(description="Rollout mode the run was evaluated under.")
    provisional: bool = Field(description="Whether the taxonomy still predates D1 approval.")
    report: Report = Field(description="The reconciled report.")

    def passed(self) -> bool:
        """
        Report mode always passes; ratchet mode requires no blocking dispositions and an approved taxonomy.
        """

        if self.mode is not GovernanceMode.RATCHET:
            return True

        return not self.report.blocking() and not self.provisional
