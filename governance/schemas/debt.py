from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from governance.constants import DebtState, Placeholder
from governance.schemas.finding import Violation
from governance.schemas.selector import Selector


class DebtRecord(BaseModel):
    """
    One recorded exception to an architecture-fitness rule.

    The ``reference`` is the permanent identity; the nested ``selector`` is mutable and must
    be updated by hand when code moves or a symbol is renamed (the checker does not track
    moves automatically).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reference: str = Field(min_length=1, description="Permanent debt identifier.")
    selector: Selector = Field(description="Rule and construct the exception targets.")
    owner: str = Field(min_length=1, description="Team or person accountable for closing the debt.")
    ticket: str = Field(min_length=1, description="Tracking issue for the remediation.")
    reason: str = Field(min_length=1, description="Why the exception exists.")
    expires: Optional[date] = Field(
        default=None, description="Date the exception must be closed by."
    )
    state: DebtState = Field(
        default=DebtState.BASELINE,
        description="Governance state; APPROVED requires full ownership.",
    )

    def matches(self, *, violation: Violation) -> bool:
        """
        Whether this record targets the given violation.
        """

        return self.selector == violation.selector

    def governed(self) -> bool:
        """
        Whether an APPROVED record carries a real owner, ticket, and expiry.
        """

        return (
            self.state is DebtState.APPROVED
            and self.owner != Placeholder.OWNER.value
            and self.ticket != Placeholder.TICKET.value
            and self.expires is not None
        )
