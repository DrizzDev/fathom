from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, model_validator

from governance.constants import GovernanceMode
from governance.schemas.debt import DebtRecord


class Manifest(BaseModel):
    """
    The checked-in governance configuration: the ratchet mode and the accepted debt records.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: GovernanceMode = Field(
        default=GovernanceMode.REPORT, description="Rollout mode shared by the CLI and tests."
    )
    records: List[DebtRecord] = Field(default_factory=list, description="Recorded debt exceptions.")

    @model_validator(mode="after")
    def __unique(self) -> "Manifest":
        """
        Reject duplicate debt references or selectors; a manifest must be unambiguous.
        """

        references = [record.reference for record in self.records]
        if len(references) != len(set(references)):
            raise ValueError("duplicate debt references in manifest")

        selectors = [record.selector for record in self.records]
        if len(selectors) != len(set(selectors)):
            raise ValueError("duplicate debt selectors in manifest")

        return self
