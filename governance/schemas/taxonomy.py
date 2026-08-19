from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Taxonomy(BaseModel):
    """
    Package-layer classification consumed by the fitness rules.

    Owned by the D1 taxonomy design; until D1 is approved it is ``provisional`` and its
    findings are candidates, not a complete statement of the domain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provisional: bool = Field(
        default=True, description="Whether the classification predates D1 approval."
    )
    domain: Tuple[str, ...] = Field(
        min_length=1, description="Dotted package prefixes required to stay pure-domain."
    )

    @model_validator(mode="after")
    def __unique(self) -> "Taxonomy":
        """
        Reject duplicate domain prefixes; the classification must be unambiguous.
        """

        if len(self.domain) != len(set(self.domain)):
            raise ValueError("duplicate domain prefixes in taxonomy")

        return self
