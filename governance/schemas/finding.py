from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from governance.schemas.selector import Selector


class Violation(BaseModel):
    """
    A single architecture-fitness violation found in one module.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    selector: Selector = Field(description="Rule and construct the violation targets.")
    line: int = Field(ge=1, description="One-based line of the offending construct.")
    message: str = Field(min_length=1, description="Actionable description of the violation.")
