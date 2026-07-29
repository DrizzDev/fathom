from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from governance.constants import RuleId


class Selector(BaseModel):
    """
    Identifies the rule and source construct a finding or debt record targets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: RuleId = Field(description="Rule producing the finding.")
    path: str = Field(min_length=1, description="Repository-relative source path.")
    detail: str = Field(min_length=1, description="Rule-specific construct identity.")
