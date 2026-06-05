from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.agent import DirectiveKind


class OperatorDirective(BaseModel):
    """
    Authoritative instruction issued by an operator (HITL response or remote operator)
    that overrides autonomous-mode guards for the next matching planner turn.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: DirectiveKind = Field(description="Classification of the directive.")
    source_text: str = Field(
        min_length=1, description="Raw operator text that produced this directive."
    )
    target_descriptor: Optional[str] = Field(
        default=None,
        description="Normalized target the directive points to (e.g. tap label or completion phrase).",
    )
    set_at_step: int = Field(ge=0, description="Step index when the directive was issued.")
    ttl_turns: int = Field(
        gt=0,
        description="Remaining planner turns the directive stays active before auto-clearing.",
    )

    def decremented(self) -> "OperatorDirective":
        """
        Return a copy with ``ttl_turns`` reduced by one.
        """

        return self.model_copy(update={"ttl_turns": max(0, self.ttl_turns - 1)})

    @property
    def expired(self) -> bool:
        """
        Whether this directive has exhausted its TTL.
        """

        return self.ttl_turns <= 0
