from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fathom.constants.reasoning import MAXIMUM_DECOMPOSITION_SUB_GOALS
from fathom.schemas.base.common import NonBlank, SealedModel
from fathom.schemas.proposal import DecompositionProposal


class DecomposedTask(SealedModel):
    """
    One decomposed sub-goal: an imperative objective and its untrusted typed success proposal.
    """

    objective: NonBlank = Field(description="Imperative objective this sub-goal must achieve.")
    proposal: DecompositionProposal = Field(
        description="Untrusted per-goal success proposal, translated to canonical Success at the boundary."
    )


class DecompositionSchema(BaseModel):
    """
    Schema for intent decomposition output.
    """

    model_config = ConfigDict(extra="forbid")

    # Non-nullable: Gemini structured output rejects a nullable number carrying minimum/maximum
    # bounds (an Optional[float] with ge/le), so confidence is a plain bounded float with a default.
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Decomposer self-reported confidence in the produced plan.",
    )
    # No ``min_length``/``max_length``: Gemini structured output rejects ``minItems``/``maxItems`` on
    # a list of union-typed items. The bounds are enforced by the validator below instead.
    sub_goals: List[DecomposedTask] = Field(
        description="Ordered, non-empty typed tasks; raw strings are rejected at the boundary.",
    )

    @field_validator("sub_goals")
    @classmethod
    def __bounded_non_empty(cls, value: List[DecomposedTask]) -> List[DecomposedTask]:
        """
        Reject an empty plan or one exceeding the decomposition ceiling.
        """

        if not value:
            raise ValueError("decomposition must contain at least one sub-goal")
        if len(value) > MAXIMUM_DECOMPOSITION_SUB_GOALS:
            raise ValueError(
                f"decomposition exceeds the {MAXIMUM_DECOMPOSITION_SUB_GOALS}-sub-goal ceiling"
            )
        return value
