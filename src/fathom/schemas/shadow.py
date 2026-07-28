from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from fathom.constants.completion import GateOutcome, RetainReason
from fathom.schemas.base.common import SealedModel
from fathom.schemas.completion import CompletionEvidence, GateDecision
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.turn import TurnEvidence
from fathom.schemas.vision import ActionKind


class Reading(SealedModel):
    """
    One decider's outcome for a single gate adjudication.
    """

    outcome: GateOutcome = Field(description="Decision the decider produced for the turn.")
    reason: Optional[RetainReason] = Field(
        default=None,
        description="Retain diagnostic present only when the outcome is RETAIN.",
    )

    @classmethod
    def from_decision(cls, *, decision: GateDecision) -> "Reading":
        """
        Project a gate decision onto its comparable reading.
        """

        return cls(outcome=decision.outcome, reason=decision.retain_reason)


class Trace(SealedModel):
    """
    One recorded gate adjudication with the full evidence that produced it.
    """

    turn: int = Field(ge=0, description="Zero-based adjudication index within the run.")
    task: SubGoal = Field(description="Sub-goal the gate adjudicated on this turn.")
    kind: ActionKind = Field(description="Emitted action kind passed to the gate.")

    reading: Reading = Field(description="Decision recorded on the live run.")
    evidence: CompletionEvidence = Field(description="Evidence bundle the gate consumed.")
    measured: Optional[TurnEvidence] = Field(
        default=None,
        description="Measured evidence channel captured alongside the gate's inputs.",
    )


class Tape(SealedModel):
    """
    Ordered gate adjudications recorded from one run.
    """

    run: str = Field(description="Run identifier the traces were recorded from.")
    traces: List[Trace] = Field(
        default_factory=list,
        description="Recorded adjudications in turn order.",
    )
