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


class Shadow(SealedModel):
    """
    Side-by-side live and trial readings for one adjudication.
    """

    turn: int = Field(ge=0, description="Zero-based adjudication index within the run.")

    live: Reading = Field(description="Decision that drove the run.")
    trial: Reading = Field(description="Decision the candidate decider produced.")

    @property
    def agrees(self) -> bool:
        """
        Return whether the trial reading matches the live reading exactly.
        """

        return self.live == self.trial


class Parity(SealedModel):
    """
    Replay agreement summary between a tape's recorded readings and a decider.
    """

    run: str = Field(description="Run identifier the tape was recorded from.")
    total: int = Field(ge=0, description="Number of adjudications replayed.")
    divergences: List[Shadow] = Field(
        default_factory=list,
        description="Turns where the decider disagreed with the recording.",
    )

    @property
    def matched(self) -> int:
        """
        Return the number of replayed adjudications that agreed with the recording.
        """

        return self.total - len(self.divergences)
