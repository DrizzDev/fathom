from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.completion import (
    GateOutcome,
    RetainReason,
    VerifyEvidenceDimension,
)
from fathom.schemas.tasks import ExecutionTaskState


class CompletionVerdict(BaseModel):
    """
    Verdict returned by the verify-node completion service for one task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    complete: bool = Field(description="Whether the task is observably complete.")
    next_state: ExecutionTaskState = Field(description="Target lifecycle state for the task.")

    reason: str = Field(description="Actionable reason for the verdict.")
    missing: List[VerifyEvidenceDimension] = Field(
        default_factory=list,
        description="Evidence dimensions required for completion but not observed.",
    )


class ClaimEvidence(BaseModel):
    """
    Evidence drawn from the LLM's own completion claim on this turn.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    asserted: bool = Field(
        description=(
            "True when the LLM set subgoal.completed or intent.completed this turn, "
            "or when the planned action_type is the dedicated completion action."
        ),
    )


class ActionEvidence(BaseModel):
    """
    Evidence about the action emitted on this turn.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dispatched: bool = Field(
        description=(
            "True when the planned action is a real dispatch-able action that reached "
            "the device adapter (not a no-op or planning-only directive)."
        ),
    )
    executed: bool = Field(
        description=(
            "True when the command primitive ran without an executor/device/control error, "
            "sourced from the runtime ExecutionResult.success (distinct from screen change and "
            "sub-goal completion)."
        ),
    )


class ScreenEvidence(BaseModel):
    """
    Evidence about the screen-state transition observed across this turn.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    evolved: bool = Field(
        description=(
            "True when the post-action screen differs from the pre-action screen, either by screen_changed flag or visual-hash divergence. "
            "Vetoed to false when the consolidated ActionEffect classifier reports NO_PROGRESS, so animation noise alone cannot satisfy the gate."
        ),
    )


class ValidationEvidence(BaseModel):
    """
    Evidence that a validate command executed on this turn.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    executed: bool = Field(
        default=False,
        description="True when this turn executed a validate command with a structured subject.",
    )


class CriterionEvidence(BaseModel):
    """
    Optional evidence from the typed-criterion observer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    observed: bool = Field(
        description=(
            "True when the sub-goal's typed criterion text is observable on the "
            "current screen. Provided as an additive RCA signal; never gates the completion decision on its own."
        ),
    )


class CompletionEvidence(BaseModel):
    """
    Per-turn evidence bundle adjudicated by the completion gate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim: ClaimEvidence = Field(
        description="Evidence drawn from the LLM's own completion claim this turn.",
    )
    action: ActionEvidence = Field(
        description="Evidence about the action emitted this turn.",
    )
    screen: ScreenEvidence = Field(
        description="Evidence about the screen transition observed across this turn.",
    )
    validation: ValidationEvidence = Field(
        default_factory=ValidationEvidence,
        description="Evidence that this turn executed a concrete validate command.",
    )
    criterion: Optional[CriterionEvidence] = Field(
        default=None,
        description=(
            "Optional typed-criterion evidence; logged for RCA but never used to "
            "veto an otherwise-conclusive gate decision."
        ),
    )
    notes: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Structured provenance strings explaining each individual signal.",
    )


class GateDecision(BaseModel):
    """
    The completion gate's structured decision for one turn.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: GateOutcome = Field(
        description="Whether to ADVANCE the sub-goal index, RETAIN it, or FAIL the run.",
    )
    retain_reason: Optional[RetainReason] = Field(
        default=None,
        description=(
            "Diagnostic code present only when outcome is RETAIN; identifies which "
            "missing signal blocked advancement."
        ),
    )
