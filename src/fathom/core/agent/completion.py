from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Dict, List, Type

from pydantic import BaseModel, ConfigDict, Field

from fathom.core.exceptions import ConfigurationError
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.subgoal import SubGoalKind


class CompletionEvidence(StrEnum):
    """
    Named evidence dimensions that the completion gate inspects on a :class:`SubGoalCompletionSignal`.
    """

    CLAIM_VERIFIED = "CLAIM_VERIFIED"
    ACTION_EFFECTIVE = "ACTION_EFFECTIVE"
    FLAGGED_COMPLETE = "FLAGGED_COMPLETE"


class CompletionVerdict(BaseModel):
    """
    Outcome of a :class:`CompletionGate` evaluation.
    ``reason`` is the human-readable explanation for telemetry;
    ``complete`` is the binary decision used by RECORD to advance the sub-goal index;
    ``missing`` lists evidence dimensions the gate required but did not observe (empty when ``complete=True``).
    """

    complete: bool = Field(description="Whether the sub-goal is considered complete")
    reason: str = Field(description="Human-readable explanation of the verdict")
    missing: List[CompletionEvidence] = Field(
        default_factory=list, description="Evidence dimensions required but not observed"
    )

    model_config = ConfigDict(frozen=True)


class CompletionGate(ABC):
    """
    Strategy for deciding whether a :class:`SubGoalCompletionSignal` satisfies the completion policy for one sub-goal kind.
    Replaces threshold/count arithmetic with named, kind-specific rules so adding or removing an evidence dimension touches exactly one gate.
    """

    @abstractmethod
    def evaluate(self, *, signal: SubGoalCompletionSignal) -> CompletionVerdict:
        """
        Evaluate the signal and return the verdict for this gate kind.
        """

        raise NotImplementedError


class ActionStepGate(CompletionGate):
    """
    Completion gate for action-type sub-goals (tap, type, swipe, etc.).
    Requires BOTH model self-report agreement (``claim_verified``) and
    effective action landing (``action_effective``); neither alone is sufficient to advance.
    """

    def evaluate(self, *, signal: SubGoalCompletionSignal) -> CompletionVerdict:
        """
        Verify both ``claim_verified`` and ``action_effective`` are set;
        otherwise return the verdict with the missing dimensions listed.
        """

        missing: List[CompletionEvidence] = []

        if not signal.claim_verified:
            missing.append(CompletionEvidence.CLAIM_VERIFIED)

        if not signal.action_effective:
            missing.append(CompletionEvidence.ACTION_EFFECTIVE)

        if missing:
            return CompletionVerdict(
                complete=False,
                missing=missing,
                reason=f"Action sub-goal missing evidence: {', '.join(missing)}",
            )

        return CompletionVerdict(
            complete=True,
            reason="Action sub-goal verified (claim + effective action)",
        )


class ValidationStepGate(CompletionGate):
    """
    Completion gate for validation-type sub-goals (verify / check /
    confirm). The screen is not expected to change for a validation;
    the model's explicit completion flag is the load-bearing signal.
    """

    def evaluate(self, *, signal: SubGoalCompletionSignal) -> CompletionVerdict:
        """
        Verify the model raised ``flagged_complete`` for this step.
        Validation does not require an effective action since observing
        the screen is itself the goal.
        """

        if signal.flagged_complete:
            return CompletionVerdict(
                complete=True,
                reason="Validation sub-goal verified by model claim",
            )

        return CompletionVerdict(
            complete=False,
            missing=[CompletionEvidence.FLAGGED_COMPLETE],
            reason="Validation sub-goal missing evidence: flagged_complete",
        )


class CompletionGateFactory:
    """
    Resolves the :class:`CompletionGate` strategy for a given
    :class:`SubGoalKind`. Mirrors the ``PromptFactory`` pattern: a
    class-level dict populated at import time, classmethod resolver.
    """

    __gates: Dict[SubGoalKind, Type[CompletionGate]] = {
        SubGoalKind.ACTION: ActionStepGate,
        SubGoalKind.VALIDATION: ValidationStepGate,
    }

    @classmethod
    def for_kind(cls, *, kind: SubGoalKind) -> CompletionGate:
        """
        Return the gate strategy for ``kind``. Raises
        :class:`ConfigurationError` for unknown kinds so misconfiguration
        fails fast at the call site.
        """

        __class = cls.__gates.get(kind)

        if __class is None:
            raise ConfigurationError(f"No completion gate registered for sub-goal kind: {kind}")

        return __class()
