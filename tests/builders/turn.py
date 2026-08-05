from __future__ import annotations

from typing import Optional

from fathom.constants.turn.advancement import ObservationPhase
from fathom.constants.turn.stall import StallState
from fathom.schemas.actions import Action
from fathom.schemas.capture import Capture
from fathom.schemas.completion import ActionEvidence, ClaimEvidence
from fathom.schemas.criterion import CriterionVerdict, Verdict
from fathom.schemas.requirement import CommandRequirement
from fathom.schemas.stall import StallSignal
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.success import ObservationRequirement
from fathom.schemas.turn import TurnEvidence
from tests.builders.actions import ActionFixtures


class TurnFixtures:
    """
    Canonical builders for Step, StepResult, Verdict, and correlated TurnEvidence.
    """

    @staticmethod
    def step(
        *,
        action: Optional[Action] = None,
        requirement: Optional[CommandRequirement] = None,
        step_number: int = 1,
    ) -> Step:
        """
        Build a Step carrying an optional admitted command requirement.
        """

        return Step(
            action=action if action is not None else ActionFixtures.make(),
            screen_hash="hash",
            requirement=requirement,
            step_number=step_number,
        )

    @classmethod
    def result(
        cls,
        *,
        step: Optional[Step] = None,
        success: bool = True,
        executed: bool = True,
        capture: Optional[Capture] = None,
    ) -> StepResult:
        """
        Build a StepResult correlated to a Step.
        """

        return StepResult(
            step=step if step is not None else cls.step(),
            success=success,
            executed=executed,
            capture=capture,
            pre_hash="pre",
            post_hash="post",
            screen_changed=True,
            duration=0,
        )

    @staticmethod
    def verdict(
        *, outcome: CriterionVerdict = CriterionVerdict.SATISFIED, confidence: float = 0.95
    ) -> Verdict:
        """
        Build an oracle verdict.
        """

        return Verdict(outcome=outcome, confidence=confidence, evidence="observed")

    @classmethod
    def evidence(
        cls,
        *,
        phase: ObservationPhase = ObservationPhase.POST_DISPATCH,
        execution: Optional[StepResult] = None,
        observation: Optional[ObservationRequirement] = None,
        verdict: Optional[Verdict] = None,
        claim: bool = False,
    ) -> TurnEvidence:
        """
        Build correlated TurnEvidence for the advancement policy.
        """

        return TurnEvidence(
            claim=ClaimEvidence(asserted=claim),
            action=ActionEvidence(dispatched=execution is not None, executed=execution is not None),
            phase=phase,
            execution=execution,
            observation=observation,
            verdict=verdict,
            stall=StallSignal(state=StallState.FLOWING, streak=0),
        )
