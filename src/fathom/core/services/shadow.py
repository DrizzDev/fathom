from __future__ import annotations

from logging import getLogger
from typing import Optional

from fathom.interfaces.gate import Adjudicator
from fathom.schemas.completion import CompletionEvidence, GateDecision
from fathom.schemas.shadow import Reading, Shadow, Trace
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.turn import TurnEvidence
from fathom.schemas.vision import ActionKind

logger = getLogger(__name__)


class ShadowRecorder:
    """
    Mirrors live gate adjudications against an optional trial decider, recording both without acting.
    """

    def __init__(self, *, trial: Optional[Adjudicator] = None) -> None:
        """
        Bind the optional trial decider whose readings are recorded, never obeyed.
        """

        self.__trial = trial

    def observe(
        self,
        *,
        turn: int,
        workflow_id: str,
        sub_goal: SubGoal,
        decision: GateDecision,
        action_kind: ActionKind,
        evidence: CompletionEvidence,
        measured: Optional[TurnEvidence] = None,
    ) -> Optional[Shadow]:
        """
        Record the live adjudication and the trial reading; never raise into the live loop.
        """

        try:
            trace = Trace(
                turn=turn,
                task=sub_goal,
                kind=action_kind,
                evidence=evidence,
                measured=measured,
                reading=Reading.from_decision(decision=decision),
            )
            self.__record(workflow_id=workflow_id, trace=trace)

            if (trial := self.__trial) is None:
                return None

            return self.__mirror(workflow_id=workflow_id, trace=trace, trial=trial)
        except Exception as exception:
            logger.warning(
                "Shadow recording failed; live decision unaffected",
                extra={
                    "workflow.id": workflow_id,
                    "event": "shadow.trial.failed",
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            return None

    def __record(self, *, workflow_id: str, trace: Trace) -> None:
        """
        Emit the typed trace as a structured log entry for corpus extraction.
        """

        logger.info(
            "Shadow trace recorded",
            extra={
                "workflow.id": workflow_id,
                "event": "shadow.trace.recorded",
                "shadow.trace": trace.model_dump(mode="json"),
            },
        )

    def __mirror(self, *, workflow_id: str, trace: Trace, trial: Adjudicator) -> Shadow:
        """
        Adjudicate the trial decider on the recorded evidence and log the comparison.
        """

        decision = trial.adjudicate(
            sub_goal=trace.task,
            action_kind=trace.kind,
            evidence=trace.evidence,
            measured=trace.measured,
        )
        shadow = Shadow(
            turn=trace.turn,
            live=trace.reading,
            trial=Reading.from_decision(decision=decision),
        )

        logger.info(
            "Shadow trial compared",
            extra={
                "workflow.id": workflow_id,
                "event": "shadow.trial.compared",
                "shadow.agrees": shadow.agrees,
                "shadow.comparison": shadow.model_dump(mode="json"),
            },
        )
        return shadow
