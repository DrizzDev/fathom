from __future__ import annotations

from typing import Optional, Protocol

from fathom.schemas.completion import CompletionEvidence, GateDecision
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.turn import TurnEvidence
from fathom.schemas.vision import ActionKind


class Adjudicator(Protocol):
    """
    Structural contract for a decider that adjudicates one turn's completion evidence.
    """

    def adjudicate(
        self,
        *,
        sub_goal: SubGoal,
        action_kind: ActionKind,
        evidence: CompletionEvidence,
        measured: Optional[TurnEvidence] = None,
    ) -> GateDecision:
        """
        Return the gate decision for this turn.
        """
        ...
