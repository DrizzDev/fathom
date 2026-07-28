from __future__ import annotations

from typing import List

from fathom.interfaces.gate import Adjudicator
from fathom.schemas.shadow import Parity, Reading, Shadow, Tape


class Replayer:
    """
    Replays recorded gate tapes through a decider and reports agreement.
    """

    def __init__(self, *, decider: Adjudicator) -> None:
        """
        Bind the decider whose decisions are compared against the recordings.
        """

        self.__decider = decider

    def replay(self, *, tape: Tape) -> Parity:
        """
        Re-adjudicate every trace on the tape and collect divergences.
        """

        divergences: List[Shadow] = []

        for trace in tape.traces:
            decision = self.__decider.adjudicate(
                sub_goal=trace.task,
                action_kind=trace.kind,
                evidence=trace.evidence,
                measured=trace.measured,
            )

            if (trial := Reading.from_decision(decision=decision)) != trace.reading:
                divergences.append(Shadow(turn=trace.turn, live=trace.reading, trial=trial))

        return Parity(run=tape.run, total=len(tape.traces), divergences=divergences)
