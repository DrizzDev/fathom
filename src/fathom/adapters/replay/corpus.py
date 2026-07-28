from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, Final, FrozenSet, List, Optional

from fathom.constants.completion import GateOutcome, RetainReason
from fathom.schemas.completion import (
    ActionEvidence,
    ClaimEvidence,
    CompletionEvidence,
    ScreenEvidence,
)
from fathom.schemas.shadow import Reading, Tape, Trace
from fathom.schemas.subgoal import SubGoal, SubGoalKind
from fathom.schemas.vision import ActionKind

if TYPE_CHECKING:
    from pathlib import Path


class Corpus:
    """
    Loads recorded gate tapes from disk for replay.
    """

    __PATTERN: Final[str] = "*.tape.json"
    __UNKNOWN_DESCRIPTION: Final[str] = ""

    __KINDS: Final[FrozenSet[str]] = frozenset(kind.value for kind in ActionKind)
    __TASKS: Final[FrozenSet[str]] = frozenset(kind.value for kind in SubGoalKind)
    __REASONS: Final[FrozenSet[str]] = frozenset(reason.value for reason in RetainReason)
    __OUTCOMES: Final[FrozenSet[str]] = frozenset(outcome.value for outcome in GateOutcome)

    def __init__(self, *, root: Path) -> None:
        """
        Bind the directory containing typed tape files.
        """

        self.__root = root

    def load(self) -> List[Tape]:
        """
        Load every typed tape file under the corpus root in name order.
        """

        return [
            Tape.model_validate_json(path.read_text())
            for path in sorted(self.__root.glob(self.__PATTERN))
        ]

    @classmethod
    def legacy(cls, *, path: Path) -> List[Tape]:
        """
        Convert the POC extraction archive into typed tapes, dropping turns without evidence.
        """

        payload: Dict[str, List[Dict[str, Any]]] = json.loads(path.read_text())
        return [Tape(run=run, traces=cls.__traces(turns=turns)) for run, turns in payload.items()]

    @classmethod
    def __traces(cls, *, turns: List[Dict[str, Any]]) -> List[Trace]:
        """
        Convert extracted turns into traces, preserving turn order.
        """

        traces: List[Trace] = []
        for index, turn in enumerate(turns):
            if (trace := cls.__trace(index=index, turn=turn)) is not None:
                traces.append(trace)

        return traces

    @classmethod
    def __trace(cls, *, index: int, turn: Dict[str, Any]) -> Optional[Trace]:
        """
        Convert one extracted turn; return None when it lacks replayable evidence.
        """

        if "claim" not in turn or str(turn.get("outcome")) not in cls.__OUTCOMES:
            return None

        return Trace(
            turn=index,
            task=cls.__task(turn=turn),
            reading=cls.__reading(turn=turn),
            evidence=cls.__evidence(turn=turn),
            kind=cls.__kind(value=str(turn.get("act"))),
        )

    @classmethod
    def __task(cls, *, turn: Dict[str, Any]) -> SubGoal:
        """
        Build the sub-goal skeleton; legacy archives carry no sub-goal text.
        """

        value = str(turn.get("kind"))
        kind = SubGoalKind(value) if value in cls.__TASKS else SubGoalKind.ACTION

        return SubGoal(
            kind=kind,
            description=cls.__UNKNOWN_DESCRIPTION,
            index=int(turn.get("sg", 0)),
        )

    @classmethod
    def __kind(cls, *, value: str) -> ActionKind:
        """
        Map the recorded action kind token, defaulting to UNKNOWN.
        """

        return ActionKind(value) if value in cls.__KINDS else ActionKind.UNKNOWN

    @classmethod
    def __evidence(cls, *, turn: Dict[str, Any]) -> CompletionEvidence:
        """
        Rebuild the evidence bundle; executed mirrors dispatched because the gate never reads it.
        """

        dispatched = bool(turn.get("dispatched"))

        return CompletionEvidence(
            claim=ClaimEvidence(
                asserted=bool(turn.get("claim")),
                explained=bool(turn.get("explained")),
            ),
            screen=ScreenEvidence(evolved=bool(turn.get("evolved"))),
            action=ActionEvidence(dispatched=dispatched, executed=dispatched),
        )

    @classmethod
    def __reading(cls, *, turn: Dict[str, Any]) -> Reading:
        """
        Rebuild the recorded decision; advance-path annotations carry no retain reason.
        """

        value = str(turn.get("reason"))
        reason = RetainReason(value) if value in cls.__REASONS else None

        return Reading(outcome=GateOutcome(str(turn.get("outcome"))), reason=reason)
