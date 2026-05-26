"""
Unit pins for :class:`EscalationGate.decide` covering the full decision matrix.
"""

from __future__ import annotations

import unittest
from typing import Tuple

from fathom.core.agent.escalation_gate import EscalationGate
from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.escalation import (
    EscalationPolicy,
    EscalationReason,
    StuckSource,
)
from fathom.schemas.loop import LoopEvidence, LoopReason, LoopTurn
from fathom.schemas.vision import ActionKind


class EscalationGateDecisionTest(unittest.TestCase):
    """
    Pins the decision matrix of :class:`EscalationGate`.
    """

    @staticmethod
    def __turn(
        kind: ActionKind,
        *,
        effect: ActionEffectStatus = ActionEffectStatus.NO_PROGRESS,
        action_type: str = "x",
    ) -> LoopTurn:
        """
        Build a typed :class:`LoopTurn` with sensible defaults.
        """

        return LoopTurn(action_kind=kind, action_type=action_type, effect_status=effect)

    @classmethod
    def __evidence(cls, *turns: LoopTurn) -> LoopEvidence:
        """
        Wrap turns in a :class:`LoopEvidence` snapshot using the same tuple
        for ``recent`` and ``since_progress`` so tests are explicit about
        what the gate is asked to consider.
        """

        recent: Tuple[LoopTurn, ...] = tuple(turns)
        return LoopEvidence(
            stuck=True,
            reason=LoopReason.INERT_REPETITION,
            recent=recent,
            since_progress=recent,
        )

    def __gate(self, *, policy: EscalationPolicy = EscalationPolicy()) -> EscalationGate:
        return EscalationGate(policy=policy)

    def test_disabled_policy_always_allows(self) -> None:
        """
        With the master switch off, every signal escalates.
        """

        gate = self.__gate(policy=EscalationPolicy(enabled=False))
        decision = gate.decide(
            source=StuckSource.LOOP_DETECTOR,
            evidence=self.__evidence(self.__turn(ActionKind.VALIDATION)),
            deferrals=0,
        )
        self.assertTrue(decision.allow)
        self.assertIs(decision.reason, EscalationReason.DISABLED)

    def test_deferrals_above_limit_triggers_escape_valve(self) -> None:
        """
        When deferrals exceed the configured limit, the gate must escalate.
        """

        gate = self.__gate(policy=EscalationPolicy(deferral_limit=2))
        decision = gate.decide(
            source=StuckSource.LOOP_DETECTOR,
            evidence=self.__evidence(self.__turn(ActionKind.VALIDATION)),
            deferrals=3,
        )
        self.assertTrue(decision.allow)
        self.assertIs(decision.reason, EscalationReason.DEFERRAL_LIMIT)

    def test_deferrals_at_limit_does_not_escape(self) -> None:
        """
        Escape valve fires strictly above the limit, not at it.
        """

        gate = self.__gate(policy=EscalationPolicy(deferral_limit=2))
        decision = gate.decide(
            source=StuckSource.LOOP_DETECTOR,
            evidence=self.__evidence(self.__turn(ActionKind.VALIDATION)),
            deferrals=2,
        )
        self.assertFalse(decision.allow)
        self.assertIs(decision.reason, EscalationReason.PASSIVE_RUN)

    def test_subgoal_budget_source_always_escalates(self) -> None:
        """
        Sub-goal budget exhaustion is a hard signal regardless of action mix.
        """

        gate = self.__gate()
        decision = gate.decide(
            source=StuckSource.SUBGOAL_BUDGET,
            evidence=self.__evidence(self.__turn(ActionKind.VALIDATION)),
            deferrals=0,
        )
        self.assertTrue(decision.allow)
        self.assertIs(decision.reason, EscalationReason.SUBGOAL_BUDGET)

    def test_passive_run_below_tolerance_defers(self) -> None:
        """
        Three consecutive validate-only NO_PROGRESS turns with tolerance=3 defer.
        """

        gate = self.__gate()
        decision = gate.decide(
            source=StuckSource.LOOP_DETECTOR,
            evidence=self.__evidence(
                self.__turn(ActionKind.VALIDATION),
                self.__turn(ActionKind.VALIDATION),
                self.__turn(ActionKind.VALIDATION),
            ),
            deferrals=0,
        )
        self.assertFalse(decision.allow)
        self.assertIs(decision.reason, EscalationReason.PASSIVE_RUN)

    def test_passive_run_above_tolerance_escalates(self) -> None:
        """
        Four consecutive validate-only NO_PROGRESS turns at tolerance=3 escalate.
        """

        gate = self.__gate()
        decision = gate.decide(
            source=StuckSource.LOOP_DETECTOR,
            evidence=self.__evidence(
                *(self.__turn(ActionKind.VALIDATION) for _ in range(4))
            ),
            deferrals=0,
        )
        self.assertTrue(decision.allow)
        self.assertIs(decision.reason, EscalationReason.PASSIVE_LIMIT)

    def test_active_stall_in_tail_escalates(self) -> None:
        """
        Any non-passive NO_PROGRESS turn in ``since_progress`` escalates.
        """

        gate = self.__gate()
        decision = gate.decide(
            source=StuckSource.LOOP_DETECTOR,
            evidence=self.__evidence(
                self.__turn(ActionKind.NAVIGATION),
                self.__turn(ActionKind.VALIDATION),
            ),
            deferrals=0,
        )
        self.assertTrue(decision.allow)
        self.assertIs(decision.reason, EscalationReason.ACTIVE_STALL)

    def test_all_uncertain_tail_escalates_not_defers(self) -> None:
        """
        Tail with no explicit NO_PROGRESS evidence escalates — Blocker 3 fix.
        Do not fabricate no-progress from absent data.
        """

        gate = self.__gate()
        decision = gate.decide(
            source=StuckSource.LOOP_DETECTOR,
            evidence=self.__evidence(
                self.__turn(ActionKind.NAVIGATION, effect=ActionEffectStatus.UNCERTAIN),
                self.__turn(ActionKind.NAVIGATION, effect=ActionEffectStatus.UNCERTAIN),
                self.__turn(ActionKind.NAVIGATION, effect=ActionEffectStatus.UNCERTAIN),
            ),
            deferrals=0,
        )
        self.assertTrue(decision.allow)
        self.assertIs(decision.reason, EscalationReason.ACTIVE_STALL)

    def test_uncertain_validation_turn_is_pass_through(self) -> None:
        """
        UNCERTAIN validation turns are pass-through (do not count, do not break).
        """

        gate = self.__gate()
        decision = gate.decide(
            source=StuckSource.LOOP_DETECTOR,
            evidence=self.__evidence(
                self.__turn(ActionKind.VALIDATION, effect=ActionEffectStatus.UNCERTAIN),
                self.__turn(ActionKind.VALIDATION),
                self.__turn(ActionKind.VALIDATION),
            ),
            deferrals=0,
        )
        # Two NO_PROGRESS validates with tolerance=3 still defer (within run).
        self.assertFalse(decision.allow)
        self.assertIs(decision.reason, EscalationReason.PASSIVE_RUN)

    def test_decision_carries_input_deferrals(self) -> None:
        """
        ``deferrals`` on the decision mirrors the value passed in at decision time.
        """

        gate = self.__gate()
        decision = gate.decide(
            source=StuckSource.LOOP_DETECTOR,
            evidence=self.__evidence(self.__turn(ActionKind.VALIDATION)),
            deferrals=1,
        )
        self.assertEqual(decision.deferrals, 1)

    def test_decision_carries_input_source(self) -> None:
        """
        ``stuck_source`` on the decision mirrors the source the gate was asked about.
        """

        gate = self.__gate()
        decision = gate.decide(
            source=StuckSource.LOOP_DETECTOR,
            evidence=self.__evidence(self.__turn(ActionKind.VALIDATION)),
            deferrals=0,
        )
        self.assertIs(decision.stuck_source, StuckSource.LOOP_DETECTOR)
