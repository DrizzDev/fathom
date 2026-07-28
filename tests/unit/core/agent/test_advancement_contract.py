from __future__ import annotations

import unittest
from typing import Dict, Optional, Tuple

from fathom.constants.capability import CompletionMode
from fathom.constants.completion import RetainReason
from fathom.constants.turn.advancement import AdvanceKind
from fathom.constants.turn.stall import StallState
from fathom.constants.turn.validation import ValidationSource
from fathom.core.agent.advancement import AdvancementPolicy
from fathom.schemas.advancement import Advancement, RetainHistory
from fathom.schemas.completion import ActionEvidence, ClaimEvidence
from fathom.schemas.criterion import CriterionVerdict, Verdict
from fathom.schemas.effect import ActionEffectStatus, EffectReading
from fathom.schemas.stall import StallSignal
from fathom.schemas.subgoal import SubGoalKind
from fathom.schemas.tasks import Task
from fathom.schemas.turn import TurnEvidence
from fathom.schemas.validation import Validation

Facets = Dict[str, object]
Row = Tuple[str, CompletionMode, Facets, int, AdvanceKind, Optional[RetainReason], bool]


class AdvancementContractTest(unittest.TestCase):
    """
    Golden-master truth table pinning AdvancementPolicy.decide over every branch and boundary.

    Every row is derived directly from the decision code so any structural refactor must
    reproduce the exact (kind, reason, redispatch) triple or this contract fails.
    """

    def setUp(self) -> None:
        """
        Build the policy at its production defaults (floor 0.7, backstop limit 3).
        """

        self.policy = AdvancementPolicy()

    def test_signal_required_modes(self) -> None:
        """
        Terminal, validation, and claim-or-timeout advance on their one required signal, else retain.
        """

        rows: Tuple[Row, ...] = (
            (
                "terminal dispatched",
                CompletionMode.TERMINAL,
                {"dispatched": True},
                0,
                AdvanceKind.ADVANCE,
                None,
                True,
            ),
            (
                "terminal not dispatched",
                CompletionMode.TERMINAL,
                {},
                0,
                AdvanceKind.RETAIN,
                RetainReason.MISSING_DISPATCH,
                True,
            ),
            (
                "claim_verified validated",
                CompletionMode.CLAIM_VERIFIED,
                {"validation": True},
                0,
                AdvanceKind.ADVANCE,
                None,
                True,
            ),
            (
                "claim_verified missing",
                CompletionMode.CLAIM_VERIFIED,
                {},
                0,
                AdvanceKind.RETAIN,
                RetainReason.MISSING_VALIDATION,
                True,
            ),
            (
                "claim_or_timeout claimed",
                CompletionMode.CLAIM_OR_TIMEOUT,
                {"claim": True},
                0,
                AdvanceKind.ADVANCE,
                None,
                True,
            ),
            (
                "claim_or_timeout missing",
                CompletionMode.CLAIM_OR_TIMEOUT,
                {},
                0,
                AdvanceKind.RETAIN,
                RetainReason.MISSING_CLAIM,
                True,
            ),
        )
        self.__check(rows=rows)

    def test_capture_mode_always_retains(self) -> None:
        """
        Capture completion is decided by a separate upstream policy; this decider always retains.
        """

        rows: Tuple[Row, ...] = (
            (
                "capture bare",
                CompletionMode.CAPTURE_VERIFIED,
                {},
                0,
                AdvanceKind.RETAIN,
                RetainReason.MISSING_CAPTURE,
                True,
            ),
            (
                "capture ignores satisfied verdict",
                CompletionMode.CAPTURE_VERIFIED,
                {"verdict": CriterionVerdict.SATISFIED, "confidence": 0.99, "dispatched": True},
                0,
                AdvanceKind.RETAIN,
                RetainReason.MISSING_CAPTURE,
                True,
            ),
        )
        self.__check(rows=rows)

    def test_durable_mode(self) -> None:
        """
        Durable tasks advance only on an observed outcome or a validation; every retain bars re-dispatch.
        """

        rows: Tuple[Row, ...] = (
            (
                "observed and acted",
                CompletionMode.OUTCOME_VERIFIED,
                {"verdict": CriterionVerdict.SATISFIED, "confidence": 0.9, "dispatched": True},
                0,
                AdvanceKind.ADVANCE,
                None,
                True,
            ),
            (
                "observed already true",
                CompletionMode.OUTCOME_VERIFIED,
                {"verdict": CriterionVerdict.SATISFIED, "confidence": 0.9},
                0,
                AdvanceKind.SATISFIED_PRIOR,
                None,
                True,
            ),
            (
                "satisfied at the floor",
                CompletionMode.OUTCOME_VERIFIED,
                {"verdict": CriterionVerdict.SATISFIED, "confidence": 0.7, "dispatched": True},
                0,
                AdvanceKind.ADVANCE,
                None,
                True,
            ),
            (
                "below the floor awaits proof",
                CompletionMode.OUTCOME_VERIFIED,
                {"verdict": CriterionVerdict.SATISFIED, "confidence": 0.69, "dispatched": True},
                0,
                AdvanceKind.RETAIN,
                RetainReason.AWAITING_PROOF,
                False,
            ),
            (
                "validated advances",
                CompletionMode.OUTCOME_VERIFIED,
                {"validation": True},
                0,
                AdvanceKind.ADVANCE,
                None,
                True,
            ),
            (
                "regression bars redispatch",
                CompletionMode.OUTCOME_VERIFIED,
                {"effect": ActionEffectStatus.REGRESSION},
                0,
                AdvanceKind.RETAIN,
                RetainReason.LEFT_APPLICATION,
                False,
            ),
            (
                "bare awaits proof",
                CompletionMode.OUTCOME_VERIFIED,
                {},
                0,
                AdvanceKind.RETAIN,
                RetainReason.AWAITING_PROOF,
                False,
            ),
            (
                "refuted still awaits proof pre-exhaustion",
                CompletionMode.OUTCOME_VERIFIED,
                {"verdict": CriterionVerdict.UNSATISFIED, "confidence": 0.9},
                0,
                AdvanceKind.RETAIN,
                RetainReason.AWAITING_PROOF,
                False,
            ),
        )
        self.__check(rows=rows)

    def test_transient_mode(self) -> None:
        """
        Screen-verified tasks advance on an observed outcome or a progress-corroborated claim.
        """

        rows: Tuple[Row, ...] = (
            (
                "regression retains but allows redispatch",
                CompletionMode.SCREEN_VERIFIED,
                {"effect": ActionEffectStatus.REGRESSION},
                0,
                AdvanceKind.RETAIN,
                RetainReason.LEFT_APPLICATION,
                True,
            ),
            (
                "observed and acted",
                CompletionMode.SCREEN_VERIFIED,
                {"verdict": CriterionVerdict.SATISFIED, "confidence": 0.9, "dispatched": True},
                0,
                AdvanceKind.ADVANCE,
                None,
                True,
            ),
            (
                "observed already true",
                CompletionMode.SCREEN_VERIFIED,
                {"verdict": CriterionVerdict.SATISFIED, "confidence": 0.9},
                0,
                AdvanceKind.SATISFIED_PRIOR,
                None,
                True,
            ),
            (
                "refuted vetoes claim advance",
                CompletionMode.SCREEN_VERIFIED,
                {
                    "verdict": CriterionVerdict.UNSATISFIED,
                    "confidence": 0.9,
                    "claim": True,
                    "dispatched": True,
                    "effect": ActionEffectStatus.PROGRESS,
                },
                0,
                AdvanceKind.RETAIN,
                RetainReason.AWAITING_PROOF,
                True,
            ),
            (
                "validated claim advances",
                CompletionMode.SCREEN_VERIFIED,
                {"validation": True, "claim": True},
                0,
                AdvanceKind.ADVANCE,
                None,
                True,
            ),
            (
                "progress-corroborated claim advances",
                CompletionMode.SCREEN_VERIFIED,
                {"claim": True, "dispatched": True, "effect": ActionEffectStatus.PROGRESS},
                0,
                AdvanceKind.ADVANCE,
                None,
                True,
            ),
            (
                "validation without claim is insufficient",
                CompletionMode.SCREEN_VERIFIED,
                {"validation": True},
                0,
                AdvanceKind.RETAIN,
                RetainReason.MISSING_CLAIM,
                True,
            ),
        )
        self.__check(rows=rows)

    def test_transient_diagnose(self) -> None:
        """
        The transient fallthrough names the first missing signal.
        """

        rows: Tuple[Row, ...] = (
            (
                "no claim",
                CompletionMode.SCREEN_VERIFIED,
                {},
                0,
                AdvanceKind.RETAIN,
                RetainReason.MISSING_CLAIM,
                True,
            ),
            (
                "claim but no dispatch",
                CompletionMode.SCREEN_VERIFIED,
                {"claim": True},
                0,
                AdvanceKind.RETAIN,
                RetainReason.MISSING_DISPATCH,
                True,
            ),
            (
                "dispatched but no progress",
                CompletionMode.SCREEN_VERIFIED,
                {"claim": True, "dispatched": True, "effect": ActionEffectStatus.NO_PROGRESS},
                0,
                AdvanceKind.RETAIN,
                RetainReason.MISSING_SCREEN_EVOLUTION,
                True,
            ),
        )
        self.__check(rows=rows)

    def test_escalation_on_exhaustion_and_stall(self) -> None:
        """
        A retaining turn escalates once the streak hits the limit or momentum stalls; refutation is terminal.
        """

        rows: Tuple[Row, ...] = (
            (
                "exhausted streak escalates",
                CompletionMode.SCREEN_VERIFIED,
                {},
                3,
                AdvanceKind.ESCALATE,
                None,
                False,
            ),
            (
                "exhausted refuted is unsatisfiable",
                CompletionMode.SCREEN_VERIFIED,
                {"verdict": CriterionVerdict.UNSATISFIED, "confidence": 0.9},
                3,
                AdvanceKind.UNSATISFIABLE,
                None,
                False,
            ),
            (
                "stall escalates before the limit",
                CompletionMode.SCREEN_VERIFIED,
                {"stall": StallState.STALLED},
                0,
                AdvanceKind.ESCALATE,
                None,
                False,
            ),
            (
                "durable awaiting proof escalates when exhausted",
                CompletionMode.OUTCOME_VERIFIED,
                {},
                3,
                AdvanceKind.ESCALATE,
                None,
                False,
            ),
            (
                "durable refuted is unsatisfiable when exhausted",
                CompletionMode.OUTCOME_VERIFIED,
                {"verdict": CriterionVerdict.UNSATISFIED, "confidence": 0.9},
                3,
                AdvanceKind.UNSATISFIABLE,
                None,
                False,
            ),
        )
        self.__check(rows=rows)

    def test_escalation_negatives(self) -> None:
        """
        Escalation never fires below the limit without a stall, and never touches a non-retaining turn.
        """

        rows: Tuple[Row, ...] = (
            (
                "below limit stays retained",
                CompletionMode.SCREEN_VERIFIED,
                {},
                2,
                AdvanceKind.RETAIN,
                RetainReason.MISSING_CLAIM,
                True,
            ),
            (
                "flowing stall does not escalate",
                CompletionMode.SCREEN_VERIFIED,
                {"stall": StallState.FLOWING},
                0,
                AdvanceKind.RETAIN,
                RetainReason.MISSING_CLAIM,
                True,
            ),
            (
                "advance never escalates",
                CompletionMode.TERMINAL,
                {"dispatched": True},
                5,
                AdvanceKind.ADVANCE,
                None,
                True,
            ),
            (
                "satisfied-prior never escalates",
                CompletionMode.OUTCOME_VERIFIED,
                {"verdict": CriterionVerdict.SATISFIED, "confidence": 0.9},
                5,
                AdvanceKind.SATISFIED_PRIOR,
                None,
                True,
            ),
        )
        self.__check(rows=rows)

    def __check(self, *, rows: Tuple[Row, ...]) -> None:
        """
        Assert each row against the exact decision triple.
        """

        for name, completion, facets, history, kind, reason, redispatch in rows:
            with self.subTest(row=name):
                decision = self.__decide(completion=completion, history=history, facets=facets)
                self.assertEqual(decision.kind, kind)
                self.assertEqual(decision.reason, reason)
                self.assertEqual(decision.redispatch, redispatch)

    def __decide(self, *, completion: CompletionMode, history: int, facets: Facets) -> Advancement:
        """
        Run the policy for one task mode and evidence pattern at the given retain streak.
        """

        claim = bool(facets.get("claim", False))
        dispatched = bool(facets.get("dispatched", False))
        verdict = facets.get("verdict")
        effect = facets.get("effect")
        stall = facets.get("stall")
        raw_confidence = facets.get("confidence", 0.9)
        confidence = raw_confidence if isinstance(raw_confidence, float) else 0.9

        return self.policy.decide(
            task=Task(
                index=1,
                description="task",
                kind=SubGoalKind.ACTION,
                completion=completion,
            ),
            evidence=TurnEvidence(
                claim=ClaimEvidence(asserted=claim),
                action=ActionEvidence(dispatched=dispatched, executed=dispatched),
                effect=(
                    EffectReading(live=effect, trial=effect)
                    if isinstance(effect, ActionEffectStatus)
                    else None
                ),
                verdict=(
                    Verdict(outcome=verdict, confidence=confidence)
                    if isinstance(verdict, CriterionVerdict)
                    else None
                ),
                validation=(
                    Validation(subject="cart shows the item", source=ValidationSource.STATE)
                    if facets.get("validation")
                    else None
                ),
                stall=(
                    StallSignal(state=stall, streak=3) if isinstance(stall, StallState) else None
                ),
            ),
            history=RetainHistory(consecutive=history),
        )
