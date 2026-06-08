from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.schemas.escalation import (
    EscalationDecision,
    EscalationPolicy,
    EscalationReason,
    StuckSource,
)


class EscalationPolicyTest(unittest.TestCase):
    """
    Pins :class:`EscalationPolicy` defaults and bounds.
    """

    def test_defaults_match_documented_values(self) -> None:
        """
        Production-facing defaults: gate enabled, deferral_limit=2, passive_tolerance=3.
        """

        policy = EscalationPolicy()
        self.assertTrue(policy.enabled)
        self.assertEqual(policy.deferral_limit, 2)
        self.assertEqual(policy.passive_tolerance, 3)

    def test_is_frozen(self) -> None:
        """
        Policy must be immutable so callers cannot mutate shared instances.
        """

        policy = EscalationPolicy()
        with self.assertRaises(ValidationError):
            policy.enabled = False  # type: ignore[misc]

    def test_deferral_limit_lower_bound_enforced(self) -> None:
        """
        ``deferral_limit`` must be non-negative.
        """

        with self.assertRaises(ValidationError):
            EscalationPolicy(deferral_limit=-1)

    def test_deferral_limit_upper_bound_enforced(self) -> None:
        """
        ``deferral_limit`` must stay within the documented 0..10 range.
        """

        with self.assertRaises(ValidationError):
            EscalationPolicy(deferral_limit=11)

    def test_passive_tolerance_lower_bound_enforced(self) -> None:
        """
        ``passive_tolerance`` must be at least 1.
        """

        with self.assertRaises(ValidationError):
            EscalationPolicy(passive_tolerance=0)

    def test_passive_tolerance_upper_bound_enforced(self) -> None:
        """
        ``passive_tolerance`` must stay within the documented 1..10 range.
        """

        with self.assertRaises(ValidationError):
            EscalationPolicy(passive_tolerance=11)


class EscalationDecisionTest(unittest.TestCase):
    """
    Pins :class:`EscalationDecision` field shape.
    """

    def test_decision_is_frozen(self) -> None:
        """
        Decisions must be immutable so telemetry can hold references safely.
        """

        decision = EscalationDecision(
            allow=False,
            reason=EscalationReason.PASSIVE_RUN,
            stuck_source=StuckSource.LOOP_DETECTOR,
            deferrals=0,
        )
        with self.assertRaises(ValidationError):
            decision.allow = True  # type: ignore[misc]

    def test_deferrals_non_negative_enforced(self) -> None:
        """
        Negative deferral counts are not representable.
        """

        with self.assertRaises(ValidationError):
            EscalationDecision(
                allow=False,
                reason=EscalationReason.PASSIVE_RUN,
                stuck_source=StuckSource.LOOP_DETECTOR,
                deferrals=-1,
            )

    def test_message_optional(self) -> None:
        """
        ``message`` defaults to None when no human-readable detail is provided.
        """

        decision = EscalationDecision(
            allow=True,
            reason=EscalationReason.DISABLED,
            stuck_source=StuckSource.LOOP_DETECTOR,
            deferrals=0,
        )
        self.assertIsNone(decision.message)


class EscalationReasonTest(unittest.TestCase):
    """
    Pins :class:`EscalationReason` token stability for telemetry consumers.
    """

    def test_tokens_are_stable_strings(self) -> None:
        """
        Telemetry consumers depend on these exact values; do not rename.
        """

        self.assertEqual(EscalationReason.DISABLED.value, "disabled")
        self.assertEqual(EscalationReason.DEFERRAL_LIMIT.value, "deferral_limit")
        self.assertEqual(EscalationReason.SUBGOAL_BUDGET.value, "subgoal_budget")
        self.assertEqual(EscalationReason.ACTIVE_STALL.value, "active_stall")
        self.assertEqual(EscalationReason.PASSIVE_LIMIT.value, "passive_limit")
        self.assertEqual(EscalationReason.PASSIVE_RUN.value, "passive_run")


class StuckSourceTest(unittest.TestCase):
    """
    Pins :class:`StuckSource` token stability.
    """

    def test_tokens_are_stable_strings(self) -> None:
        """
        Source values are documented telemetry fields.
        """

        self.assertEqual(StuckSource.LOOP_DETECTOR.value, "loop_detector")
        self.assertEqual(StuckSource.SUBGOAL_BUDGET.value, "subgoal_budget")
