from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants.assessment import PhaseComparison, PhaseIncomparability
from fathom.constants.turn.advancement import AdvanceKind
from fathom.schemas.advancement import Advancement
from fathom.schemas.planner import PlannerMetrics
from fathom.schemas.shadow import (
    ComparablePhase,
    IncomparablePhase,
    ShadowApplication,
    ShadowExecution,
    ShadowPostDispatch,
)
from fathom.schemas.target import TargetAuthority


def _advance() -> Advancement:
    """
    Build a minimal advancing decision.
    """

    return Advancement(kind=AdvanceKind.ADVANCE)


def _retain() -> Advancement:
    """
    Build a minimal retaining decision.
    """

    return Advancement(kind=AdvanceKind.RETAIN)


class ShadowPhaseUnionTest(unittest.TestCase):
    """
    Comparability is a typed phase variant, never a caller-supplied boolean.
    """

    def test_a_phase_cannot_be_constructed_with_an_arbitrary_comparability_flag(self) -> None:
        """
        No phase accepts a boolean comparability field; comparability is the variant itself.
        """

        with self.assertRaises(ValidationError):
            ComparablePhase(candidate=_advance(), live=_advance(), comparable=True)
        with self.assertRaises(ValidationError):
            IncomparablePhase(
                candidate=_advance(),
                live=_retain(),
                reason=PhaseIncomparability.EXECUTION_FAILED,
                comparable=False,
            )

    def test_an_incomparable_phase_requires_a_typed_reason(self) -> None:
        """
        An incomparable phase cannot exist without a typed reason from the enum.
        """

        with self.assertRaises(ValidationError):
            IncomparablePhase(candidate=_advance(), live=_retain())
        with self.assertRaises(ValidationError):
            IncomparablePhase(candidate=_advance(), live=_retain(), reason="made-up")

    def test_only_a_comparable_phase_exposes_divergence(self) -> None:
        """
        Divergence is meaningful only for comparable phases; incomparable phases never expose it.
        """

        self.assertTrue(ComparablePhase(candidate=_advance(), live=_retain()).diverges)
        self.assertFalse(ComparablePhase(candidate=_advance(), live=_advance()).diverges)
        incomparable = IncomparablePhase(
            candidate=_advance(),
            live=_retain(),
            reason=PhaseIncomparability.EVIDENCE_SOURCE_DIFFERENT,
        )
        self.assertFalse(hasattr(incomparable, "diverges"))

    def test_the_phase_discriminator_round_trips_by_kind(self) -> None:
        """
        A serialized post-dispatch phase reloads into the exact variant named by its discriminator.
        """

        comparable = ShadowPostDispatch(
            screen="post",
            foreground="app",
            phase=ComparablePhase(candidate=_advance(), live=_advance()),
        )
        reloaded = ShadowPostDispatch.model_validate(comparable.model_dump())
        self.assertIsInstance(reloaded.phase, ComparablePhase)
        self.assertEqual(reloaded.phase.kind, PhaseComparison.COMPARABLE)


class ShadowExecutionBoundaryTest(unittest.TestCase):
    """
    Execution owns only the receipt; post-dispatch provenance lives on ShadowPostDispatch.
    """

    def test_execution_cannot_carry_screen_or_foreground(self) -> None:
        """
        The execution receipt carries no observation provenance.
        """

        self.assertEqual(tuple(ShadowExecution.model_fields), ("receipt",))
        with self.assertRaises(ValidationError):
            ShadowExecution(receipt=None, screen="post")
        with self.assertRaises(ValidationError):
            ShadowExecution(receipt=None, foreground="app")

    def test_post_dispatch_owns_screen_foreground_and_phase(self) -> None:
        """
        Post-dispatch provenance and its comparison phase belong to ShadowPostDispatch.
        """

        post = ShadowPostDispatch(
            screen="post",
            foreground="app",
            phase=IncomparablePhase(
                candidate=_advance(),
                live=_retain(),
                reason=PhaseIncomparability.VISUAL_EVIDENCE_DEFERRED,
            ),
        )
        self.assertEqual(post.screen, "post")
        self.assertEqual(post.foreground, "app")
        self.assertIsInstance(post.phase, IncomparablePhase)


class ShadowIdentifierProvenanceTest(unittest.TestCase):
    """
    Optional identifiers may be absent, but a present-but-blank identifier is invalid provenance.
    """

    def test_optional_post_identifiers_reject_blank(self) -> None:
        """
        An optional post-dispatch screen or foreground is either absent or non-blank, never blank.
        """

        phase = ComparablePhase(candidate=_advance(), live=_advance())
        self.assertIsNone(ShadowPostDispatch(phase=phase).screen)
        with self.assertRaises(ValidationError):
            ShadowPostDispatch(screen="   ", phase=phase)
        with self.assertRaises(ValidationError):
            ShadowPostDispatch(foreground="", phase=phase)

    def test_optional_application_foreground_rejects_blank(self) -> None:
        """
        An optional application foreground is either absent or non-blank.
        """

        self.assertIsNone(ShadowApplication(authority=TargetAuthority.unbound()).foreground)
        with self.assertRaises(ValidationError):
            ShadowApplication(authority=TargetAuthority.unbound(), foreground="  ")


class PlannerMetricsInvariantTest(unittest.TestCase):
    """
    Measured planner metrics always record at least one call; absence is never a zero-valued measurement.
    """

    def test_metrics_reject_a_zero_call_measurement(self) -> None:
        """
        A produced analysis ran at least once, so zero calls can never be recorded as telemetry.
        """

        with self.assertRaises(ValidationError):
            PlannerMetrics(latency=0.0, calls=0)
        self.assertEqual(PlannerMetrics(latency=0.0, calls=1).calls, 1)
