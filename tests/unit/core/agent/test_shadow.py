from __future__ import annotations

from fathom.constants import ActionType
from fathom.constants.assessment import ShadowDivergenceKind, VisualVerdict
from fathom.constants.success import CaptureNameProvenance
from fathom.core.agent.shadow import ShadowAssessor
from fathom.schemas.assessment import VisualAssessment
from fathom.schemas.capture import CaptureIdentity
from fathom.schemas.requirement import PressRequirement
from fathom.schemas.success import (
    CaptureSuccess,
    CommandSuccess,
    ObservationRequirement,
    ObservedSuccess,
    SourceLocation,
    SourceSpan,
)
from fathom.schemas.target import TargetAuthority


def _observed(*, assertion: str = "Amazon home screen shown") -> ObservedSuccess:
    return ObservedSuccess(observation=ObservationRequirement(assertion=assertion))


def _command() -> CommandSuccess:
    return CommandSuccess(
        requirement=PressRequirement(operation=ActionType.TAP, target="Login"),
        source=SourceSpan(quote="tap", location=SourceLocation(start=0, end=3)),
    )


def _capture() -> CaptureSuccess:
    return CaptureSuccess(
        target=CaptureIdentity(name="price", provenance=CaptureNameProvenance.USER),
        subject="item price",
    )


def _assessment(verdict: VisualVerdict) -> VisualAssessment:
    return VisualAssessment(verdict=verdict, confidence=0.9, evidence="seen")


def _kinds(divergences: tuple) -> set:
    return {d.kind for d in divergences}


class TestGoalTypeRelationship:
    def test_observed_goal_without_assessment_is_missing(self) -> None:
        out = ShadowAssessor().assess(success=_observed(), assessment=None, action_present=False)
        assert _kinds(out) == {ShadowDivergenceKind.MISSING_ASSESSMENT}

    def test_command_goal_with_assessment_is_wrong_goal(self) -> None:
        out = ShadowAssessor().assess(
            success=_command(), assessment=_assessment(VisualVerdict.SATISFIED), action_present=True
        )
        assert _kinds(out) == {ShadowDivergenceKind.WRONG_GOAL}

    def test_capture_goal_with_assessment_is_wrong_goal(self) -> None:
        out = ShadowAssessor().assess(
            success=_capture(),
            assessment=_assessment(VisualVerdict.NOT_SATISFIED),
            action_present=False,
        )
        assert _kinds(out) == {ShadowDivergenceKind.WRONG_GOAL}

    def test_command_goal_without_assessment_is_clean(self) -> None:
        out = ShadowAssessor().assess(success=_command(), assessment=None, action_present=True)
        assert out == ()


class TestSatisfiedWithActionIsRecordedNotEnforced:
    def test_satisfied_with_action_is_a_divergence(self) -> None:
        # Slice 2 records the conflict; the action still executes live and no goal advances.
        out = ShadowAssessor().assess(
            success=_observed(),
            assessment=_assessment(VisualVerdict.SATISFIED),
            action_present=True,
        )
        assert ShadowDivergenceKind.SATISFIED_WITH_ACTION in _kinds(out)

    def test_satisfied_without_action_is_clean(self) -> None:
        out = ShadowAssessor().assess(
            success=_observed(),
            assessment=_assessment(VisualVerdict.SATISFIED),
            action_present=False,
        )
        assert out == ()


class TestPackageAuthorityContradiction:
    __SHOPPING = "com.amazon.mShop.android.shopping"

    def test_amazon_music_does_not_satisfy_amazon_shopping(self) -> None:
        out = ShadowAssessor().assess(
            success=_observed(assertion="Open Amazon"),
            assessment=_assessment(VisualVerdict.SATISFIED),
            action_present=False,
            authority=TargetAuthority.requested(package=self.__SHOPPING),
            foreground_package="com.amazon.mp3",
        )
        assert _kinds(out) == {ShadowDivergenceKind.PACKAGE_CONTRADICTION}

    def test_exact_authoritative_package_matches(self) -> None:
        out = ShadowAssessor().assess(
            success=_observed(assertion="Open Amazon"),
            assessment=_assessment(VisualVerdict.SATISFIED),
            action_present=False,
            authority=TargetAuthority.requested(package=self.__SHOPPING),
            foreground_package=self.__SHOPPING,
        )
        assert out == ()

    def test_prefix_is_not_package_identity(self) -> None:
        # P0 regression: package identity is exact, never a prefix. ``com.amazon.app.evil``
        # must not satisfy an authority bound to ``com.amazon.app``.
        out = ShadowAssessor().assess(
            success=_observed(),
            assessment=_assessment(VisualVerdict.SATISFIED),
            action_present=False,
            authority=TargetAuthority.requested(package="com.amazon.app"),
            foreground_package="com.amazon.app.evil",
        )
        assert _kinds(out) == {ShadowDivergenceKind.PACKAGE_CONTRADICTION}

    def test_unbound_authority_never_contradicts(self) -> None:
        out = ShadowAssessor().assess(
            success=_observed(assertion="Open Amazon"),
            assessment=_assessment(VisualVerdict.SATISFIED),
            action_present=False,
            authority=TargetAuthority.unbound(),
            foreground_package="com.amazon.mp3",
        )
        assert out == ()


class TestTruthReconciliation:
    def test_satisfied_but_oracle_false_is_false_positive(self) -> None:
        out = ShadowAssessor().assess(
            success=_observed(),
            assessment=_assessment(VisualVerdict.SATISFIED),
            action_present=False,
            truth_satisfied=False,
        )
        assert _kinds(out) == {ShadowDivergenceKind.FALSE_POSITIVE}

    def test_withheld_but_oracle_true_is_false_negative(self) -> None:
        out = ShadowAssessor().assess(
            success=_observed(),
            assessment=_assessment(VisualVerdict.UNCLEAR),
            action_present=False,
            truth_satisfied=True,
        )
        assert _kinds(out) == {ShadowDivergenceKind.FALSE_NEGATIVE}
