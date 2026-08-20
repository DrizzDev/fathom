from __future__ import annotations

from typing import Any, Optional

from fathom.constants import ActionType
from fathom.constants.assessment import VisualVerdict
from fathom.constants.flow import SwipeDirection
from fathom.constants.success import CaptureNameProvenance
from fathom.constants.turn.advancement import AdvanceKind, ObservationPhase
from fathom.constants.turn.binding import BindingState
from fathom.constants.turn.stall import StallState
from fathom.core.agent.advancement import AdvancementPolicy
from fathom.schemas.actions import Action
from fathom.schemas.assessment import VisualAssessment
from fathom.schemas.binding import Binding
from fathom.schemas.capture import Capture, CaptureIdentity, CaptureRequest
from fathom.schemas.completion import ActionEvidence, ClaimEvidence
from fathom.schemas.criterion import CriterionVerdict, Verdict
from fathom.schemas.requirement import (
    CommandRequirement,
    PressRequirement,
    SwipeRequirement,
    TypeRequirement,
    WaitRequirement,
)
from fathom.schemas.stall import StallSignal
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.success import (
    CaptureSuccess,
    CommandSuccess,
    ObservationRequirement,
    ObservedSuccess,
    SourceLocation,
    SourceSpan,
    Success,
)
from fathom.schemas.target import TargetAuthority
from fathom.schemas.turn import TurnEvidence
from fathom.schemas.visual import VisualEvidence


def _obs(assertion: str) -> ObservationRequirement:
    return ObservationRequirement(assertion=assertion)


def _source(quote: str, intent: str) -> SourceSpan:
    start = intent.find(quote)
    return SourceSpan(quote=quote, location=SourceLocation(start=start, end=start + len(quote)))


def _command_success(
    requirement: CommandRequirement,
    *,
    postcondition: Optional[ObservationRequirement] = None,
) -> CommandSuccess:
    return CommandSuccess(
        requirement=requirement,
        source=_source(quote="tap", intent="tap the target"),
        postcondition=postcondition,
    )


def _action(action_type: ActionType, **kwargs: Any) -> Action:
    return Action(action_type=action_type, rationale="test", **kwargs)


def _step(action: Action, *, requirement: Optional[CommandRequirement] = None) -> Step:
    return Step(action=action, screen_hash="h", step_number=1, requirement=requirement)


def _result(step: Step, *, executed: bool = True, capture: Optional[Capture] = None) -> StepResult:
    return StepResult(
        step=step,
        success=True,
        executed=executed,
        capture=capture,
        pre_hash="a",
        post_hash="b",
        screen_changed=True,
        duration=0,
    )


def _turn(
    *,
    phase: ObservationPhase = ObservationPhase.POST_DISPATCH,
    execution: Optional[StepResult] = None,
    observation: Optional[ObservationRequirement] = None,
    verdict: Optional[Verdict] = None,
    stall: StallState = StallState.FLOWING,
    binding: Optional[Binding] = None,
) -> TurnEvidence:
    return TurnEvidence(
        claim=ClaimEvidence(asserted=False),
        action=ActionEvidence(dispatched=True, executed=execution is not None),
        phase=phase,
        execution=execution,
        observation=observation,
        verdict=verdict,
        binding=binding,
        stall=StallSignal(state=stall, streak=0),
    )


def _refuted() -> Verdict:
    return Verdict(outcome=CriterionVerdict.UNSATISFIED, confidence=0.95, evidence="absent")


def _bound() -> Binding:
    return Binding(state=BindingState.BOUND, confidence=1.0, anchor="element")


def _unbound() -> Binding:
    return Binding(state=BindingState.MISSING, confidence=0.0)


def _inferred() -> Binding:
    return Binding(state=BindingState.INFERRED, confidence=0.5, anchor="element")


def _satisfied() -> Verdict:
    return Verdict(outcome=CriterionVerdict.SATISFIED, confidence=0.95, evidence="seen")


def _decide(success: Success, evidence: TurnEvidence) -> AdvanceKind:
    return AdvancementPolicy().decide(success=success, evidence=evidence).kind


class TestCommandActionMatch:
    def test_correct_requirement_wrong_executed_operation_retains(self) -> None:
        requirement = PressRequirement(operation=ActionType.TAP, target="Login")
        success = _command_success(requirement)
        # Attached requirement equals success.requirement, but the executed action is a TYPE.
        action = _action(ActionType.TYPE, natural_language_target="Login", text="anything")
        evidence = _turn(execution=_result(_step(action, requirement=requirement)))
        assert _decide(success, evidence) is AdvanceKind.RETAIN

    def test_correct_operation_wrong_type_text_retains(self) -> None:
        requirement = TypeRequirement(operation=ActionType.TYPE, target="Search", text="hello")
        success = _command_success(requirement)
        action = _action(ActionType.TYPE, natural_language_target="Search", text="wrong")
        evidence = _turn(execution=_result(_step(action, requirement=requirement)))
        assert _decide(success, evidence) is AdvanceKind.RETAIN

    def test_free_text_target_mismatch_no_longer_gates(self) -> None:
        # Target descriptions are authored independently by decomposer and planner; a differing
        # free-text target is not completion authority. The operation matches, the target is bound,
        # so the command advances; semantic target correctness is the VLM's responsibility.
        requirement = PressRequirement(operation=ActionType.TAP, target="Login")
        success = _command_success(requirement)
        action = _action(ActionType.TAP, natural_language_target="Logout")
        evidence = _turn(
            execution=_result(_step(action, requirement=requirement)), binding=_bound()
        )
        assert _decide(success, evidence) is AdvanceKind.ADVANCE

    def test_unbound_target_retains(self) -> None:
        # The runtime-bound target replaces free-text equality: a command whose target did not
        # ground to a real on-screen element (MISSING binding) cannot complete.
        requirement = PressRequirement(operation=ActionType.TAP, target="Login")
        success = _command_success(requirement)
        action = _action(ActionType.TAP, natural_language_target="Login")
        evidence = _turn(
            execution=_result(_step(action, requirement=requirement)), binding=_unbound()
        )
        assert _decide(success, evidence) is AdvanceKind.RETAIN

    def test_bound_target_advances(self) -> None:
        requirement = PressRequirement(operation=ActionType.TAP, target="Login")
        success = _command_success(requirement)
        action = _action(ActionType.TAP, natural_language_target="Login")
        evidence = _turn(
            execution=_result(_step(action, requirement=requirement)), binding=_bound()
        )
        assert _decide(success, evidence) is AdvanceKind.ADVANCE

    def test_inferred_target_advances(self) -> None:
        # "Grounded target exists" is the bar, not firm anchoring: an inferred-geometry target still
        # advances, so vision-localized taps that work today are not regressed.
        requirement = PressRequirement(operation=ActionType.TAP, target="Login")
        success = _command_success(requirement)
        action = _action(ActionType.TAP, natural_language_target="Login")
        evidence = _turn(
            execution=_result(_step(action, requirement=requirement)), binding=_inferred()
        )
        assert _decide(success, evidence) is AdvanceKind.ADVANCE

    def test_wrong_swipe_direction_retains(self) -> None:
        requirement = SwipeRequirement(operation=ActionType.SWIPE, direction=SwipeDirection.UP)
        success = _command_success(requirement)
        action = _action(ActionType.SWIPE_DOWN)
        evidence = _turn(execution=_result(_step(action, requirement=requirement)))
        assert _decide(success, evidence) is AdvanceKind.RETAIN

    def test_free_text_swipe_surface_does_not_gate(self) -> None:
        # Surface descriptions are authored independently by decomposer and planner; only the finger
        # direction gates a swipe. A differing free-text surface does not block a matching swipe.
        requirement = SwipeRequirement(
            operation=ActionType.SWIPE, direction=SwipeDirection.UP, target="results list"
        )
        success = _command_success(requirement)
        action = _action(ActionType.SWIPE_UP, surface="other surface")
        evidence = _turn(
            execution=_result(_step(action, requirement=requirement)), binding=_bound()
        )
        assert _decide(success, evidence) is AdvanceKind.ADVANCE

    def test_wrong_wait_bound_retains(self) -> None:
        requirement = WaitRequirement(
            operation=ActionType.WAIT, condition="results appear", bound=5.0
        )
        success = _command_success(requirement)
        action = _action(ActionType.WAIT, wait_subject="results appear", wait_duration=9.0)
        evidence = _turn(execution=_result(_step(action, requirement=requirement)))
        assert _decide(success, evidence) is AdvanceKind.RETAIN

    def test_missing_dispatch_retains(self) -> None:
        requirement = PressRequirement(operation=ActionType.TAP, target="Login")
        success = _command_success(requirement)
        action = _action(ActionType.TAP, natural_language_target="Login")
        evidence = _turn(execution=_result(_step(action, requirement=requirement), executed=False))
        assert _decide(success, evidence) is AdvanceKind.RETAIN

    def test_matching_command_advances(self) -> None:
        requirement = PressRequirement(operation=ActionType.TAP, target="Login")
        success = _command_success(requirement)
        action = _action(ActionType.TAP, natural_language_target="Login")
        evidence = _turn(execution=_result(_step(action, requirement=requirement)))
        assert _decide(success, evidence) is AdvanceKind.ADVANCE


class TestObservationIdentity:
    def test_satisfied_verdict_for_other_assertion_cannot_complete(self) -> None:
        success = ObservedSuccess(observation=_obs("login screen displayed"))
        other = _obs("search results displayed")
        evidence = _turn(
            execution=_result(_step(_action(ActionType.TAP))),
            observation=other,
            verdict=_satisfied(),
        )
        assert _decide(success, evidence) is AdvanceKind.RETAIN

    def test_satisfied_own_observation_post_dispatch_advances(self) -> None:
        target = _obs("login screen displayed")
        success = ObservedSuccess(observation=target)
        evidence = _turn(
            execution=_result(_step(_action(ActionType.TAP))),
            observation=target,
            verdict=_satisfied(),
        )
        assert _decide(success, evidence) is AdvanceKind.ADVANCE

    def test_satisfied_own_observation_pre_dispatch_is_prior(self) -> None:
        target = _obs("login screen displayed")
        success = ObservedSuccess(observation=target)
        evidence = _turn(
            phase=ObservationPhase.PRE_DISPATCH,
            execution=None,
            observation=target,
            verdict=_satisfied(),
        )
        assert _decide(success, evidence) is AdvanceKind.SATISFIED_PRIOR

    def test_command_wrong_postcondition_verdict_retains(self) -> None:
        requirement = PressRequirement(operation=ActionType.TAP, target="Login")
        post = _obs("home screen shown")
        success = _command_success(requirement, postcondition=post)
        action = _action(ActionType.TAP, natural_language_target="Login")
        evidence = _turn(
            execution=_result(_step(action, requirement=requirement)),
            observation=post,
            verdict=Verdict(outcome=CriterionVerdict.UNSATISFIED, confidence=0.95, evidence="no"),
        )
        assert _decide(success, evidence) is AdvanceKind.RETAIN


class TestCaptureExecution:
    @staticmethod
    def _capture_success() -> CaptureSuccess:
        return CaptureSuccess(
            target=CaptureIdentity(name="price", provenance=CaptureNameProvenance.USER),
            subject="item price",
        )

    @staticmethod
    def _store_step() -> Step:
        action = _action(
            ActionType.STORE,
            capture=CaptureRequest(name="price", subject="item price", value="9.99"),
        )
        return _step(action)

    def test_successful_capture_with_executed_false_retains(self) -> None:
        capture = Capture(name="price", step=1, success=True, value="9.99")
        evidence = _turn(execution=_result(self._store_step(), executed=False, capture=capture))
        assert _decide(self._capture_success(), evidence) is AdvanceKind.RETAIN

    def test_matching_committed_capture_advances(self) -> None:
        capture = Capture(name="price", step=1, success=True, value="9.99")
        evidence = _turn(execution=_result(self._store_step(), executed=True, capture=capture))
        assert _decide(self._capture_success(), evidence) is AdvanceKind.ADVANCE

    def test_capture_advances_despite_free_text_subject_difference(self) -> None:
        # Regression (live Retail run): the planner phrases the STORE subject independently of the
        # decomposer's; identity is the NAME, so a differing free-text subject must still advance.
        action = _action(
            ActionType.STORE,
            capture=CaptureRequest(
                name="price", subject="the price shown on the product card", value="9.99"
            ),
        )
        capture = Capture(name="price", step=1, success=True, value="9.99")
        evidence = _turn(execution=_result(_step(action), executed=True, capture=capture))
        assert _decide(self._capture_success(), evidence) is AdvanceKind.ADVANCE


class TestConfidenceFloor:
    """
    Ported truth-table boundary: an observed verdict advances only at or above the confidence floor.
    """

    @staticmethod
    def _observed() -> ObservedSuccess:
        return ObservedSuccess(observation=_obs("login screen displayed"))

    def _turn_at(self, confidence: float) -> TurnEvidence:
        target = _obs("login screen displayed")
        return _turn(
            execution=_result(_step(_action(ActionType.TAP))),
            observation=target,
            verdict=Verdict(
                outcome=CriterionVerdict.SATISFIED, confidence=confidence, evidence="s"
            ),
        )

    def test_satisfied_at_floor_advances(self) -> None:
        assert _decide(self._observed(), self._turn_at(0.7)) is AdvanceKind.ADVANCE

    def test_satisfied_below_floor_retains(self) -> None:
        assert _decide(self._observed(), self._turn_at(0.69)) is AdvanceKind.RETAIN


class TestEscalation:
    """
    A stalled retain escalates, and a refuted own-observation is terminal: a sub-goal that cannot
    prove itself may not retain unboundedly once momentum reports a stall.
    """

    @staticmethod
    def _observed() -> ObservedSuccess:
        return ObservedSuccess(observation=_obs("login screen displayed"))

    def test_stalled_awaiting_proof_escalates(self) -> None:
        evidence = _turn(
            execution=_result(_step(_action(ActionType.TAP))),
            stall=StallState.STALLED,
        )
        assert _decide(self._observed(), evidence) is AdvanceKind.ESCALATE

    def test_stalled_refuted_own_observation_escalates_not_unsatisfiable(self) -> None:
        target = _obs("login screen displayed")
        evidence = _turn(
            execution=_result(_step(_action(ActionType.TAP))),
            observation=target,
            verdict=_refuted(),
            stall=StallState.STALLED,
        )
        # A vision refute may be a false negative, so a stalled refute escalates for help, never hard-kills.
        assert _decide(ObservedSuccess(observation=target), evidence) is AdvanceKind.ESCALATE

    def test_flowing_awaiting_proof_does_not_escalate(self) -> None:
        evidence = _turn(
            execution=_result(_step(_action(ActionType.TAP))),
            stall=StallState.FLOWING,
        )
        assert _decide(self._observed(), evidence) is AdvanceKind.RETAIN


class TestStallResolution:
    """
    A stalled command that cannot match its gesture advances when the outcome is observably satisfied
    (run e25975d0: five satisfied@1.0 oracle reads discarded, run killed), and escalates otherwise. The
    resolution reads real proof, never the stall alone, so it can never fabricate a pass; capture stays strict.
    """

    @staticmethod
    def _pending_command(
        *, postcondition: Optional[ObservationRequirement] = None
    ) -> CommandSuccess:
        return _command_success(
            PressRequirement(operation=ActionType.TAP, target="Login"), postcondition=postcondition
        )

    @staticmethod
    def _validating(
        *, verdict: Verdict, observation: Optional[ObservationRequirement] = None
    ) -> TurnEvidence:
        # A validate while a command is pending: the step carries no matching requirement, so the
        # command adjudication retains; the oracle verdict is the only outcome signal available.
        return _turn(
            execution=_result(_step(_action(ActionType.VALIDATE))),
            observation=observation,
            verdict=verdict,
            stall=StallState.STALLED,
        )

    def test_stalled_command_advances_on_satisfied_oracle(self) -> None:
        evidence = self._validating(verdict=_satisfied())
        assert _decide(self._pending_command(), evidence) is AdvanceKind.ADVANCE

    def test_stalled_command_advances_on_satisfied_postcondition(self) -> None:
        postcondition = _obs("the bill summary is displayed")
        evidence = self._validating(verdict=_satisfied(), observation=postcondition)
        assert (
            _decide(self._pending_command(postcondition=postcondition), evidence)
            is AdvanceKind.ADVANCE
        )

    def test_stalled_command_escalates_when_oracle_not_satisfied(self) -> None:
        evidence = self._validating(verdict=_refuted())
        assert _decide(self._pending_command(), evidence) is AdvanceKind.ESCALATE

    def test_flowing_command_does_not_advance_on_satisfied_oracle(self) -> None:
        # The safety net is a stall-time backstop: without a stall the command keeps its normal path.
        evidence = _turn(
            execution=_result(_step(_action(ActionType.VALIDATE))),
            verdict=_satisfied(),
            stall=StallState.FLOWING,
        )
        assert _decide(self._pending_command(), evidence) is AdvanceKind.RETAIN

    def test_stalled_capture_never_advances_on_satisfied_oracle(self) -> None:
        # A capture cannot be confirmed by observation: if nothing was stored, advancing would lose data.
        success = CaptureSuccess(
            target=CaptureIdentity(name="price", provenance=CaptureNameProvenance.USER),
            subject="the item price",
        )
        evidence = self._validating(verdict=_satisfied())
        assert _decide(success, evidence) is AdvanceKind.ESCALATE


class TestVisualAuthorityIsolation:
    """
    Regressions from run 7f64d39f10e440d5ad9334acf15094ce: a visual verdict is authority only for its
    own visual goal; it can neither advance nor veto command/capture evidence.
    """

    @staticmethod
    def _capture_success() -> CaptureSuccess:
        return CaptureSuccess(
            target=CaptureIdentity(name="item_price", provenance=CaptureNameProvenance.USER),
            subject="the first result price",
        )

    @staticmethod
    def _store_step() -> Step:
        return _step(
            _action(
                ActionType.STORE,
                capture=CaptureRequest(name="item_price", subject="price", value="4.99"),
            )
        )

    def test_satisfied_verdict_for_other_goal_cannot_advance(self) -> None:
        # "Open Retail" satisfied must not advance the "search ghar soap" goal.
        success = ObservedSuccess(observation=_obs("search results for ghar soap shown"))
        evidence = _turn(
            execution=_result(_step(_action(ActionType.TAP))),
            observation=_obs("Retail home screen shown"),
            verdict=_satisfied(),
        )
        assert _decide(success, evidence) is AdvanceKind.RETAIN

    def test_capture_goal_not_advanced_by_visual_verdict(self) -> None:
        # A satisfied visual verdict without a commit cannot complete a capture goal.
        evidence = _turn(
            execution=_result(self._store_step(), executed=True, capture=None),
            observation=_obs("price is visible on screen"),
            verdict=_satisfied(),
        )
        assert _decide(self._capture_success(), evidence) is AdvanceKind.RETAIN

    def test_committed_capture_not_vetoed_by_refuting_verdict(self) -> None:
        # The final-verifier contradiction at the policy layer: a committed capture advances even when
        # a visual judge refutes the screen; non-visual evidence cannot be vetoed by a visual judge.
        capture = Capture(name="item_price", step=1, success=True, value="4.99")
        evidence = _turn(
            execution=_result(self._store_step(), executed=True, capture=capture),
            observation=_obs("price is visible on screen"),
            verdict=_refuted(),
        )
        assert _decide(self._capture_success(), evidence) is AdvanceKind.ADVANCE

    def test_command_goal_not_advanced_by_visual_verdict(self) -> None:
        requirement = PressRequirement(operation=ActionType.TAP, target="Login")
        success = _command_success(requirement)
        # A satisfied visual verdict, but no matching executed command, cannot advance a command goal.
        evidence = _turn(
            execution=_result(_step(_action(ActionType.SCROLL, surface="list"))),
            observation=_obs("login button visible"),
            verdict=_satisfied(),
        )
        assert _decide(success, evidence) is AdvanceKind.RETAIN


class TestVisualEvidenceAdvancement:
    """
    Pins the additive visual-evidence branch: an observed goal advances from a settled-screen assessment,
    and a command postcondition advances only from a receipt combined with a later matching assessment.
    """

    @staticmethod
    def __visual(
        *,
        observation: ObservationRequirement,
        verdict: Optional[VisualVerdict],
        action_present: bool = False,
        malformed: bool = False,
        authority: Optional[TargetAuthority] = None,
        foreground: Optional[str] = None,
        confidence: float = 0.9,
    ) -> VisualEvidence:
        """
        Build settled-screen visual evidence for the given observation.
        """

        assessment = (
            None
            if verdict is None or malformed
            else VisualAssessment(verdict=verdict, confidence=confidence, evidence="cited")
        )
        return VisualEvidence(
            observation=observation,
            assessment=assessment,
            malformed=malformed,
            phase=ObservationPhase.POST_DISPATCH,
            action_present=action_present,
            screen="screen-1",
            authority=authority if authority is not None else TargetAuthority.unbound(),
            foreground=foreground,
        )

    @staticmethod
    def __turn(
        *,
        observation: ObservationRequirement,
        visual: VisualEvidence,
        execution: Optional[StepResult] = None,
        binding: Optional[Binding] = None,
    ) -> TurnEvidence:
        """
        Build a post-dispatch turn carrying visual evidence and an optional executed receipt.
        """

        return TurnEvidence(
            claim=ClaimEvidence(asserted=False),
            action=ActionEvidence(dispatched=execution is not None, executed=execution is not None),
            phase=ObservationPhase.POST_DISPATCH,
            execution=execution,
            observation=observation,
            binding=binding,
            visual=visual,
            stall=StallSignal(state=StallState.FLOWING, streak=0),
        )

    def test_observed_satisfied_action_free_advances(self) -> None:
        """
        A satisfied, action-free assessment of the goal's own observation advances it.
        """

        obs = _obs("search results for ghar soap shown")
        evidence = self.__turn(
            observation=obs, visual=self.__visual(observation=obs, verdict=VisualVerdict.SATISFIED)
        )
        assert _decide(ObservedSuccess(observation=obs), evidence) is AdvanceKind.ADVANCE

    def test_observed_satisfied_with_action_retains(self) -> None:
        """
        A satisfied verdict accompanied by a proposed action is contradictory and retains.
        """

        obs = _obs("search results for ghar soap shown")
        evidence = self.__turn(
            observation=obs,
            visual=self.__visual(
                observation=obs, verdict=VisualVerdict.SATISFIED, action_present=True
            ),
        )
        assert _decide(ObservedSuccess(observation=obs), evidence) is AdvanceKind.RETAIN

    def test_observed_bound_matching_foreground_advances(self) -> None:
        """
        A satisfied verdict on the exact bound foreground package advances.
        """

        obs = _obs("Retail home screen shown")
        evidence = self.__turn(
            observation=obs,
            visual=self.__visual(
                observation=obs,
                verdict=VisualVerdict.SATISFIED,
                authority=TargetAuthority.requested(package="com.retail.mShop.android.shopping"),
                foreground="com.retail.mShop.android.shopping",
            ),
        )
        assert _decide(ObservedSuccess(observation=obs), evidence) is AdvanceKind.ADVANCE

    def test_observed_bound_mismatched_foreground_retains(self) -> None:
        """
        A satisfied verdict on a foreground other than the bound target retains.
        """

        obs = _obs("Retail home screen shown")
        evidence = self.__turn(
            observation=obs,
            visual=self.__visual(
                observation=obs,
                verdict=VisualVerdict.SATISFIED,
                authority=TargetAuthority.requested(package="com.retail.mShop.android.shopping"),
                foreground="com.android.launcher",
            ),
        )
        assert _decide(ObservedSuccess(observation=obs), evidence) is AdvanceKind.RETAIN

    def test_observed_bound_missing_foreground_fails_closed(self) -> None:
        """
        A satisfied verdict under a bound target with an unknown foreground fails closed and retains.
        """

        obs = _obs("Retail home screen shown")
        evidence = self.__turn(
            observation=obs,
            visual=self.__visual(
                observation=obs,
                verdict=VisualVerdict.SATISFIED,
                authority=TargetAuthority.requested(package="com.retail.mShop.android.shopping"),
                foreground=None,
            ),
        )
        assert _decide(ObservedSuccess(observation=obs), evidence) is AdvanceKind.RETAIN

    def test_observed_not_satisfied_retains(self) -> None:
        """
        A withheld verdict retains the goal.
        """

        obs = _obs("search results for ghar soap shown")
        evidence = self.__turn(
            observation=obs,
            visual=self.__visual(observation=obs, verdict=VisualVerdict.NOT_SATISFIED),
        )
        assert _decide(ObservedSuccess(observation=obs), evidence) is AdvanceKind.RETAIN

    def test_observed_malformed_retains(self) -> None:
        """
        A schema-failed assessment retains the goal.
        """

        obs = _obs("search results for ghar soap shown")
        evidence = self.__turn(
            observation=obs, visual=self.__visual(observation=obs, verdict=None, malformed=True)
        )
        assert _decide(ObservedSuccess(observation=obs), evidence) is AdvanceKind.RETAIN

    def test_observed_satisfied_at_floor_advances(self) -> None:
        """
        A satisfied assessment exactly at the confidence floor confirms and advances.
        """

        obs = _obs("search results for ghar soap shown")
        evidence = self.__turn(
            observation=obs,
            visual=self.__visual(observation=obs, verdict=VisualVerdict.SATISFIED, confidence=0.7),
        )
        assert _decide(ObservedSuccess(observation=obs), evidence) is AdvanceKind.ADVANCE

    def test_observed_below_floor_retains(self) -> None:
        """
        Vision is the live authority: a satisfied verdict below the confidence floor cannot advance.
        """

        obs = _obs("search results for ghar soap shown")
        evidence = self.__turn(
            observation=obs,
            visual=self.__visual(observation=obs, verdict=VisualVerdict.SATISFIED, confidence=0.69),
        )
        assert _decide(ObservedSuccess(observation=obs), evidence) is AdvanceKind.RETAIN

    def test_command_postcondition_receipt_and_assessment_advances(self) -> None:
        """
        A command with a postcondition advances when a matching receipt and a satisfied assessment both hold.
        """

        requirement = PressRequirement(operation=ActionType.TAP, target="Login")
        post = _obs("the login dialog closed")
        success = _command_success(requirement, postcondition=post)
        action = _action(ActionType.TAP, natural_language_target="Login")
        evidence = self.__turn(
            observation=post,
            visual=self.__visual(observation=post, verdict=VisualVerdict.SATISFIED),
            execution=_result(_step(action, requirement=requirement)),
            binding=_bound(),
        )
        assert _decide(success, evidence) is AdvanceKind.ADVANCE

    def test_command_postcondition_receipt_without_assessment_retains(self) -> None:
        """
        A command postcondition with a matching receipt but an unsatisfied assessment retains.
        """

        requirement = PressRequirement(operation=ActionType.TAP, target="Login")
        post = _obs("the login dialog closed")
        success = _command_success(requirement, postcondition=post)
        action = _action(ActionType.TAP, natural_language_target="Login")
        evidence = self.__turn(
            observation=post,
            visual=self.__visual(observation=post, verdict=VisualVerdict.NOT_SATISFIED),
            execution=_result(_step(action, requirement=requirement)),
            binding=_bound(),
        )
        assert _decide(success, evidence) is AdvanceKind.RETAIN

    def test_command_postcondition_assessment_without_receipt_retains(self) -> None:
        """
        A command postcondition with a satisfied assessment but no executed receipt retains.
        """

        requirement = PressRequirement(operation=ActionType.TAP, target="Login")
        post = _obs("the login dialog closed")
        success = _command_success(requirement, postcondition=post)
        evidence = self.__turn(
            observation=post,
            visual=self.__visual(observation=post, verdict=VisualVerdict.SATISFIED),
        )
        assert _decide(success, evidence) is AdvanceKind.RETAIN
