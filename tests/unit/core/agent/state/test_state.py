from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.constants.retries import (
    DEFAULT_PLANNER_RETRY_LIMIT,
    RetryBranch,
    RetryKind,
)
from fathom.constants.runtime import DEFAULT_VERIFICATION_REJECTION_LIMIT
from fathom.core.agent.state import AgentState
from fathom.schemas.actions import Action
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.retries import RetryLimits
from fathom.schemas.screens import ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.supervision import BlockReason
from tests.builders import SubGoalFixtures


class AgentStateContinuationTest(unittest.TestCase):
    """
    Covers continuation policy across autonomous and HITL runtimes via state.capabilities.
    """

    @staticmethod
    def __screen() -> ScreenState:
        """
        Return a stable screen used to create loop-detector evidence.
        """

        return ScreenState(
            timestamp=0,
            activity="app",
            visual_hash="b" * 16,
            activity_hash="a" * 16,
        )

    def test_hitl_capability_can_continue_after_autonomous_recovery_exhausted(self) -> None:
        """
        HITL-capable runtime is governed by the realignment budget, not the
        autonomous recovery budget. The capability flag on AgentState picks
        the correct stuck-rescue budget.
        """

        autonomous = AgentState(
            intent="complete onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        detector = autonomous.runtime.screen.detector

        for _ in range(detector.threshold):
            detector.record(
                action_type="tap",
                screen=self.__screen(),
                action_description="Tap Continue",
            )

        self.assertTrue(autonomous.is_stuck)

        while detector.can_recover():
            detector.record_recovery_attempt()

        self.assertFalse(autonomous.can_continue)

        hitl = AgentState(
            intent="complete onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )
        hitl_detector = hitl.runtime.screen.detector

        for _ in range(hitl_detector.threshold):
            hitl_detector.record(
                action_type="tap",
                screen=self.__screen(),
                action_description="Tap Continue",
            )
        while hitl_detector.can_recover():
            hitl_detector.record_recovery_attempt()

        self.assertTrue(hitl.can_continue)

    def test_from_checkpoint_uses_live_capabilities_not_stale(self) -> None:
        """
        Restoration must bind capabilities from the resuming runtime, not the checkpoint.
        """

        original = AgentState(
            intent="x",
            max_steps=7,
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )
        original.mark_complete(reason="done")
        payload = original.to_checkpoint()

        self.assertNotIn("hitl", payload)
        self.assertNotIn("capabilities", payload)

        restored = AgentState.from_checkpoint(
            payload,
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        self.assertTrue(restored.is_complete)
        self.assertEqual(restored.intent, "x")
        self.assertFalse(restored.capabilities.hitl.enabled)
        self.assertEqual(restored.completion_reason, "done")

    def test_from_checkpoint_requires_capabilities_argument(self) -> None:
        """
        from_checkpoint must reject a missing capabilities kwarg fail-fast.
        """

        payload = AgentState(
            intent="x",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        ).to_checkpoint()

        with self.assertRaises(TypeError):
            AgentState.from_checkpoint(payload)  # type: ignore[call-arg]

    def test_hitl_capability_stops_when_realignment_budget_is_exhausted(self) -> None:
        """
        HITL-capable runtime terminates once the realignment budget is exhausted.
        """

        state = AgentState(
            realignment_budget=1,
            intent="complete onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )
        detector = state.runtime.screen.detector

        for _ in range(detector.threshold):
            detector.record(
                action_type="tap",
                screen=self.__screen(),
                action_description="Tap Continue",
            )

        state.bump_realignment_budget()

        self.assertTrue(state.is_stuck)
        self.assertFalse(state.can_continue)


class AgentStateSubGoalCursorTest(unittest.TestCase):
    """
    Pins sub-goal cursor helpers used by VERIFY handoff logic.
    """

    @staticmethod
    def __caps() -> RuntimeCapabilities:
        """
        Return autonomous runtime capabilities for cursor tests.
        """

        return RuntimeCapabilities(hitl=HITLCapability(enabled=False))

    @staticmethod
    def __completion_signal() -> SubGoalCompletionSignal:
        """
        Return a valid completion signal for advancing the cursor.
        """

        return SubGoalCompletionSignal(
            llm_confidence=1.0,
            screen_verified=True,
            action_executed=True,
            flagged_complete=True,
            rationale_verified=True,
            evidence="unit test",
        )

    def test_current_sub_goal_final_is_false_without_sub_goals(self) -> None:
        """
        Empty sub-goal plans are never treated as final active work.
        """

        state = AgentState(intent="finish checkout", capabilities=self.__caps())

        self.assertFalse(state.has_active_final_sub_goal())

    def test_current_sub_goal_final_tracks_cursor_before_and_after_advance(self) -> None:
        """
        The helper is true only when the active cursor points at the terminal sub-goal.
        """

        state = AgentState(intent="finish checkout", capabilities=self.__caps())
        state.set_sub_goals(
            [
                SubGoal(index=0, description="Open cart"),
                SubGoal(index=1, description="Confirm checkout"),
            ]
        )

        self.assertFalse(state.has_active_final_sub_goal())

        state.mark_current_sub_goal_complete(completion_signal=self.__completion_signal())

        self.assertTrue(state.has_active_final_sub_goal())

    def test_current_sub_goal_final_is_false_after_all_sub_goals_complete(self) -> None:
        """
        Once the cursor moves past the terminal sub-goal, no active final sub-goal remains.
        """

        state = AgentState(intent="finish checkout", capabilities=self.__caps())
        state.set_sub_goals([SubGoal(index=0, description="Confirm checkout")])

        self.assertTrue(state.has_active_final_sub_goal())

        state.mark_current_sub_goal_complete(completion_signal=self.__completion_signal())

        self.assertIsNone(state.get_current_sub_goal())
        self.assertFalse(state.has_active_final_sub_goal())


class AgentStateLastActionPersistenceTest(unittest.TestCase):
    """
    Behavioral pins for the last-action descriptor that the LoopDetector consumes via update_screen on the next graph turn.
    """

    @staticmethod
    def __caps() -> RuntimeCapabilities:
        """
        Return autonomous capabilities — HITL is irrelevant for these tests.
        """

        return RuntimeCapabilities(hitl=HITLCapability(enabled=False))

    @staticmethod
    def __screen(*, visual_hash: str = "a" * 16) -> ScreenState:
        """
        Return a stable :class:`ScreenState` for recording.
        """

        return ScreenState(
            timestamp=0,
            activity="app",
            activity_hash="0" * 16,
            visual_hash=visual_hash,
        )

    @staticmethod
    def __tap_action(*, target: str) -> Action:
        """
        Return a tap :class:`Action` with the requested target descriptor.
        """

        return Action(
            target=target,
            confidence=1.0,
            rationale="test fixture",
            action_type=ActionType.TAP,
        )

    def test_last_action_round_trips_through_checkpoint(self) -> None:
        """``to_checkpoint`` / ``from_checkpoint`` must preserve ``__last_action_type`` and ``__last_action_description`` so a graph-state restore between nodes does not wipe the descriptor the LoopDetector will consume on the next ``update_screen``."""

        state = AgentState(intent="test", capabilities=self.__caps())

        state.record_attempt(
            reason="test_seed",
            action=self.__tap_action(target="Play button"),
        )

        checkpoint = state.to_checkpoint()
        restored = AgentState.from_checkpoint(checkpoint, capabilities=self.__caps())

        self.assertEqual(restored.last_action_type, "tap")
        self.assertEqual(
            restored._AgentState__last_action_description,  # type: ignore[attr-defined]
            "Tap on Play button",
        )

    def test_legacy_checkpoint_without_last_action_keys_restores_to_none(self) -> None:
        """
        Old checkpoints written before this change have no ``last_action_*`` keys.
        Restore must default to ``None`` to match the original first-turn semantics.
        """

        state = AgentState(intent="test", capabilities=self.__caps())

        payload = state.to_checkpoint()

        payload.pop("last_action_type")
        payload.pop("last_action_description")

        restored = AgentState.from_checkpoint(payload, capabilities=self.__caps())

        self.assertIsNone(restored.last_action_type)

    def test_record_attempt_pushes_intent_into_loop_detector(self) -> None:
        """
        Four rejected attempts of the same target are enough to flip ``is_stuck`` via the LoopDetector.
        Without ``record_attempt`` the intent never accumulates because no ``record_step`` runs on SUPERVISE-rejected paths.
        """

        state = AgentState(intent="test", capabilities=self.__caps())
        state.update_screen(screen=self.__screen())

        for _ in range(4):
            state.record_attempt(
                reason="supervise_spatial_unresolved",
                action=self.__tap_action(target="Play button"),
            )

        self.assertTrue(state.is_stuck)

    def test_record_attempt_updates_last_action(self) -> None:
        """
        After a rejected attempt the agent's intent must be the last
        recorded action so the next ``update_screen`` capture inherits it and the LoopDetector window stays in sync.
        """

        state = AgentState(intent="test", capabilities=self.__caps())

        state.record_attempt(
            reason="low_confidence",
            action=self.__tap_action(target="Submit"),
        )

        self.assertEqual(state.last_action_type, "tap")
        self.assertEqual(state.last_action_description, "Tap on Submit")

    def test_last_action_description_returns_none_when_no_action_recorded(self) -> None:
        """
        ``last_action_description`` must surface ``None`` on a fresh state with no recorded action.
        """

        state = AgentState(intent="test", capabilities=self.__caps())

        self.assertIsNone(state.last_action_description)

    def test_last_action_description_survives_checkpoint_round_trip(self) -> None:
        """
        ``last_action_description`` must round-trip through the checkpoint so a graph-state restore inherits the descriptor.
        """

        state = AgentState(intent="test", capabilities=self.__caps())
        state.record_attempt(
            reason="low_confidence",
            action=self.__tap_action(target="Continue"),
        )

        payload = state.to_checkpoint()
        restored = AgentState.from_checkpoint(data=payload, capabilities=self.__caps())

        self.assertEqual(restored.last_action_description, "Tap on Continue")

    def test_record_blocked_action_trips_loop_detector(self) -> None:
        """
        Four blocks of the same target must flip ``is_stuck`` via the LoopDetector.
        """

        state = AgentState(intent="test", capabilities=self.__caps())
        state.update_screen(screen=self.__screen())

        for _ in range(4):
            state.record_blocked_action(
                reason="Action 'Tap on Free Offers' already succeeded on the current screen during this workflow.",
                action=self.__tap_action(target="Free Offers"),
                block_reason=BlockReason.REPEATED_CURRENT_SCREEN_ACTION,
            )

        self.assertTrue(state.is_stuck)

    def test_record_blocked_action_preserves_failure_context(self) -> None:
        """
        ``record_blocked_action`` must still push history, record failure context, and set ``last_error``.
        """

        state = AgentState(intent="test", capabilities=self.__caps())
        state.update_screen(screen=self.__screen())

        state.record_blocked_action(
            reason="Action already succeeded on this screen.",
            action=self.__tap_action(target="Free Offers"),
            block_reason=BlockReason.REPEATED_CURRENT_SCREEN_ACTION,
        )

        self.assertEqual(
            state._AgentState__last_error,  # type: ignore[attr-defined]
            "Action already succeeded on this screen.",
        )
        self.assertEqual(state.last_action_type, "tap")


class AgentStatePlannerRetryBudgetTest(unittest.TestCase):
    """
    Pins the per-step planner-retry budget — the bug class missed in production on PFTXN run 77873149 where 55 consecutive should_retry=True returns never bounded the workflow.
    """

    @staticmethod
    def __caps() -> RuntimeCapabilities:
        """
        Return autonomous capabilities; HITL toggle is irrelevant for budget arithmetic.
        """

        return RuntimeCapabilities(hitl=HITLCapability(enabled=False))

    @staticmethod
    def __state(*, cap: int = 5) -> AgentState:
        """
        Build an AgentState with the requested planner-retry cap.
        """

        return AgentState(
            intent="x",
            capabilities=AgentStatePlannerRetryBudgetTest.__caps(),
            max_steps=10,
            retries=RetryLimits(planner=cap),
        )

    @staticmethod
    def __tap_action(*, target: str = "More on Swiggy") -> Action:
        """
        Tap action used to exercise the budget reset on EXECUTE dispatch.
        """

        return Action(
            target=target,
            confidence=1.0,
            rationale="test fixture",
            action_type=ActionType.TAP,
        )

    def test_initial_budget_count_is_zero(self) -> None:
        """
        A freshly-built AgentState starts with no consumed retries.
        """

        state = self.__state(cap=5)

        self.assertEqual(state.retries.planner.count, 0)
        self.assertEqual(state.retries.planner.cap, 5)
        self.assertFalse(state.retries.planner.exhausted)

    def test_cap_is_clamped_by_caller_max_steps(self) -> None:
        """
        Caller-passed max_steps is the hard bound; retries.planner must not silently exceed it.
        """

        state = AgentState(
            intent="x",
            capabilities=self.__caps(),
            max_steps=1,
            retries=RetryLimits(planner=5),
        )

        self.assertEqual(state.retries.planner.cap, 1)

    def test_cap_clamp_defends_against_internal_callers_passing_zero(self) -> None:
        """
        Schema validators reject ``max_steps=0`` at the boundary; but internal callers that bypass the schema must still get a working budget (cap >= 1) instead of a Pydantic error.
        """

        state = AgentState(
            intent="x",
            capabilities=self.__caps(),
            max_steps=0,
            retries=RetryLimits(planner=5),
        )

        self.assertEqual(state.retries.planner.cap, 1)

    def test_can_continue_returns_false_when_planner_retries_exhausted(self) -> None:
        """
        ``can_continue`` is the central termination gate consulted by the planner; it must honor an exhausted planner-retry budget.
        """

        state = self.__state(cap=2)
        for _ in range(2):
            state.tick_planner_retry(
                kind=RetryKind.SILENT_REJECTION,
                branch=RetryBranch.SHOULD_AVOID_ACTION,
                action="Swipe left",
            )

        self.assertTrue(state.retries.planner.exhausted)
        self.assertFalse(state.can_continue)

    def test_silent_rejection_consumes_budget(self) -> None:
        """
        Silent-rejection retries are the bug class under audit; they must increment the counter.
        """

        state = self.__state(cap=3)

        for index in range(3):
            count = state.tick_planner_retry(
                kind=RetryKind.SILENT_REJECTION,
                branch=RetryBranch.SHOULD_AVOID_ACTION,
                action="Swipe left",
            )
            self.assertEqual(count, index + 1)

        self.assertTrue(state.retries.planner.exhausted)

    def test_llm_feedback_consumes_budget(self) -> None:
        """
        Feedback-bearing retries (current_screen_repeat / is_action_repeating) also count.
        """

        state = self.__state(cap=2)

        state.tick_planner_retry(
            kind=RetryKind.LLM_FEEDBACK,
            branch=RetryBranch.CURRENT_SCREEN_REPEAT,
            action="Tap Continue",
        )
        state.tick_planner_retry(
            kind=RetryKind.LLM_FEEDBACK,
            branch=RetryBranch.IS_ACTION_REPEATING_ON_SCREEN,
            action="Tap Continue",
        )

        self.assertEqual(state.retries.planner.count, 2)
        self.assertTrue(state.retries.planner.exhausted)

    def test_escalation_deferred_does_not_consume_budget_with_active_sub_goal(self) -> None:
        """
        With an active sub-goal, escalation-deferred is bounded by ``deferral_count``; planner budget must stay clean.
        """

        state = self.__state(cap=5)
        state.set_sub_goals([SubGoalFixtures.make(description="Open settings")])

        for _ in range(20):
            state.tick_planner_retry(
                kind=RetryKind.ESCALATION_DEFERRED,
                branch=RetryBranch.ESCALATION_DEFERRED,
            )

        self.assertEqual(state.retries.planner.count, 0)
        self.assertFalse(state.retries.planner.exhausted)

    def test_escalation_deferred_consumes_budget_without_active_sub_goal(self) -> None:
        """
        Without an active sub-goal there is no ``deferral_count`` to bound escalation; the planner-retry budget must catch the loop.
        """

        state = self.__state(cap=3)
        # No sub-goals — non-decomposed run; deferral_count is a no-op here.

        for index in range(3):
            count = state.tick_planner_retry(
                kind=RetryKind.ESCALATION_DEFERRED,
                branch=RetryBranch.ESCALATION_DEFERRED,
            )
            self.assertEqual(count, index + 1)

        self.assertTrue(state.retries.planner.exhausted)

    def test_clear_resets_counter(self) -> None:
        """
        ``clear_planner_retries`` resets the count to zero without altering ``cap``.
        """

        state = self.__state(cap=5)

        state.tick_planner_retry(
            kind=RetryKind.SILENT_REJECTION,
            branch=RetryBranch.SHOULD_AVOID_ACTION,
            action="Swipe left",
        )
        state.clear_planner_retries()

        self.assertEqual(state.retries.planner.count, 0)
        self.assertEqual(state.retries.planner.cap, 5)

    def test_record_step_clears_counter_on_success(self) -> None:
        """
        A SUCCESSFUL EXECUTE→RECORD round-trip is the load-bearing reset signal.
        """

        state = self.__state(cap=5)

        for _ in range(3):
            state.tick_planner_retry(
                kind=RetryKind.SILENT_REJECTION,
                branch=RetryBranch.SHOULD_AVOID_ACTION,
                action="Swipe left",
            )
        self.assertEqual(state.retries.planner.count, 3)

        result = StepResult(
            step=Step(action=self.__tap_action(), step_number=1, screen_hash="b" * 16),
            error=None,
            success=True,
            duration=10,
            pre_hash="b" * 16,
            post_hash="c" * 16,
            screen_changed=True,
        )
        state.record_step(result=result)

        self.assertEqual(state.retries.planner.count, 0)
        self.assertEqual(state.step_count, 1)

    def test_record_step_does_not_clear_counter_on_failure(self) -> None:
        """
        A failed-dispatch ``record_step`` must NOT reset the counter; otherwise the silent-rejection loop would reset endlessly when a flaky dispatch fails.
        """

        state = self.__state(cap=5)

        for _ in range(3):
            state.tick_planner_retry(
                kind=RetryKind.SILENT_REJECTION,
                branch=RetryBranch.SHOULD_AVOID_ACTION,
                action="Swipe left",
            )

        failed_result = StepResult(
            step=Step(action=self.__tap_action(), step_number=1, screen_hash="b" * 16),
            error="dispatch error",
            success=False,
            duration=10,
            pre_hash="b" * 16,
            post_hash="b" * 16,
            screen_changed=False,
        )
        state.record_step(result=failed_result)

        self.assertEqual(state.retries.planner.count, 3)
        self.assertEqual(state.step_count, 1)

    def test_verifier_rejection_streak_survives_recorded_validate_steps(self) -> None:
        """
        Gate-routed verifier loops execute and record validate steps between VERIFY rejections; same-screen rejections must still accumulate.
        """

        state = self.__state(cap=5)
        screen = ScreenState(
            activity="save-account",
            activity_hash="a" * 16,
            visual_hash="1" * 16,
            timestamp=1,
        )

        first = state.record_verify_rejection(screen=screen, activity="save-account")
        self.assertEqual(first.consecutive_rejections, 1)

        for step_number in range(1, DEFAULT_VERIFICATION_REJECTION_LIMIT):
            result = StepResult(
                step=Step(
                    action=self.__tap_action(target="validate"),
                    step_number=step_number,
                    screen_hash="b" * 16,
                ),
                error=None,
                success=True,
                duration=10,
                pre_hash="b" * 16,
                post_hash="b" * 16,
                screen_changed=False,
            )
            state.record_step(result=result)
            loop_state = state.record_verify_rejection(screen=screen, activity="save-account")

        self.assertEqual(state.step_count, DEFAULT_VERIFICATION_REJECTION_LIMIT - 1)
        self.assertEqual(loop_state.recorded_step_count, 0)
        self.assertEqual(
            loop_state.consecutive_rejections,
            DEFAULT_VERIFICATION_REJECTION_LIMIT,
        )

    def test_verifier_rejection_streak_does_not_use_activity_without_screen(self) -> None:
        """
        Activity alone is not enough evidence to continue a verifier rejection streak.
        """

        state = self.__state(cap=5)

        first = state.record_verify_rejection(screen=None, activity="save-account")
        second = state.record_verify_rejection(screen=None, activity="save-account")

        self.assertEqual(first.consecutive_rejections, 1)
        self.assertEqual(second.consecutive_rejections, 1)

    def test_verifier_rejection_streak_resets_on_screen_progress_same_activity(self) -> None:
        """
        Same activity is not enough to continue a verifier streak when screen identity changes.
        """

        state = self.__state(cap=5)

        first = state.record_verify_rejection(
            screen=ScreenState(
                activity="save-account",
                activity_hash="a" * 16,
                visual_hash="1" * 16,
                timestamp=1,
            ),
            activity="save-account",
        )
        self.assertEqual(first.consecutive_rejections, 1)

        second = state.record_verify_rejection(
            screen=ScreenState(
                activity="save-account",
                activity_hash="a" * 16,
                visual_hash="f" * 16,
                timestamp=2,
            ),
            activity="save-account",
        )

        self.assertEqual(second.consecutive_rejections, 1)

    def test_last_attempt_snapshot_captures_branch_and_action(self) -> None:
        """
        ``last_attempt`` reflects the most recent tick so observability has structured branch / action context.
        """

        state = self.__state(cap=5)

        state.tick_planner_retry(
            kind=RetryKind.SILENT_REJECTION,
            branch=RetryBranch.SHOULD_AVOID_ACTION,
            action="Swipe left on More on Swiggy",
        )

        attempt = state.last_retry_attempt
        self.assertIsNotNone(attempt)
        assert attempt is not None
        self.assertIs(attempt.kind, RetryKind.SILENT_REJECTION)
        self.assertIs(attempt.branch, RetryBranch.SHOULD_AVOID_ACTION)
        self.assertEqual(attempt.action, "Swipe left on More on Swiggy")


class AgentStatePlannerRetryCheckpointTest(unittest.TestCase):
    """
    Pins that the planner-retry budget survives the persist/restore round-trip — the original RCA flagged this as a separate bug class on the cozy-orbiting-sonnet plan.
    """

    @staticmethod
    def __caps() -> RuntimeCapabilities:
        """
        Return autonomous capabilities.
        """

        return RuntimeCapabilities(hitl=HITLCapability(enabled=False))

    def test_count_round_trips_through_checkpoint(self) -> None:
        """
        Count must survive ``to_checkpoint`` / ``from_checkpoint`` so a graph restore mid-loop cannot wipe progress toward the cap.
        """

        state = AgentState(intent="x", capabilities=self.__caps(), retries=RetryLimits(planner=5))
        for _ in range(3):
            state.tick_planner_retry(
                kind=RetryKind.SILENT_REJECTION,
                branch=RetryBranch.SHOULD_AVOID_ACTION,
                action="Swipe left",
            )

        checkpoint = state.to_checkpoint()
        restored = AgentState.from_checkpoint(checkpoint, capabilities=self.__caps())

        self.assertEqual(restored.retries.planner.count, 3)
        self.assertEqual(restored.retries.planner.cap, 5)
        self.assertFalse(restored.retries.planner.exhausted)

    def test_legacy_checkpoint_without_retries_restores_with_defaults(self) -> None:
        """
        Old checkpoints written before this change have no ``retries`` key; restore must default to zero with the constant cap.
        """

        state = AgentState(intent="x", capabilities=self.__caps())
        payload = state.to_checkpoint()
        payload.pop("retries")

        restored = AgentState.from_checkpoint(payload, capabilities=self.__caps())

        self.assertEqual(restored.retries.planner.count, 0)
        self.assertEqual(restored.retries.planner.cap, DEFAULT_PLANNER_RETRY_LIMIT)

    def test_exhausted_state_round_trips_through_checkpoint(self) -> None:
        """
        Exhaustion must survive restore so a resumed workflow does not get a free retry budget after the ceiling fired.
        """

        state = AgentState(intent="x", capabilities=self.__caps(), retries=RetryLimits(planner=2))
        for _ in range(2):
            state.tick_planner_retry(
                kind=RetryKind.SILENT_REJECTION,
                branch=RetryBranch.SHOULD_AVOID_ACTION,
                action="Swipe left",
            )
        self.assertTrue(state.retries.planner.exhausted)

        checkpoint = state.to_checkpoint()
        restored = AgentState.from_checkpoint(checkpoint, capabilities=self.__caps())

        self.assertTrue(restored.retries.planner.exhausted)

    def test_last_retry_attempt_round_trips_through_checkpoint(self) -> None:
        """
        Diagnostic ``last_retry_attempt`` must survive restore so observability does not lose the most recent rejection on resume.
        """

        state = AgentState(intent="x", capabilities=self.__caps(), retries=RetryLimits(planner=5))

        state.tick_planner_retry(
            action="Swipe left on More",
            kind=RetryKind.SILENT_REJECTION,
            branch=RetryBranch.SHOULD_AVOID_ACTION,
        )

        restored = AgentState.from_checkpoint(state.to_checkpoint(), capabilities=self.__caps())

        self.assertIsNotNone(restored.last_retry_attempt)
        attempt = restored.last_retry_attempt

        assert attempt is not None
        self.assertEqual(attempt.action, "Swipe left on More")
        self.assertEqual(attempt.kind, RetryKind.SILENT_REJECTION)
        self.assertEqual(attempt.branch, RetryBranch.SHOULD_AVOID_ACTION)

    def test_clear_planner_retries_drops_last_attempt(self) -> None:
        """
        ``clear_planner_retries`` is called on successful EXECUTE dispatch; stale diagnostics from the prior cycle must not survive into the next step.
        """

        state = AgentState(intent="x", capabilities=self.__caps(), retries=RetryLimits(planner=5))

        state.tick_planner_retry(
            action="Swipe left on More",
            kind=RetryKind.SILENT_REJECTION,
            branch=RetryBranch.SHOULD_AVOID_ACTION,
        )
        self.assertIsNotNone(state.last_retry_attempt)

        state.clear_planner_retries()

        self.assertIsNone(state.last_retry_attempt)
        self.assertEqual(state.retries.planner.count, 0)
