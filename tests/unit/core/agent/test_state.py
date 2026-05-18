"""
Unit pins for :class:`fathom.core.agent.state.AgentState`.

Mirrors the source file at `src/fathom/core/agent/state.py`. Covers
the action-effect trajectory plumbing exposed to ANALYZE and RECORD,
and the cosmetic-replan counter preservation that prevents the step-24
unresolvable-target cascade observed in the healing.txt runs.
"""

from __future__ import annotations

from fathom.constants.screen import NO_PROGRESS_RECOVERY_THRESHOLD
from fathom.core.agent.state import AgentState
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.schemas.screens import ScreenDiff
from fathom.schemas.subgoal import SubGoal


def _effect(status: ActionEffectStatus, *, visual_progress: float = 0.0) -> ActionEffect:
    """
    Build a minimal :class:`ActionEffect` for trajectory tests.
    """

    return ActionEffect(
        status=status,
        visual_progress=visual_progress,
        phash_distance=0,
    )


def _goal(description: str, index: int = 0) -> SubGoal:
    """
    Build a minimal :class:`SubGoal` for replan tests.
    """

    return SubGoal(index=index, description=description)


class TestActionEffectTrajectory:
    """
    Pins for AgentState action-effect tracking and its consumers.
    """

    def test_recent_effects_empty_before_any_record(self) -> None:
        """
        A fresh :class:`AgentState` has no recorded effects.
        """

        state = AgentState(intent="x")

        assert state.get_recent_effects() == []
        assert state.get_last_action_effect() is None
        assert state.consecutive_no_progress_count == 0

    def test_record_action_effect_round_trips_through_accessor(self) -> None:
        """
        Recording an effect makes it observable via both accessors.
        """

        state = AgentState(intent="x")
        effect = _effect(ActionEffectStatus.PROGRESS, visual_progress=0.42)
        state.record_action_effect(effect=effect)

        assert state.get_recent_effects() == [effect]
        assert state.get_last_action_effect() == effect

    def test_consecutive_no_progress_counts_only_trailing_tail(self) -> None:
        """
        A trailing PROGRESS resets the trailing NO_PROGRESS counter.
        """

        state = AgentState(intent="x")
        state.record_action_effect(effect=_effect(ActionEffectStatus.NO_PROGRESS))
        state.record_action_effect(effect=_effect(ActionEffectStatus.NO_PROGRESS))
        state.record_action_effect(effect=_effect(ActionEffectStatus.PROGRESS))
        state.record_action_effect(effect=_effect(ActionEffectStatus.NO_PROGRESS))

        assert state.consecutive_no_progress_count == 1

    def test_consecutive_no_progress_meets_recovery_threshold(self) -> None:
        """
        Threshold-many NO_PROGRESS records cross the escalation bar.
        """

        state = AgentState(intent="x")
        for _ in range(NO_PROGRESS_RECOVERY_THRESHOLD):
            state.record_action_effect(effect=_effect(ActionEffectStatus.NO_PROGRESS))

        assert state.consecutive_no_progress_count == NO_PROGRESS_RECOVERY_THRESHOLD

    def test_uncertain_breaks_no_progress_run(self) -> None:
        """
        UNCERTAIN classification breaks the trailing-tail count.
        """

        state = AgentState(intent="x")
        state.record_action_effect(effect=_effect(ActionEffectStatus.NO_PROGRESS))
        state.record_action_effect(effect=_effect(ActionEffectStatus.UNCERTAIN))
        state.record_action_effect(effect=_effect(ActionEffectStatus.NO_PROGRESS))

        assert state.consecutive_no_progress_count == 1

    def test_visual_no_progress_wins_over_xml_jitter(self) -> None:
        """
        phash=0 SSIM=1.0 frames must classify as NO_PROGRESS despite XML jitter.
        """

        diff = ScreenDiff(
            phash_distance=0,
            ssim_score=1.0,
            content_pixel_diff_ratio=0.0008,
            xml_hash_changed=True,
            interaction_hash_changed=True,
            activity_changed=False,
        )

        effect = ActionEffect.from_screen_diff(diff=diff)

        assert effect.status == ActionEffectStatus.NO_PROGRESS

    def test_recent_effects_and_subgoal_budget_survive_checkpoint(self) -> None:
        """
        Checkpoint round-trip preserves effects and per-task budget.
        """

        state = AgentState(intent="order dosa")
        state.set_sub_goals([SubGoal(index=0, description="Scroll up 40% auto suggest page")])
        state.record_sub_goal_action()
        state.record_action_effect(effect=_effect(ActionEffectStatus.NO_PROGRESS))

        restored = AgentState.from_checkpoint(state.to_checkpoint())

        assert restored.current_sub_goal_action_count == 1
        assert restored.consecutive_no_progress_count == 1
        last = restored.get_last_action_effect()
        assert last is not None
        assert last.status == ActionEffectStatus.NO_PROGRESS


class TestReplanBudgetPreservation:
    """
    Pins for the cosmetic-replan counter preservation on AgentState.
    """

    def test_cosmetic_replan_preserves_counter(self) -> None:
        """
        Replan to the same target must NOT reset the action counter.
        """

        state = AgentState(intent="order dosa")
        state.set_sub_goals([_goal("Scroll up 40% auto suggest page")])
        state.record_sub_goal_action()
        state.record_sub_goal_action()
        state.record_sub_goal_action()
        assert state.current_sub_goal_action_count == 3

        state.replan_pending_sub_goals(new_sub_goals=[_goal("Scroll up 40% auto suggest page")])

        assert state.current_sub_goal_action_count == 3

    def test_cosmetic_replan_eventually_trips_budget(self) -> None:
        """
        Cosmetic replans cross the per-task attempt budget.
        """

        budget = 3
        state = AgentState(intent="order dosa")
        state.set_sub_goals(
            [SubGoal(index=0, description="Scroll up 40% auto suggest page", max_steps=budget)]
        )

        for _ in range(budget):
            state.record_sub_goal_action()
            state.replan_pending_sub_goals(
                new_sub_goals=[
                    SubGoal(
                        index=0,
                        description="Scroll up 40% auto suggest page",
                        max_steps=budget,
                    )
                ]
            )

        assert state.current_sub_goal_over_budget is True

    def test_structural_replan_resets_counter(self) -> None:
        """
        Replan to a genuinely different target must reset the counter.
        """

        state = AgentState(intent="order dosa")
        state.set_sub_goals([_goal("Tap on Alright, got it button")])
        state.record_sub_goal_action()
        state.record_sub_goal_action()
        assert state.current_sub_goal_action_count == 2

        state.replan_pending_sub_goals(new_sub_goals=[_goal("Tap on the cross icon")])

        assert state.current_sub_goal_action_count == 0

    def test_normalization_ignores_case_whitespace_punctuation(self) -> None:
        """
        Surface differences that do not change the imperative target stay cosmetic.
        """

        state = AgentState(intent="order dosa")
        state.set_sub_goals([_goal("Tap on Alright, got it button")])
        state.record_sub_goal_action()

        state.replan_pending_sub_goals(new_sub_goals=[_goal("  tap on alright, got it button.  ")])

        assert state.current_sub_goal_action_count == 1

    def test_replan_with_empty_new_goals_resets_counter(self) -> None:
        """
        Empty replan resets the counter and advances past the list.
        """

        state = AgentState(intent="order dosa")
        state.set_sub_goals([_goal("Scroll up 40% auto suggest page")])
        state.record_sub_goal_action()

        state.replan_pending_sub_goals(new_sub_goals=[])

        assert state.current_sub_goal_action_count == 0


class TestRuntimeTaskMirror:
    """
    Pins that AgentState mirrors task state into the runtime aggregate.
    """

    def test_set_sub_goals_loads_runtime_tasks(self) -> None:
        """
        Setting sub-goals populates the runtime task component.
        """

        state = AgentState(intent="order dosa")
        state.set_sub_goals(
            [
                SubGoal(index=0, description="Open Swiggy app"),
                SubGoal(index=1, description="Tap on search bar"),
            ]
        )

        assert state.runtime.tasks.progress() == (0, 2)
        active = state.runtime.tasks.active()
        assert active is not None
        assert active.objective == "Open Swiggy app"

    def test_record_sub_goal_action_increments_runtime_attempt(self) -> None:
        """
        Sub-goal action recording increments runtime task attempts.
        """

        state = AgentState(intent="order dosa")
        state.set_sub_goals([SubGoal(index=0, description="Open Swiggy app")])
        state.record_sub_goal_action()
        state.record_sub_goal_action()

        active = state.runtime.tasks.active()
        assert active is not None
        assert active.attempts.count == 2


class TestCompleteDeferralCounter:
    """
    Pins the consecutive-complete-deferral counter the intent router
    uses to bound how many times an ``is_complete=True`` ANALYZE verdict
    can be rejected before the planner's claim is honoured. Without
    this bound the router can ground-loop indefinitely on a finished
    screen, burning the sub-goal budget before VERIFY ever runs
    (observed at log 10.txt steps 19-25).
    """

    def test_default_deferral_count_is_zero(self) -> None:
        """
        Fresh state must start with no deferrals on record so the first
        complete claim is treated as a clean signal.
        """

        state = AgentState(intent="order dosa")

        assert state.consecutive_complete_deferrals == 0

    def test_record_complete_deferral_increments_and_returns_count(self) -> None:
        """
        Each call increments by one and returns the post-increment
        value so the router can compare directly to the cap.
        """

        state = AgentState(intent="order dosa")

        assert state.record_complete_deferral() == 1
        assert state.record_complete_deferral() == 2
        assert state.consecutive_complete_deferrals == 2

    def test_reset_complete_deferrals_clears_streak(self) -> None:
        """
        Explicit reset (called on forward progress) zeros the counter
        so a stale streak does not bias the next planner verdict.
        """

        state = AgentState(intent="order dosa")
        state.record_complete_deferral()
        state.record_complete_deferral()

        state.reset_complete_deferrals()

        assert state.consecutive_complete_deferrals == 0

    def test_sub_goal_advance_resets_deferrals(self) -> None:
        """
        Advancing a sub-goal counts as forward progress; the
        complete-deferral streak must reset implicitly so a late
        deferral from the previous sub-goal does not poison the next
        one's planner budget.
        """

        from fathom.schemas.reasoning import SubGoalCompletionSignal

        state = AgentState(intent="order dosa")
        state.set_sub_goals(
            [
                SubGoal(index=0, description="Open Swiggy"),
                SubGoal(index=1, description="Search dosa"),
            ],
        )
        state.record_complete_deferral()
        state.record_complete_deferral()

        state.mark_current_sub_goal_complete(
            completion_signal=SubGoalCompletionSignal(
                trace_verified=False,
                evidence="planner flagged complete",
                keyword_match=False,
                llm_confidence=1.0,
                action_executed=True,
                flagged_complete=True,
                rationale_verified=True,
            ),
        )

        assert state.consecutive_complete_deferrals == 0
