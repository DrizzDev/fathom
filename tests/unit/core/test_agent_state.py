"""Tests for AgentState sub-goal management, checkpoint persistence, and replanning."""

from __future__ import annotations

from fathom.core.agent.state import AgentState
from fathom.schemas.subgoal import SubGoal, SubGoalStatus


def _make_sub_goals(descriptions: list[str]) -> list[SubGoal]:
    return [SubGoal(index=i, description=d) for i, d in enumerate(descriptions)]


class TestReplaceRemainingSubGoals:
    def test_preserves_completed_sub_goals(self) -> None:
        state = AgentState(intent="do everything", max_steps=50)
        state.set_sub_goals(_make_sub_goals(["open app", "search for X", "add to cart"]))

        # Complete first two
        from fathom.schemas.reasoning import SubGoalCompletionSignal

        signal = SubGoalCompletionSignal(
            evidence="done", llm_signaled=True, action_executed=True, screen_verified=True
        )
        state.mark_current_sub_goal_complete(completion_signal=signal)
        state.mark_current_sub_goal_complete(completion_signal=signal)

        # Replace remaining with new goals
        new_goals = _make_sub_goals(["go to cart", "checkout"])
        state.replace_remaining_sub_goals(new_goals)

        all_goals = state.sub_goals
        assert len(all_goals) == 4  # 2 completed + 2 new
        assert all_goals[0].is_complete()
        assert all_goals[1].is_complete()
        assert all_goals[0].description == "open app"
        assert all_goals[1].description == "search for X"
        assert all_goals[2].description == "go to cart"
        assert all_goals[3].description == "checkout"
        assert all_goals[2].status == SubGoalStatus.IN_PROGRESS

    def test_resets_counters(self) -> None:
        state = AgentState(intent="test", max_steps=50)
        state.set_sub_goals(_make_sub_goals(["step 1", "step 2"]))

        # Simulate actions and verify failures
        for _ in range(10):
            state.record_sub_goal_action()
        state.record_verify_failure()
        state.record_verify_failure()

        assert state.sub_goal_action_count == 10
        assert state.sub_goal_verify_failures == 2

        state.replace_remaining_sub_goals(_make_sub_goals(["new step"]))

        assert state.sub_goal_action_count == 0
        assert state.sub_goal_verify_failures == 0

    def test_empty_replacement_clears_unfinished(self) -> None:
        state = AgentState(intent="test", max_steps=50)
        state.set_sub_goals(_make_sub_goals(["step 1"]))

        state.replace_remaining_sub_goals([])

        assert len(state.sub_goals) == 0


class TestCheckpointPersistence:
    def test_verify_failure_counter_survives_checkpoint(self) -> None:
        state = AgentState(intent="test intent", max_steps=100)
        state.set_sub_goals(_make_sub_goals(["step 1", "step 2"]))
        state.record_verify_failure()
        state.record_verify_failure()
        state.record_verify_failure()

        checkpoint = state.to_checkpoint()
        restored = AgentState.from_checkpoint(checkpoint)

        assert restored.sub_goal_verify_failures == 3

    def test_action_count_survives_checkpoint(self) -> None:
        state = AgentState(intent="test intent", max_steps=100)
        state.set_sub_goals(_make_sub_goals(["step 1"]))
        for _ in range(7):
            state.record_sub_goal_action()

        checkpoint = state.to_checkpoint()
        restored = AgentState.from_checkpoint(checkpoint)

        assert restored.sub_goal_action_count == 7

    def test_counters_default_zero_for_old_checkpoints(self) -> None:
        """Checkpoints from before these fields were added should default to 0."""

        state = AgentState(intent="test", max_steps=50)
        checkpoint = state.to_checkpoint()

        # Simulate old checkpoint without these keys
        del checkpoint["sub_goal_verify_failures"]
        del checkpoint["sub_goal_action_count"]

        restored = AgentState.from_checkpoint(checkpoint)
        assert restored.sub_goal_verify_failures == 0
        assert restored.sub_goal_action_count == 0
