from __future__ import annotations

import pytest
from pydantic import ValidationError

from fathom.constants.subgoal import DEFAULT_SUB_GOAL_MAX_STEPS
from fathom.schemas.subgoal import GoalState, Progress, SubGoal, SubGoalStatus
from tests.builders import SuccessFixtures


class TestProgressBudget:
    """
    Behavioural pins for mutable sub-goal progress: status, budget, and validated counters.
    """

    def test_status_defaults_to_pending(self) -> None:
        """
        A freshly constructed progress starts PENDING.
        """

        assert Progress().status == SubGoalStatus.PENDING

    def test_default_limit_uses_named_constant(self) -> None:
        """
        Progress without an explicit limit picks up the module-level default budget.
        """

        assert Progress().limit == DEFAULT_SUB_GOAL_MAX_STEPS

    def test_limit_override_is_preserved(self) -> None:
        """
        An explicit budget flows through unchanged.
        """

        assert Progress(limit=3).limit == 3

    def test_zero_limit_is_rejected(self) -> None:
        """
        A zero budget would prevent any progress and must not validate.
        """

        with pytest.raises(ValidationError):
            Progress(limit=0)

    def test_negative_counter_assignment_is_rejected(self) -> None:
        """
        Assignment validation forbids a negative attempt counter after construction.
        """

        progress = Progress()
        with pytest.raises(ValidationError):
            progress.attempts = -1


class TestSubGoalDefinition:
    """
    Pins the immutable sub-goal definition and its GoalState aggregate.
    """

    @staticmethod
    def __goal() -> SubGoal:
        return SubGoal(index=0, objective="Open Swiggy app", success=SuccessFixtures.observed())

    def test_goal_is_immutable(self) -> None:
        """
        The sub-goal definition is frozen: its objective cannot be reassigned.
        """

        with pytest.raises(ValidationError):
            self.__goal().objective = "mutated"  # type: ignore[misc]

    def test_goal_state_exposes_definition(self) -> None:
        """
        GoalState surfaces the immutable definition and defaults progress to PENDING.
        """

        state = GoalState(goal=self.__goal())
        assert state.index == 0
        assert state.objective == "Open Swiggy app"
        assert state.progress.status == SubGoalStatus.PENDING

    def test_lifecycle_transitions_apply_to_progress(self) -> None:
        """
        Lifecycle transitions mutate progress while leaving the definition intact.
        """

        state = GoalState(goal=self.__goal())
        state.mark_in_progress()
        assert not state.is_complete()
        state.mark_complete()
        assert state.is_complete()
