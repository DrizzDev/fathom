"""
Pins for :class:`SubGoal` schema changes introduced in Phase 1C.

The active concern is the per-sub-goal step budget: the schema must
expose a ``max_steps`` field with a sensible default without requiring
decomposer changes.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fathom.constants.subgoal import (
    DEFAULT_SUB_GOAL_MAX_STEPS,
)
from fathom.schemas.subgoal import (
    SubGoal,
    SubGoalStatus,
)


class TestSubGoalBudget:
    """
    Behavioural pins for the Phase 1C step-budget field.
    """

    def test_default_max_steps_uses_named_constant(self) -> None:
        """
        Constructing a sub-goal without a ``max_steps`` override
        picks up the module-level default constant.
        """

        sub_goal = SubGoal(index=0, description="Open Swiggy app")
        assert sub_goal.max_steps == DEFAULT_SUB_GOAL_MAX_STEPS

    def test_max_steps_override_is_preserved(self) -> None:
        """
        Explicit ``max_steps`` values flow through the model unchanged.
        """

        sub_goal = SubGoal(index=0, description="x", max_steps=3)
        assert sub_goal.max_steps == 3

    def test_zero_max_steps_is_rejected(self) -> None:
        """
        ``max_steps`` is bounded ``ge=1`` — a zero budget would prevent
        the agent from making any progress and must not validate.
        """

        with pytest.raises(ValidationError):
            SubGoal(index=0, description="x", max_steps=0)

    def test_status_defaults_to_pending(self) -> None:
        """
        Pin behaviour of the existing status field so the budget
        change didn't accidentally shift the default.
        """

        sub_goal = SubGoal(index=0, description="x")
        assert sub_goal.status == SubGoalStatus.PENDING
