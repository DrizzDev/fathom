"""
Pins for :class:`SubGoal` schema changes introduced in Phase 1C.

The active concern is the per-sub-goal step budget: the schema must
expose a ``max_steps`` field with a sensible default so the RECORD
node's ``SUBGOAL_BUDGET_EXCEEDED`` trigger has something to compare
against without requiring decomposer changes.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fathom.constants.subgoal import (
    DEFAULT_SUB_GOAL_MAX_STEPS,
    INPUT_SUB_GOAL_MAX_STEPS,
    SCROLL_SUB_GOAL_MAX_STEPS,
    TAP_SUB_GOAL_MAX_STEPS,
)
from fathom.schemas.subgoal import (
    ExecutionContract,
    RequiredActionFamily,
    ScrollAxis,
    SubGoal,
    SubGoalStatus,
    default_max_steps_for_execution_contract,
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

    def test_scroll_contract_uses_scroll_budget(self) -> None:
        """
        Scroll-family contracts receive the larger action budget.
        """

        assert (
            default_max_steps_for_execution_contract(
                contract=ExecutionContract(
                    required_action_family=RequiredActionFamily.SCROLL,
                )
            )
            == SCROLL_SUB_GOAL_MAX_STEPS
        )

    def test_tap_contract_uses_tap_budget(self) -> None:
        """
        Tap-family contracts receive the smaller interaction budget.
        """

        assert (
            default_max_steps_for_execution_contract(
                contract=ExecutionContract(
                    required_action_family=RequiredActionFamily.TAP,
                )
            )
            == TAP_SUB_GOAL_MAX_STEPS
        )

    def test_input_contract_uses_input_budget(self) -> None:
        """
        Input-family contracts receive the typing/search budget.
        """

        assert (
            default_max_steps_for_execution_contract(
                contract=ExecutionContract(
                    required_action_family=RequiredActionFamily.INPUT,
                )
            )
            == INPUT_SUB_GOAL_MAX_STEPS
        )

    def test_surface_is_preserved_on_execution_contract(self) -> None:
        """
        Structured surface context must round-trip without any parsing heuristics.
        """

        contract = ExecutionContract(
            required_action_family=RequiredActionFamily.SCROLL,
            scroll_axis=ScrollAxis.HORIZONTAL,
            surface="below Fast Delivery section",
        )

        assert contract.surface == "below Fast Delivery section"
