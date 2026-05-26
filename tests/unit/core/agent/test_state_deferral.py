"""
Unit pins for :class:`AgentState` deferral helpers and the observable-progress reset.
"""

from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.agent.state import AgentState
from fathom.schemas.actions import Action
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.subgoal import SubGoal


class AgentStateDeferralTest(unittest.TestCase):
    """
    Pins the deferral counter API and its automatic reset on real progress.
    """

    @staticmethod
    def __state_with_subgoal() -> AgentState:
        state = AgentState(intent="x")
        state.set_sub_goals([SubGoal(description="active", index=0)])
        return state

    @staticmethod
    def __step_result(
        *,
        action_type: ActionType,
        success: bool,
        screen_changed: bool,
    ) -> StepResult:
        action = Action(
            action_type=action_type,
            target="t",
            rationale="r",
            confidence=1.0,
        )
        step = Step(action=action, step_number=0, screen_hash="pre")
        return StepResult(
            step=step,
            success=success,
            duration=10,
            screen_changed=screen_changed,
            pre_hash="ph",
            post_hash="ph2" if screen_changed else "ph",
        )

    def test_default_count_zero(self) -> None:
        """
        Fresh state with an active sub-goal starts at zero deferrals.
        """

        state = self.__state_with_subgoal()
        self.assertEqual(state.deferral_count, 0)

    def test_record_deferral_increments(self) -> None:
        """
        :meth:`record_deferral` bumps the active sub-goal's counter.
        """

        state = self.__state_with_subgoal()
        state.record_deferral()
        state.record_deferral()
        self.assertEqual(state.deferral_count, 2)

    def test_clear_deferrals_resets(self) -> None:
        """
        :meth:`clear_deferrals` returns the counter to zero.
        """

        state = self.__state_with_subgoal()
        state.record_deferral()
        state.record_deferral()
        state.clear_deferrals()
        self.assertEqual(state.deferral_count, 0)

    def test_no_active_subgoal_record_is_noop(self) -> None:
        """
        Without an active sub-goal the helpers are silent no-ops.
        """

        state = AgentState(intent="x")
        state.record_deferral()
        self.assertEqual(state.deferral_count, 0)

    def test_record_step_clears_deferrals_on_navigation_progress(self) -> None:
        """
        A successful TAP with screen_changed=True clears the deferral count.
        """

        state = self.__state_with_subgoal()
        state.record_deferral()
        state.record_deferral()
        state.record_step(
            result=self.__step_result(
                action_type=ActionType.TAP, success=True, screen_changed=True
            )
        )
        self.assertEqual(state.deferral_count, 0)

    def test_record_step_preserves_deferrals_on_validate(self) -> None:
        """
        VALIDATE actions are passive and must not clear the deferral count.
        """

        state = self.__state_with_subgoal()
        state.record_deferral()
        state.record_step(
            result=self.__step_result(
                action_type=ActionType.VALIDATE, success=True, screen_changed=False
            )
        )
        self.assertEqual(state.deferral_count, 1)

    def test_record_step_preserves_deferrals_on_ask_user(self) -> None:
        """
        ASK_USER does not represent progress and must not clear the counter.
        """

        state = self.__state_with_subgoal()
        state.record_deferral()
        state.record_step(
            result=self.__step_result(
                action_type=ActionType.ASK_USER, success=True, screen_changed=False
            )
        )
        self.assertEqual(state.deferral_count, 1)

    def test_record_step_preserves_deferrals_on_wait(self) -> None:
        """
        WAIT is observation-only and must not clear the counter.
        """

        state = self.__state_with_subgoal()
        state.record_deferral()
        state.record_step(
            result=self.__step_result(
                action_type=ActionType.WAIT, success=True, screen_changed=False
            )
        )
        self.assertEqual(state.deferral_count, 1)

    def test_record_step_preserves_deferrals_when_no_screen_change(self) -> None:
        """
        Even a navigation action does not clear when screen_changed=False.
        """

        state = self.__state_with_subgoal()
        state.record_deferral()
        state.record_step(
            result=self.__step_result(
                action_type=ActionType.TAP, success=True, screen_changed=False
            )
        )
        self.assertEqual(state.deferral_count, 1)

    def test_record_step_preserves_deferrals_on_failure(self) -> None:
        """
        A failed action — even one that changed the screen — is not progress.
        """

        state = self.__state_with_subgoal()
        state.record_deferral()
        state.record_step(
            result=self.__step_result(
                action_type=ActionType.TAP, success=False, screen_changed=True
            )
        )
        self.assertEqual(state.deferral_count, 1)
