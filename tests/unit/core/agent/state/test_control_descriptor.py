from __future__ import annotations

import unittest

from tests.builders import ActionFixtures, AgentFixtures

from fathom.constants import ActionType
from fathom.schemas.steps import Step, StepResult


class AgentStateControlDescriptorTest(unittest.TestCase):
    """
    Pins :meth:`AgentState.record_step` last-action descriptor behavior.
    """

    @staticmethod
    def __step_result(*, action_type: ActionType, target: str = "t") -> StepResult:
        """
        Build a step-result fixture for the given action type.
        """

        action = ActionFixtures.make(
            target=target, action_type=action_type, rationale="r", confidence=1.0
        )
        step = Step(action=action, step_number=0, screen_hash="pre")
        return StepResult(
            step=step,
            duration=10,
            success=True,
            pre_hash="ph",
            post_hash="ph",
            screen_changed=False,
        )

    def test_device_action_records_its_descriptor(self) -> None:
        """
        Device actions record action_type + descriptor (regression guard).
        """

        state = AgentFixtures.state(intent="x")
        state.record_step(self.__step_result(action_type=ActionType.TAP, target="Submit"))

        self.assertEqual(state.last_action_type, ActionType.TAP.value)

    def test_ask_user_records_descriptor_instead_of_wiping(self) -> None:
        """
        ASK_USER is a control action whose descriptor must survive ``record_step``.
        """

        state = AgentFixtures.state(intent="x")
        state.record_step(self.__step_result(action_type=ActionType.ASK_USER, target="user help"))

        self.assertEqual(state.last_action_type, ActionType.ASK_USER.value)
        self.assertIsNotNone(state.last_action_type)

    def test_complete_records_descriptor_and_marks_complete(self) -> None:
        """
        COMPLETE must preserve its descriptor and flag the workflow complete.
        """

        state = AgentFixtures.state(intent="x")
        state.record_step(self.__step_result(action_type=ActionType.COMPLETE, target="end test"))

        self.assertTrue(state.is_complete)
        self.assertEqual(state.last_action_type, ActionType.COMPLETE.value)

    def test_validate_records_descriptor(self) -> None:
        """
        VALIDATE control actions also preserve descriptor for LoopDetector use.
        """

        state = AgentFixtures.state(intent="x")
        state.record_step(
            self.__step_result(action_type=ActionType.VALIDATE, target="welcome banner")
        )

        self.assertEqual(state.last_action_type, ActionType.VALIDATE.value)
