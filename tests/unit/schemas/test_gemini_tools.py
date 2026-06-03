from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.schemas.gemini_tools import ExecuteAction


class ExecuteActionConditionalWaitTest(unittest.TestCase):
    """
    Pins the conditional-wait normalization the planner relies on.
    """

    @staticmethod
    def __payload(**overrides: object) -> dict[str, object]:
        """
        Build a baseline ExecuteAction payload callers can override per assertion.
        """

        baseline: dict[str, object] = {
            "confidence": 0.9,
            "action_type": "wait",
            "wait_subject": "Main menu",
        }
        baseline.update(overrides)

        return baseline

    def test_conditional_wait_without_condition_derives_from_subject(self) -> None:
        """
        When the planner marks a wait as conditional and forgets the condition,
        the validator synthesizes one from the wait subject.
        """

        action = ExecuteAction.model_validate(
            self.__payload(is_conditional=True),
        )

        self.assertEqual(action.conditional_type, "transient")
        self.assertEqual(action.condition, "Main menu is visible")

    def test_conditional_wait_with_explicit_condition_preserves_it(self) -> None:
        """
        An explicit condition takes precedence over the wait-subject default.
        """

        action = ExecuteAction.model_validate(
            self.__payload(
                is_conditional=True,
                conditional_type="blocker",
                condition="Search results are visible",
            ),
        )

        self.assertEqual(action.conditional_type, "blocker")
        self.assertEqual(action.condition, "Search results are visible")

    def test_non_wait_conditional_without_condition_still_raises(self) -> None:
        """
        The fail-fast guard remains for every action type other than wait.
        """

        with self.assertRaises(ValidationError):
            ExecuteAction.model_validate(
                {
                    "confidence": 0.9,
                    "action_type": "tap",
                    "is_conditional": True,
                    "target_name": "Continue",
                },
            )

    def test_wait_without_subject_still_raises(self) -> None:
        """
        wait_subject remains mandatory for wait actions even when not conditional.
        """

        with self.assertRaises(ValidationError):
            ExecuteAction.model_validate(
                {
                    
                    "confidence": 0.9,
                    "action_type": "wait",
                },
            )

    def test_overlay_detected_branch_unchanged(self) -> None:
        """
        overlay_detected still emits the canonical 'Overlay is visible' default.
        """

        action = ExecuteAction.model_validate(
            self.__payload(
                action_type="tap",
                target_name="Dismiss",
                overlay_detected=True,
                
            ),
        )

        self.assertTrue(action.is_conditional)
        self.assertEqual(action.conditional_type, "blocker")
        self.assertEqual(action.condition, "Overlay is visible")
        
        
