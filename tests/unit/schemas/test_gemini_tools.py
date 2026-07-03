from __future__ import annotations

import unittest
from typing import Dict

from pydantic import ValidationError

from fathom.schemas.gemini_tools import ExecuteAction


class ExecuteActionStoreTest(unittest.TestCase):
    """
    Pins STORE capture enforcement: store requires a capture; capture is rejected elsewhere.
    """

    def test_store_capture_with_value_is_accepted(self) -> None:
        """
        A STORE with name, subject, and value parses into a CaptureRequest.
        """

        action = ExecuteAction.model_validate(
            {
                "action_type": "store",
                "confidence": 0.9,
                "capture": {"name": "abc", "subject": "price of soap", "value": "₹86"},
            }
        )

        assert action.capture is not None
        self.assertEqual(action.capture.value, "₹86")

    def test_store_capture_without_label_id_is_accepted(self) -> None:
        """
        STORE does not require XML or manifest grounding.
        """

        action = ExecuteAction.model_validate(
            {
                "action_type": "store",
                "confidence": 0.9,
                "capture": {
                    "name": "abc",
                    "subject": "price of the product",
                    "value": "₹499",
                },
            }
        )

        assert action.capture is not None
        self.assertEqual(action.capture.value, "₹499")

    def test_store_without_capture_is_rejected(self) -> None:
        """
        A STORE action without a capture request is rejected.
        """

        with self.assertRaises(ValidationError):
            ExecuteAction.model_validate({"action_type": "store", "confidence": 0.9})

    def test_store_capture_without_value_is_rejected(self) -> None:
        """
        A STORE capture must carry the concrete captured value.
        """

        with self.assertRaises(ValidationError):
            ExecuteAction.model_validate(
                {
                    "action_type": "store",
                    "confidence": 0.9,
                    "capture": {"name": "abc", "subject": "price"},
                }
            )

    def test_store_capture_blank_value_is_rejected(self) -> None:
        """
        Whitespace is not a captured value.
        """

        with self.assertRaises(ValidationError):
            ExecuteAction.model_validate(
                {
                    "action_type": "store",
                    "confidence": 0.9,
                    "capture": {"name": "abc", "subject": "price", "value": "   "},
                }
            )

    def test_store_capture_missing_name_is_rejected(self) -> None:
        """
        A capture request must carry a non-empty name.
        """

        with self.assertRaises(ValidationError):
            ExecuteAction.model_validate(
                {
                    "action_type": "store",
                    "confidence": 0.9,
                    "capture": {"subject": "xyz", "value": "₹86"},
                }
            )

    def test_capture_on_non_store_action_is_rejected(self) -> None:
        """
        Capture is only valid for store; a tap carrying a capture is rejected.
        """

        with self.assertRaises(ValidationError):
            ExecuteAction.model_validate(
                {
                    "action_type": "tap",
                    "confidence": 0.9,
                    "capture": {"name": "abc", "subject": "xyz", "value": "₹86"},
                }
            )


class ExecuteActionValidationTest(unittest.TestCase):
    """
    Pins validate action subject requirements.
    """

    def test_validate_without_validation_subject_is_rejected(self) -> None:
        """
        A validate action must carry a structured assertion subject.
        """

        with self.assertRaises(ValidationError):
            ExecuteAction.model_validate(
                {
                    "action_type": "validate",
                    "confidence": 0.9,
                    "target_name": "Phone Number",
                    "export_target": "Phone Number input field",
                }
            )

    def test_validate_with_validation_subject_is_accepted(self) -> None:
        """
        A validate action may carry visible anchor targets plus the assertion subject.
        """

        action = ExecuteAction.model_validate(
            {
                "action_type": "validate",
                "confidence": 0.9,
                "target_name": "Phone Number",
                "export_target": "Phone Number input field",
                "validation_subject": "Login screen",
            }
        )

        self.assertEqual(action.validation_subject, "Login screen")


class ExecuteActionConditionalWaitTest(unittest.TestCase):
    """
    Pins the conditional-wait normalization the planner relies on.
    """

    @staticmethod
    def __payload(**overrides: object) -> Dict[str, object]:
        """
        Build a baseline ExecuteAction payload callers can override per assertion.
        """

        baseline: Dict[str, object] = {
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

    def test_overlay_detected_without_condition_is_rejected(self) -> None:
        """
        overlay_detected must not invent a generic branch condition.
        """

        with self.assertRaises(ValidationError):
            ExecuteAction.model_validate(
                self.__payload(
                    action_type="tap",
                    target_name="Dismiss",
                    overlay_detected=True,
                ),
            )

    def test_overlay_detected_with_explicit_condition_sets_blocker_type(self) -> None:
        """
        overlay_detected may default the condition type, but never the condition text.
        """

        action = ExecuteAction.model_validate(
            self.__payload(
                action_type="tap",
                target_name="Dismiss",
                condition="Account chooser dialog is visible",
                overlay_detected=True,
            ),
        )

        self.assertTrue(action.is_conditional)
        self.assertEqual(action.conditional_type, "blocker")
        self.assertEqual(action.condition, "Account chooser dialog is visible")
