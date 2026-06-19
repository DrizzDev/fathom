from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants import ActionType
from fathom.schemas.actions import Action, CoordinateSource


class CoordinateSourceTest(unittest.TestCase):
    """
    Pins the stable wire values for every :class:`CoordinateSource` member.

    Production log filters and downstream cloud sinks attribute taps by these
    string values. A rename or accidental removal of any member would corrupt
    historical dashboards; this test breaks the build first.
    """

    def test_every_member_carries_its_canonical_wire_value(self) -> None:
        """
        Verify the StrEnum value for each cascade-stage tag.
        """

        self.assertEqual(CoordinateSource.XML.value, "xml")
        self.assertEqual(CoordinateSource.OCR.value, "ocr")
        self.assertEqual(CoordinateSource.MODEL.value, "model")
        self.assertEqual(CoordinateSource.MODEL_GROUNDED.value, "model_grounded")
        self.assertEqual(CoordinateSource.VISION.value, "vision")
        self.assertEqual(CoordinateSource.VIEWPORT.value, "viewport")

    def test_member_count_pins_total_surface(self) -> None:
        """
        Six members today; adding or removing one without an explicit migration
        plan must surface as a test failure.
        """

        self.assertEqual(len(CoordinateSource), 6)

    def test_lookup_by_value_round_trips_for_every_member(self) -> None:
        """
        ``CoordinateSource('vision')`` and friends round-trip; pins that no
        member loses its wire-side parser entry.
        """

        for member in CoordinateSource:
            self.assertEqual(CoordinateSource(member.value), member)


class ActionLegacyEnterRejectionTest(unittest.TestCase):
    """
    Pins that the deprecated 'enter' action_type is rejected, never silently coerced.
    """

    def test_action_type_enum_does_not_expose_enter(self) -> None:
        """
        The ENTER member must no longer be reachable through ActionType.
        """

        self.assertFalse(hasattr(ActionType, "ENTER"))
        with self.assertRaises(ValueError):
            ActionType("enter")

    def test_model_validate_rejects_legacy_enter_payload(self) -> None:
        """
        A historical Action payload with action_type='enter' must fail validation.
        """

        with self.assertRaises(ValidationError):
            Action.model_validate(
                {
                    "action_type": "enter",
                    "target": "Search",
                    "rationale": "submit search",
                    "confidence": 0.9,
                }
            )

    def test_model_validate_rejects_legacy_enter_payload_with_anchors(self) -> None:
        """
        Even with label_id/bounds present, 'enter' must not deserialize into a TAP.
        """

        with self.assertRaises(ValidationError):
            Action.model_validate(
                {
                    "action_type": "ENTER",
                    "target": "Search",
                    "rationale": "submit search",
                    "label_id": "149",
                    "confidence": 0.9,
                }
            )

    def test_model_validate_accepts_supported_action_types(self) -> None:
        """
        Supported action_types still deserialize cleanly after the shim removal.
        """

        action = Action.model_validate(
            {
                "action_type": "tap",
                "target": "Login",
                "rationale": "submit login",
                "label_id": "12",
                "confidence": 1.0,
            }
        )

        self.assertIs(action.action_type, ActionType.TAP)
        self.assertEqual(action.label_id, "12")


if __name__ == "__main__":
    unittest.main()
