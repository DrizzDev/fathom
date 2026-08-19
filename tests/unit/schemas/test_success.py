from __future__ import annotations

import unittest

from pydantic import TypeAdapter, ValidationError

from fathom.constants import ActionType
from fathom.constants.flow import SwipeDirection
from fathom.constants.success import CaptureNameProvenance, SuccessKind
from fathom.schemas.capture import CaptureIdentity
from fathom.schemas.requirement import PressRequirement, SwipeRequirement
from fathom.schemas.success import (
    CaptureSuccess,
    CommandSuccess,
    ObservationRequirement,
    ObservedSuccess,
    SourceLocation,
    SourceSpan,
    Success,
)


class SuccessUnionTest(unittest.TestCase):
    """
    Pins the exactly-one success union and its discriminated deserialization.
    """

    __adapter: TypeAdapter[Success] = TypeAdapter(Success)

    def test_observed_success_carries_observation_without_tactic(self) -> None:
        """
        Observed success is defined by an observation requirement, not a UI tactic.
        """

        success = ObservedSuccess(
            observation=ObservationRequirement(
                assertion="Search results for Ghar soaps are displayed"
            )
        )

        self.assertEqual(success.kind, SuccessKind.OBSERVED)
        self.assertEqual(
            success.observation.assertion, "Search results for Ghar soaps are displayed"
        )

    def test_command_success_requires_operation_and_located_source(self) -> None:
        """
        Command success names an explicit primitive and cites its located intent span.
        """

        success = CommandSuccess(
            requirement=PressRequirement(operation=ActionType.TAP, target="Login"),
            source=SourceSpan(quote="Tap Login", location=SourceLocation(start=0, end=9)),
        )

        self.assertEqual(success.requirement.operation, ActionType.TAP)
        self.assertEqual(success.source.quote, "Tap Login")
        self.assertIsNone(success.postcondition)

    def test_command_success_accepts_observation_postcondition(self) -> None:
        """
        A command postcondition reuses the observation requirement semantics.
        """

        success = CommandSuccess(
            requirement=SwipeRequirement(operation=ActionType.SWIPE, direction=SwipeDirection.UP),
            source=SourceSpan(
                quote="Swipe up to Settings", location=SourceLocation(start=0, end=20)
            ),
            postcondition=ObservationRequirement(assertion="Settings is visible"),
        )

        self.assertIsNotNone(success.postcondition)
        assert success.postcondition is not None
        self.assertEqual(success.postcondition.assertion, "Settings is visible")

    def test_capture_success_binds_canonical_capture_identity(self) -> None:
        """
        Capture success carries the canonical capture identity, not independent fields.
        """

        success = CaptureSuccess(
            target=CaptureIdentity(name="otp_code", provenance=CaptureNameProvenance.USER),
            subject="the verification code",
        )

        self.assertEqual(success.target.name, "otp_code")
        self.assertEqual(success.subject, "the verification code")

    def test_source_location_rejects_unordered_bounds(self) -> None:
        """
        A location whose end does not lie after its start is invalid.
        """

        with self.assertRaises(ValidationError):
            SourceLocation(start=5, end=3)

    def test_source_span_rejects_length_mismatch(self) -> None:
        """
        A span whose location length disagrees with its quote length is invalid.
        """

        with self.assertRaises(ValidationError):
            SourceSpan(quote="Tap Login", location=SourceLocation(start=0, end=4))

    def test_discriminator_selects_the_command_variant(self) -> None:
        """
        A payload tagged COMMAND deserializes to CommandSuccess via the discriminator.
        """

        success = self.__adapter.validate_python(
            {
                "kind": "COMMAND",
                "requirement": {"operation": "type", "target": "Search", "text": "Ghar soaps"},
                "source": {"quote": "Type Ghar soaps", "location": {"start": 0, "end": 15}},
            }
        )

        self.assertIsInstance(success, CommandSuccess)

    def test_discriminator_selects_the_capture_variant(self) -> None:
        """
        A payload tagged CAPTURE deserializes to CaptureSuccess via the discriminator.
        """

        success = self.__adapter.validate_python(
            {
                "kind": "CAPTURE",
                "target": {"name": "confirmation_id", "provenance": "USER"},
                "subject": "the confirmation ID",
            }
        )

        self.assertIsInstance(success, CaptureSuccess)

    def test_blank_assertion_is_rejected(self) -> None:
        """
        A blank observable assertion is not a valid observation requirement.
        """

        with self.assertRaises(ValidationError):
            ObservationRequirement(assertion="   ")

    def test_success_variants_are_frozen(self) -> None:
        """
        Success definitions are immutable value objects.
        """

        success = CaptureSuccess(
            target=CaptureIdentity(name="otp_code", provenance=CaptureNameProvenance.USER),
            subject="the verification code",
        )

        with self.assertRaises(ValidationError):
            success.target = CaptureIdentity(name="other", provenance=CaptureNameProvenance.USER)

    def test_unknown_fields_are_forbidden(self) -> None:
        """
        A success variant rejects unknown fields at the boundary.
        """

        with self.assertRaises(ValidationError):
            CommandSuccess(
                requirement=PressRequirement(operation=ActionType.TAP, target="Login"),
                source=SourceSpan(quote="Tap Login", location=SourceLocation(start=0, end=9)),
                tactic="tap",
            )
