from __future__ import annotations

import unittest

from pydantic import ValidationError
from tests.integration.healing._fixtures import (
    FixtureExpectation,
    FixtureStep,
    TerminalStatus,
)


class FixtureStepValidationTest(unittest.TestCase):
    """
    Pins the typed validation invariants on FixtureStep schema.
    """

    @staticmethod
    def __valid_payload() -> dict:
        """
        Build a canonical valid FixtureStep payload.
        """

        return {
            "index": 0,
            "intent_segment": "Tap continue",
            "model_output": {"tool": "execute_ui", "arguments": {}},
            "frame": "frames/step_0.png",
            "manifest": "manifests/step_0.xml",
        }

    def test_valid_payload_round_trips(self) -> None:
        """
        A canonical step payload validates and round-trips into a frozen model.
        """

        step = FixtureStep.model_validate(self.__valid_payload())

        self.assertEqual(step.index, 0)
        self.assertEqual(step.frame, "frames/step_0.png")

    def test_negative_index_is_rejected(self) -> None:
        """
        Step indices must be non-negative; negative values fail validation.
        """

        payload = self.__valid_payload()
        payload["index"] = -1

        with self.assertRaises(ValidationError):
            FixtureStep.model_validate(payload)

    def test_extra_fields_rejected(self) -> None:
        """
        FixtureStep is strict; unknown fields fail validation.
        """

        payload = self.__valid_payload()
        payload["unexpected"] = "x"

        with self.assertRaises(ValidationError):
            FixtureStep.model_validate(payload)


class FixtureExpectationValidationTest(unittest.TestCase):
    """
    Pins the typed validation invariants on FixtureExpectation schema.
    """

    @staticmethod
    def __valid_payload() -> dict:
        """
        Build a canonical valid FixtureExpectation payload.
        """

        return {
            "terminal_status": "SUCCEEDED",
            "max_step_count": 10,
            "max_repeated_no_effect": 2,
            "block_reasons": ["KEYBOARD_OCCLUDING"],
            "recoveries_invoked": ["OverlayRecovery"],
            "raw_llm_coordinates_executed": 0,
        }

    def test_valid_payload_round_trips(self) -> None:
        """
        A canonical expectation payload validates and round-trips into a frozen model.
        """

        expectation = FixtureExpectation.model_validate(self.__valid_payload())

        self.assertEqual(expectation.terminal_status, TerminalStatus.SUCCEEDED)
        self.assertEqual(expectation.max_step_count, 10)

    def test_zero_max_step_count_is_rejected(self) -> None:
        """
        max_step_count must be at least one; zero or negative values fail validation.
        """

        payload = self.__valid_payload()
        payload["max_step_count"] = 0

        with self.assertRaises(ValidationError):
            FixtureExpectation.model_validate(payload)

    def test_unknown_terminal_status_is_rejected(self) -> None:
        """
        An unknown terminal status fails validation against the typed enum.
        """

        payload = self.__valid_payload()
        payload["terminal_status"] = "FLOWN_AWAY"

        with self.assertRaises(ValidationError):
            FixtureExpectation.model_validate(payload)
