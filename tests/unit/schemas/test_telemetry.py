from __future__ import annotations

import unittest

import pydantic

from fathom.schemas.telemetry import HeartbeatBudget, IntentMessage, PhaseMessage, StepMessage


class IntentMessageTest(unittest.TestCase):
    """
    Pins IntentMessage defaults and immutability.
    """

    def test_defaults_carry_user_facing_strings(self) -> None:
        """
        Default IntentMessage carries non-empty messages for qualifying, decomposing, and derived phases.
        """

        message = IntentMessage()

        self.assertTrue(message.qualifying.strip())
        self.assertTrue(message.decomposing.strip())
        self.assertTrue(message.derived.strip())

    def test_model_is_frozen(self) -> None:
        """
        IntentMessage is frozen so deployment-time messages cannot be mutated at runtime.
        """

        message = IntentMessage()

        with self.assertRaises(pydantic.ValidationError):
            message.qualifying = "mutated"  # type: ignore[misc]


class HeartbeatBudgetTest(unittest.TestCase):
    """
    Pins HeartbeatBudget defaults, bounds, and immutability.
    """

    def test_defaults_within_published_bounds(self) -> None:
        """
        Default budget threshold and limit fall inside the configured Pydantic ranges.
        """

        budget = HeartbeatBudget()

        self.assertGreaterEqual(budget.threshold, 0.5)
        self.assertLessEqual(budget.threshold, 60.0)
        self.assertGreaterEqual(budget.limit, 1)
        self.assertLessEqual(budget.limit, 600)
        self.assertTrue(budget.message.strip())

    def test_threshold_below_floor_rejected(self) -> None:
        """
        Threshold below the configured floor is rejected at construction time.
        """

        with self.assertRaises(pydantic.ValidationError):
            HeartbeatBudget.model_validate({"threshold": 0.1})

    def test_threshold_above_ceiling_rejected(self) -> None:
        """
        Threshold above the configured ceiling is rejected at construction time.
        """

        with self.assertRaises(pydantic.ValidationError):
            HeartbeatBudget.model_validate({"threshold": 120.0})

    def test_limit_below_floor_rejected(self) -> None:
        """
        Limit of zero is rejected so the pulse loop always executes at least once.
        """

        with self.assertRaises(pydantic.ValidationError):
            HeartbeatBudget.model_validate({"limit": 0})

    def test_limit_above_ceiling_rejected(self) -> None:
        """
        Limit above the configured ceiling is rejected so the loop can never run unbounded.
        """

        with self.assertRaises(pydantic.ValidationError):
            HeartbeatBudget.model_validate({"limit": 1000})


class StepMessageTest(unittest.TestCase):
    """
    Pins StepMessage defaults and immutability.
    """

    def test_default_grounding_string_present(self) -> None:
        """
        Default StepMessage carries a non-empty grounding string for the GROUND node.
        """

        self.assertTrue(StepMessage().grounding.strip())

    def test_model_is_frozen(self) -> None:
        """
        StepMessage is frozen so deployment-time messages cannot be mutated at runtime.
        """

        message = StepMessage()
        with self.assertRaises(pydantic.ValidationError):
            message.grounding = "mutated"  # type: ignore[misc]


class PhaseMessageTest(unittest.TestCase):
    """
    Pins PhaseMessage composition of intent and heartbeat sub-models.
    """

    def test_defaults_compose_nested_models(self) -> None:
        """
        PhaseMessage default composes IntentMessage, StepMessage, and HeartbeatBudget defaults.
        """

        message = PhaseMessage()

        self.assertIsInstance(message.step, StepMessage)
        self.assertIsInstance(message.intent, IntentMessage)
        self.assertIsInstance(message.heartbeat, HeartbeatBudget)

    def test_explicit_overrides_propagate(self) -> None:
        """
        Explicit nested overrides are honoured rather than masked by the defaults.
        """

        message = PhaseMessage(
            intent=IntentMessage(qualifying="custom"),
            heartbeat=HeartbeatBudget(threshold=10.0, limit=5, message="alive"),
        )

        self.assertEqual(message.heartbeat.limit, 5)
        self.assertEqual(message.heartbeat.threshold, 10.0)
        self.assertEqual(message.heartbeat.message, "alive")
        self.assertEqual(message.intent.qualifying, "custom")


if __name__ == "__main__":
    unittest.main()
