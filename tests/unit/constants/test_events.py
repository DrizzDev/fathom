from __future__ import annotations

import unittest

from fathom.constants.events import FathomEvent


class FathomEventTest(unittest.TestCase):
    """
    Pins the FathomEvent enum surface; client integrations depend on these wire values.
    """

    def test_plan_synthesized_value(self) -> None:
        """
        PLAN_SYNTHESIZED is the terminal event for the decomposer phase.
        """

        self.assertEqual(FathomEvent.PLAN_SYNTHESIZED.value, "PLAN_SYNTHESIZED")

    def test_phase_heartbeat_value(self) -> None:
        """
        PHASE_HEARTBEAT is the periodic keepalive emitted while a long phase runs.
        """

        self.assertEqual(FathomEvent.PHASE_HEARTBEAT.value, "PHASE_HEARTBEAT")

    def test_grounding_event_present(self) -> None:
        """
        GROUNDING is the per-step entry event for the GROUND node.
        """

        self.assertEqual(FathomEvent.GROUNDING.value, "GROUNDING")

    def test_intent_phase_events_present(self) -> None:
        """
        Qualifier and decomposer entry events are exposed on the enum.
        """

        self.assertEqual(FathomEvent.INTENT_QUALIFYING.value, "INTENT_QUALIFYING")
        self.assertEqual(FathomEvent.INTENT_DECOMPOSING.value, "INTENT_DECOMPOSING")

    def test_plan_derived_is_no_longer_a_member(self) -> None:
        """
        PLAN_DERIVED was renamed to PLAN_SYNTHESIZED; the old name must not coexist.
        """

        names = {member.name for member in FathomEvent}
        self.assertNotIn("PLAN_DERIVED", names)


if __name__ == "__main__":
    unittest.main()
