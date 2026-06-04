from __future__ import annotations

import unittest

from fathom.constants.phase import PhaseKind


class PhaseKindTest(unittest.TestCase):
    """
    Pins the PhaseKind enum surface used by the announcer and downstream clients.
    """

    def test_members_are_string_valued(self) -> None:
        """
        Every PhaseKind member's value equals its name so client wire format is the enum name string.
        """

        for kind in PhaseKind:
            self.assertEqual(kind.value, kind.name)

    def test_qualifying_member_present(self) -> None:
        """
        QUALIFYING is defined; PhaseAnnouncer tags qualifier heartbeats with it.
        """

        self.assertEqual(PhaseKind.QUALIFYING.value, "QUALIFYING")

    def test_decomposing_member_present(self) -> None:
        """
        DECOMPOSING is defined; PhaseAnnouncer tags decomposer heartbeats with it.
        """

        self.assertEqual(PhaseKind.DECOMPOSING.value, "DECOMPOSING")

    def test_grounding_member_present(self) -> None:
        """
        GROUNDING is defined; PhaseAnnouncer tags GROUND-node heartbeats with it.
        """

        self.assertEqual(PhaseKind.GROUNDING.value, "GROUNDING")


if __name__ == "__main__":
    unittest.main()
