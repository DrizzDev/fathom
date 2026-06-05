from __future__ import annotations

import unittest

from fathom.schemas.actions import CoordinateSource


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


if __name__ == "__main__":
    unittest.main()
