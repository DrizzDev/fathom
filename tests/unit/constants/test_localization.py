from __future__ import annotations

import unittest

from fathom.constants.localization import LocalizationGridScale


class LocalizationGridScaleTest(unittest.TestCase):
    """
    Pins the published edge bounds of the vision localizer grid.
    """

    def test_minimum_is_zero(self) -> None:
        """
        Minimum edge anchors the top-left of the normalized grid.
        """

        self.assertEqual(LocalizationGridScale.MINIMUM, 0)

    def test_maximum_is_one_thousand(self) -> None:
        """
        Maximum edge defines the grid resolution consumers project against.
        """

        self.assertEqual(LocalizationGridScale.MAXIMUM, 1000)

    def test_minimum_strictly_below_maximum(self) -> None:
        """
        The two bounds form a non-empty interval for the localizer payload.
        """

        self.assertLess(LocalizationGridScale.MINIMUM, LocalizationGridScale.MAXIMUM)
