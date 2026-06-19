from __future__ import annotations

import unittest

from fathom.core.runtime.identity import TargetIdentity


class TargetIdentityTest(unittest.TestCase):
    """
    Pins for TargetIdentity normalization and equality semantics.
    """

    def test_normalize_lowercases_and_collapses_whitespace(self) -> None:
        """
        normalize() lowercases the description and collapses internal whitespace.
        """

        self.assertEqual(
            TargetIdentity.normalize(description="  Continue   Button  "),
            "continue button",
        )

    def test_normalize_strips_trailing_punctuation(self) -> None:
        """
        normalize() strips trailing sentence punctuation.
        """

        self.assertEqual(
            TargetIdentity.normalize(description="Submit!"),
            "submit",
        )
        self.assertEqual(
            TargetIdentity.normalize(description="Submit?"),
            "submit",
        )
        self.assertEqual(
            TargetIdentity.normalize(description="Submit."),
            "submit",
        )

    def test_describes_same_target_true_for_cosmetic_variants(self) -> None:
        """
        describes_same_target() is true when descriptions vary only by case and spacing.
        """

        self.assertTrue(
            TargetIdentity.describes_same_target(
                previous="Continue Button",
                replacement="  continue   button  ",
            )
        )

    def test_describes_same_target_false_for_genuine_differences(self) -> None:
        """
        describes_same_target() is false when surface forms differ semantically.
        """

        self.assertFalse(
            TargetIdentity.describes_same_target(
                previous="Continue Button",
                replacement="Cancel Button",
            )
        )
