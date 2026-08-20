from __future__ import annotations

import unittest

from fathom.core.safety.classifier import IntentSafetyClassifier


class IntentSafetyClassifierTest(unittest.TestCase):
    """
    Pins the pre-execution intent-safety classifier: it screens the stated goal once, so
    legitimate swipe / type / tap actions are never blocked by a substring false positive.
    """

    def test_safe_intent_is_cleared(self) -> None:
        """
        A benign intent passes the classifier with no matched keyword.
        """

        verdict = IntentSafetyClassifier().classify(intent="Open Delivery app and search for dosa")

        self.assertTrue(verdict.safe)
        self.assertIsNone(verdict.matched_keyword)

    def test_destructive_intent_is_blocked_with_matched_token(self) -> None:
        """
        An intent containing a destructive keyword is flagged unsafe with
        the matched keyword surfaced for operator messaging.
        """

        verdict = IntentSafetyClassifier().classify(
            intent="Open the device settings and factory reset the phone",
        )

        self.assertFalse(verdict.safe)
        self.assertEqual(verdict.matched_keyword, "factory reset")

    def test_swipe_intent_is_not_falsely_blocked_by_wipe_substring(self) -> None:
        """
        The substring trap that motivated this relocation must not
        re-emerge: an intent that mentions ``swipe`` should be safe.
        """

        verdict = IntentSafetyClassifier().classify(
            intent="Scroll the search results and swipe up to reveal more",
        )

        self.assertTrue(verdict.safe)
