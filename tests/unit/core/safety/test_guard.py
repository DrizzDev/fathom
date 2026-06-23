from __future__ import annotations

import unittest

from fathom.constants.safety import SensitiveCategory
from fathom.constants.screen import ScreenCategory
from fathom.core.safety.guard import TraversalGuard


class TraversalGuardActionTest(unittest.TestCase):
    """The guard vetoes actions whose text enters a sensitive area."""

    def setUp(self) -> None:
        self.__guard = TraversalGuard()

    def test_payment_action_is_blocked(self) -> None:
        verdict = self.__guard.inspect_action(
            target="Proceed to Pay button", rationale="open the checkout"
        )

        self.assertFalse(verdict.allowed)
        self.assertEqual(verdict.category, SensitiveCategory.PAYMENT)
        self.assertIsNotNone(verdict.reason)

    def test_auth_action_is_blocked(self) -> None:
        verdict = self.__guard.inspect_action(target="Login", rationale="enter the account")

        self.assertFalse(verdict.allowed)
        self.assertEqual(verdict.category, SensitiveCategory.AUTH)

    def test_destructive_action_is_blocked(self) -> None:
        verdict = self.__guard.inspect_action(target="Delete account", rationale="")

        self.assertFalse(verdict.allowed)
        self.assertEqual(verdict.category, SensitiveCategory.DESTRUCTIVE)

    def test_benign_action_is_allowed(self) -> None:
        verdict = self.__guard.inspect_action(target="Restaurant card", rationale="open the menu")

        self.assertTrue(verdict.allowed)
        self.assertIsNone(verdict.reason)

    def test_describable_payment_word_does_not_false_match(self) -> None:
        # Bare "payment" is intentionally not a keyword, so browsing a payment
        # screen the crawl only describes does not trip the action veto.
        verdict = self.__guard.inspect_action(target="View payment options", rationale="")

        self.assertTrue(verdict.allowed)

    def test_custom_denylist_overrides_the_defaults(self) -> None:
        guard = TraversalGuard(denylist={SensitiveCategory.AUTH: frozenset({"biometric"})})

        self.assertFalse(guard.inspect_action(target="Biometric unlock").allowed)
        self.assertTrue(guard.inspect_action(target="Login").allowed)


class TraversalGuardScreenTest(unittest.TestCase):
    """Auth and payment screen categories are sensitive areas."""

    def test_auth_and_payment_are_sensitive(self) -> None:
        self.assertTrue(TraversalGuard.is_sensitive_screen(category=ScreenCategory.AUTH))
        self.assertTrue(TraversalGuard.is_sensitive_screen(category=ScreenCategory.PAYMENT))

    def test_other_categories_are_not_sensitive(self) -> None:
        for category in (ScreenCategory.HOME, ScreenCategory.LIST, ScreenCategory.DETAIL):
            self.assertFalse(TraversalGuard.is_sensitive_screen(category=category))


if __name__ == "__main__":
    unittest.main()
