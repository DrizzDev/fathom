from __future__ import annotations

import unittest

from fathom.constants.screen import HitOutcome, ScreenCategory


class TestScreenCategory(unittest.TestCase):
    """ScreenCategory enumerates the functional kinds a screen can be classified as."""

    def test_values_are_lowercase_tokens(self) -> None:
        for category in ScreenCategory:
            self.assertEqual(category.value, category.value.lower())
            self.assertEqual(category.value, category.value.strip())

    def test_covers_the_sensitive_and_navigational_kinds(self) -> None:
        values = {category.value for category in ScreenCategory}

        self.assertEqual(
            values,
            {
                "home",
                "list",
                "detail",
                "form",
                "auth",
                "payment",
                "settings",
                "onboarding",
                "search",
                "other",
            },
        )

    def test_other_is_the_catch_all(self) -> None:
        self.assertEqual(ScreenCategory("other"), ScreenCategory.OTHER)


class TestHitOutcome(unittest.TestCase):
    """HitOutcome enumerates whether a tap landed inside an interactive element."""

    def test_values_are_lowercase_tokens(self) -> None:
        for outcome in HitOutcome:
            self.assertEqual(outcome.value, outcome.value.lower())

    def test_covers_hit_miss_and_unknown(self) -> None:
        self.assertEqual(
            {outcome.value for outcome in HitOutcome},
            {"hit", "miss", "unknown"},
        )


if __name__ == "__main__":
    unittest.main()
