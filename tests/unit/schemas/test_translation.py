from __future__ import annotations

import unittest

from fathom.constants.screen import ScreenCategory
from fathom.schemas.translation import ScreenTranslation


class TestScreenTranslation(unittest.TestCase):
    """describe_screen output renders to functional markdown sections."""

    def test_renders_all_sections(self) -> None:
        markdown = ScreenTranslation.model_validate(
            {
                "activity_name": "com.x/.Home",
                "screen_purpose": "Home feed",
                "elements": "Top bar: Cart icon - opens cart",
                "achievable_actions": "Search for restaurants",
            }
        ).to_markdown()

        self.assertIn("**Activity:** `com.x/.Home`", markdown)
        self.assertIn("## Purpose\nHome feed", markdown)
        self.assertIn("## Elements\nTop bar: Cart icon - opens cart", markdown)
        self.assertIn("## What You Can Do\nSearch for restaurants", markdown)
        self.assertNotIn("Design Tokens", markdown)
        self.assertNotIn("Blueprint", markdown)

    def test_omits_empty_sections(self) -> None:
        markdown = ScreenTranslation.model_validate(
            {"activity_name": "a", "elements": "   "}
        ).to_markdown()

        self.assertEqual(markdown, "**Activity:** `a`")

    def test_coerces_null_fields(self) -> None:
        markdown = ScreenTranslation.model_validate(
            {"activity_name": "a", "screen_purpose": None, "elements": None}
        ).to_markdown()

        self.assertEqual(markdown, "**Activity:** `a`")

    def test_accepts_field_names_and_aliases(self) -> None:
        markdown = ScreenTranslation(
            activity="a", purpose="p", elements="e", actions="x"
        ).to_markdown()

        self.assertIn("## Purpose\np", markdown)
        self.assertIn("## Elements\ne", markdown)
        self.assertIn("## What You Can Do\nx", markdown)

    def test_parses_screen_category_from_alias(self) -> None:
        translation = ScreenTranslation.model_validate(
            {"activity_name": "a", "screen_category": "payment"}
        )

        self.assertEqual(translation.category, ScreenCategory.PAYMENT)

    def test_unknown_category_coerces_to_other(self) -> None:
        translation = ScreenTranslation.model_validate(
            {"activity_name": "a", "screen_category": "frobnicate"}
        )

        self.assertEqual(translation.category, ScreenCategory.OTHER)

    def test_missing_category_defaults_to_other(self) -> None:
        translation = ScreenTranslation.model_validate({"activity_name": "a"})

        self.assertEqual(translation.category, ScreenCategory.OTHER)

    def test_category_is_not_rendered_into_markdown(self) -> None:
        markdown = ScreenTranslation.model_validate(
            {"activity_name": "a", "screen_category": "home"}
        ).to_markdown()

        self.assertEqual(markdown, "**Activity:** `a`")

    def test_to_content_splits_per_line_fields_into_entries(self) -> None:
        content = ScreenTranslation.model_validate(
            {
                "activity_name": "com.x/.Home",
                "screen_purpose": "Home feed",
                "elements": "Cart icon - opens cart\n  Search field  \n\nProfile tab",
                "achievable_actions": "Search for restaurants\nOpen the cart",
            }
        ).to_content()

        self.assertEqual(content.purpose, "Home feed")
        self.assertEqual(
            content.elements, ["Cart icon - opens cart", "Search field", "Profile tab"]
        )
        self.assertEqual(content.actions, ["Search for restaurants", "Open the cart"])

    def test_to_content_empty_fields_yield_empty_lists(self) -> None:
        content = ScreenTranslation.model_validate(
            {"activity_name": "a", "elements": "   "}
        ).to_content()

        self.assertEqual(content.purpose, "")
        self.assertEqual(content.elements, [])
        self.assertEqual(content.actions, [])
