from __future__ import annotations

import unittest

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
