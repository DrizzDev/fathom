"""
Unit tests for tool-call request models.
"""

from __future__ import annotations

from fathom.schemas.tool_requests import ScreenTranslation


class TestScreenTranslation:
    """
    describe_screen output renders to functional markdown sections.
    """

    def test_renders_all_sections(self) -> None:
        md = ScreenTranslation.model_validate(
            {
                "activity_name": "com.x/.Home",
                "screen_purpose": "Home feed",
                "elements": "Top bar: Cart icon — opens cart",
                "achievable_actions": "Search for restaurants",
            }
        ).to_markdown()
        assert "**Activity:** `com.x/.Home`" in md
        assert "## Purpose\nHome feed" in md
        assert "## Elements\nTop bar: Cart icon — opens cart" in md
        assert "## What You Can Do\nSearch for restaurants" in md
        assert "Design Tokens" not in md
        assert "Blueprint" not in md

    def test_omits_empty_sections(self) -> None:
        md = ScreenTranslation.model_validate(
            {"activity_name": "a", "elements": "   "}
        ).to_markdown()
        assert md == "**Activity:** `a`"

    def test_coerces_null_fields(self) -> None:
        md = ScreenTranslation.model_validate(
            {"activity_name": "a", "screen_purpose": None, "elements": None}
        ).to_markdown()
        assert md == "**Activity:** `a`"

    def test_accepts_field_names_in_addition_to_aliases(self) -> None:
        md = ScreenTranslation(activity="a", purpose="p", elements="e", actions="x").to_markdown()
        assert "## Purpose\np" in md
        assert "## Elements\ne" in md
        assert "## What You Can Do\nx" in md
