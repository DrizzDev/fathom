from __future__ import annotations

import unittest

from fathom.schemas.content import ScreenContent


class TestScreenContent(unittest.TestCase):
    """The structured screen-content value object keeps purpose, elements, and actions discrete."""

    def test_defaults_to_empty_structure(self) -> None:
        content = ScreenContent()

        self.assertEqual(content.purpose, "")
        self.assertEqual(content.elements, [])
        self.assertEqual(content.actions, [])

    def test_retains_discrete_fields(self) -> None:
        content = ScreenContent(
            purpose="Review and confirm the booking",
            elements=["'Cancel' button (top left)", "'Secure booking' indicator (top right)"],
            actions=["Review doctor details", "Enter member ID"],
        )

        self.assertEqual(content.purpose, "Review and confirm the booking")
        self.assertEqual(len(content.elements), 2)
        self.assertEqual(content.actions[1], "Enter member ID")

    def test_round_trips_through_json(self) -> None:
        content = ScreenContent(
            purpose="Search for items",
            elements=["Search field", "Filter chips"],
            actions=["Search the catalogue"],
        )

        restored = ScreenContent.model_validate_json(content.model_dump_json())

        self.assertEqual(restored, content)
