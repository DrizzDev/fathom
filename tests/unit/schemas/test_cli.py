from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.schemas.cli import ExploreCommandInput


class ExploreCommandInputTest(unittest.TestCase):
    """
    Validate the package identifier and focus accepted by `fathom explore`.
    """

    def test_accepts_dotted_package_identifier(self) -> None:
        """
        A dotted reverse-DNS identifier is accepted and trimmed.
        """

        command_input = ExploreCommandInput.model_validate({"package_name": "  ai.hangjam.app  "})

        self.assertEqual(command_input.package_name, "ai.hangjam.app")

    def test_defaults_to_none_when_absent(self) -> None:
        """
        Omitting the package leaves the target unset for foreground exploration.
        """

        self.assertIsNone(ExploreCommandInput().package_name)

    def test_rejects_shell_unsafe_identifier(self) -> None:
        """
        A value carrying shell metacharacters is rejected at the boundary.
        """

        with self.assertRaises(ValidationError):
            ExploreCommandInput.model_validate({"package_name": "com.evil; rm -rf /"})

    def test_accepts_free_text_focus(self) -> None:
        """
        A free-text focus is accepted and trimmed.
        """

        command_input = ExploreCommandInput.model_validate({"focus": "  map login flow  "})

        self.assertEqual(command_input.focus, "map login flow")

    def test_focus_defaults_to_none_when_absent(self) -> None:
        """
        Omitting the focus leaves the exploration goal generic.
        """

        self.assertIsNone(ExploreCommandInput().focus)

    def test_blank_focus_normalizes_to_none(self) -> None:
        """
        A whitespace-only focus normalizes to None rather than an empty goal.
        """

        self.assertIsNone(ExploreCommandInput.model_validate({"focus": "   "}).focus)

    def test_focus_allows_prose_with_punctuation(self) -> None:
        """
        Focus is prose for the LLM, so punctuation and spaces are not rejected.
        """

        command_input = ExploreCommandInput.model_validate(
            {"focus": "Focus on the checkout flow, then settings."}
        )

        self.assertEqual(command_input.focus, "Focus on the checkout flow, then settings.")
