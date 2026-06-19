from __future__ import annotations

import unittest

from fathom.core.prompts.decomposition import GeminiDecompositionPromptBuilder


class GeminiDecompositionPromptDirectiveTableTest(unittest.TestCase):
    """Pins the directive table the Decomposer offers the LLM."""

    def setUp(self) -> None:
        """Build the user prompt that carries the directive table."""

        self.__user_prompt = GeminiDecompositionPromptBuilder().build_user_prompt(
            intent="Change the address to times square new york"
        )

    def test_directive_table_omits_legacy_enter_row(self) -> None:
        """The directive table must not map 'Press enter' to a dedicated directive."""

        self.assertNotIn("'Press enter'", self.__user_prompt)
        self.assertNotIn("'Submit'", self.__user_prompt)
        self.assertNotIn("'Confirm via keyboard'", self.__user_prompt)

    def test_directive_union_omits_enter_token(self) -> None:
        """
        The JSON-output directive union must not list 'enter' as a valid token,
        anywhere in the union — head, middle, or tail.
        """

        for fragment in ("| enter ", " enter |", " enter>", "<enter ", "|enter|"):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.__user_prompt)

    def test_directive_table_retains_supported_directives(self) -> None:
        """
        Removing the 'enter' directive must not strip the supported directives from the table.
        """

        for directive in ("tap", "type", "back", "home", "hide_keyboard", "validate"):
            with self.subTest(directive=directive):
                self.assertIn(f"  {directive}", self.__user_prompt)
