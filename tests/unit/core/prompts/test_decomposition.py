from __future__ import annotations

import unittest

from fathom.core.prompts.decomposition import GeminiDecompositionPromptBuilder


class GeminiDecompositionPromptDirectiveTableTest(unittest.TestCase):
    """
    Pins the directive table the Decomposer offers the LLM.
    """

    def setUp(self) -> None:
        """
        Build the user prompt that carries the directive table.
        """

        self.__user_prompt = GeminiDecompositionPromptBuilder().build_user_prompt(
            intent="Change the address to times square new york"
        )

    def test_directive_table_omits_legacy_enter_row(self) -> None:
        """
        The directive table must not map 'Press enter' to a dedicated directive.
        """

        self.assertNotIn("'Submit'", self.__user_prompt)
        self.assertNotIn("'Press enter'", self.__user_prompt)
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

        for directive in ("tap", "type", "back", "home", "hide_keyboard", "validate", "store"):
            with self.subTest(directive=directive):
                self.assertIn(f"  {directive}", self.__user_prompt)

    def test_store_clause_must_split_from_followup_action(self) -> None:
        """
        A store/capture clause must remain its own directive instead of final-action-wins.
        """

        self.assertIn("store/capture {{value_or_subject}} as {{variable_name}}", self.__user_prompt)
        self.assertIn('"directive": "store"', self.__user_prompt)
        self.assertIn("split it into its own 'store' sub-goal", self.__user_prompt)
        self.assertIn("Capture the verification code as otp_code", self.__user_prompt)

    def test_prompt_placeholders_use_template_braces(self) -> None:
        """
        Prompt placeholders should be template-like, not XML-like tags.
        """

        self.assertIn("{{target}}", self.__user_prompt)
        self.assertNotIn("<target>", self.__user_prompt)
        self.assertIn("{{condition}}", self.__user_prompt)
        self.assertNotIn("<condition>", self.__user_prompt)

    def test_store_bad_example_rejects_merged_tap(self) -> None:
        """
        The prompt must show that merging STORE with the following TAP is invalid.
        """

        self.assertIn(
            'BAD: User says "Capture the verification code as otp_code and continue"',
            self.__user_prompt,
        )
        self.assertIn("the store clause was merged into the follow-up action", self.__user_prompt)

    def test_conditional_store_requires_separate_validation_step(self) -> None:
        """
        A checked precondition before STORE must decompose into VALIDATE then STORE.
        """

        self.assertIn(
            'GOOD: User says "Verify the balance is visible; if it is, store the balance as '
            'account_balance and open transactions"',
            self.__user_prompt,
        )
        self.assertIn('"description": "Verify the balance is visible"', self.__user_prompt)
        self.assertIn('"directive": "validate"', self.__user_prompt)
        self.assertIn(
            '"description": "If the balance is visible, store the balance as account_balance"',
            self.__user_prompt,
        )
        self.assertIn('"directive": "store"', self.__user_prompt)
        self.assertIn('"description": "Open transactions"', self.__user_prompt)

    def test_store_bad_example_rejects_merged_precondition(self) -> None:
        """
        The prompt must reject merging the precondition check into the STORE sub-goal.
        """

        self.assertIn(
            'BAD: User says "Verify the balance is visible; if it is, store the balance as '
            'account_balance"',
            self.__user_prompt,
        )
        self.assertIn(
            '"description": "Verify the balance is visible if it is, store the balance '
            'as account_balance"',
            self.__user_prompt,
        )
        self.assertIn(
            "the prerequisite validation was merged into the store command", self.__user_prompt
        )
