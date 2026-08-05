from __future__ import annotations

import unittest

from fathom.core.prompts.decomposition import GeminiDecompositionPromptBuilder
from fathom.schemas.decomposition import DecomposedTask


class GeminiDecompositionPromptProposalTest(unittest.TestCase):
    """
    Pins the proposal contract the Decomposer offers the LLM: objective + one typed proposal.
    """

    def setUp(self) -> None:
        """
        Build the user prompt that carries the proposal vocabulary.
        """

        self.__user_prompt = GeminiDecompositionPromptBuilder().build_user_prompt(
            intent="Change the address to times square new york"
        )

    def test_prompt_declares_every_mandatory_schema_field(self) -> None:
        """
        The prompt names every field the DecomposedTask schema requires, keeping prompt and schema in
        agreement — objective and proposal are mandatory in both.
        """

        required = {
            name for name, field in DecomposedTask.model_fields.items() if field.is_required()
        }

        self.assertEqual(required, {"objective", "proposal"})
        for name in required:
            self.assertIn(
                f'"{name}"',
                self.__user_prompt,
                msg=f"decomposition prompt omits mandatory schema field {name!r}",
            )

    def test_prompt_declares_the_three_proposal_kinds(self) -> None:
        """
        The vocabulary declares exactly the observed, command, and capture success kinds.
        """

        for kind in ("OBSERVED", "COMMAND", "CAPTURE"):
            with self.subTest(kind=kind):
                self.assertIn(f'"kind": "{kind}"', self.__user_prompt)

    def test_prompt_lists_command_requirement_operations(self) -> None:
        """
        The command vocabulary enumerates the canonical requirement operations, using the exact
        lowercase ActionType enum values the structured-output schema constrains the model to.
        """

        for operation in ("tap", "type", "scroll", "swipe", "wait", "back"):
            with self.subTest(operation=operation):
                self.assertIn(f'"operation": "{operation}"', self.__user_prompt)

    def test_command_requires_exact_intent_quote(self) -> None:
        """
        A command proposal must cite an exact substring of the intent.
        """

        self.assertIn("exact substring of the intent", self.__user_prompt)

    def test_capture_clause_must_split_from_followup_action(self) -> None:
        """
        A store/capture clause is its own capture sub-goal, never merged into the follow-up action.
        """

        self.assertIn("its own 'capture' sub-goal", self.__user_prompt)
        self.assertIn('"kind": "CAPTURE"', self.__user_prompt)
        self.assertIn("Capture the verification code as otp_code", self.__user_prompt)

    def test_conditional_store_requires_separate_observed_precondition(self) -> None:
        """
        A checked precondition before capture decomposes into an observed step then a capture step.
        """

        self.assertIn(
            'GOOD: User says "Verify the balance is visible; if it is, store the balance as '
            'account_balance and open transactions"',
            self.__user_prompt,
        )
        self.assertIn('"objective": "Verify the balance is visible"', self.__user_prompt)
        self.assertIn('"objective": "Store the balance as account_balance"', self.__user_prompt)

    def test_bad_example_rejects_merged_capture(self) -> None:
        """
        The prompt must show that merging a capture clause with the following action is invalid.
        """

        self.assertIn(
            'BAD: User says "Capture the verification code as otp_code and continue"',
            self.__user_prompt,
        )
        self.assertIn("the capture clause was merged into the follow-up action", self.__user_prompt)

    def test_prompt_omits_legacy_directive_and_proof_contract(self) -> None:
        """
        The legacy directive/proof output contract must not survive anywhere in the prompt.
        """

        self.assertNotIn('"directive"', self.__user_prompt)
        self.assertNotIn('"proof"', self.__user_prompt)
        self.assertNotIn("DIRECTIVE FIELD", self.__user_prompt)
        self.assertNotIn("PROOF FIELD", self.__user_prompt)


if __name__ == "__main__":
    unittest.main()
