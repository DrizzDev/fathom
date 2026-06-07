from __future__ import annotations

import unittest

from fathom.core.prompts.abort import AbortPromptBuilder, GeminiAbortPromptBuilder


class GeminiAbortPromptBuilderTest(unittest.TestCase):
    """
    Pins the Gemini-specific abort prompt builder's instruction and user-prompt shape.
    """

    def setUp(self) -> None:
        """
        Build a fresh prompt builder for every test in the class.
        """

        self.__builder = GeminiAbortPromptBuilder()

    def test_subclass_of_abstract_builder(self) -> None:
        """
        Concrete builder implements the abstract :class:`AbortPromptBuilder` contract.
        """

        self.assertIsInstance(self.__builder, AbortPromptBuilder)

    def test_system_instruction_demands_binary_json(self) -> None:
        """
        System instruction must spell out the binary JSON contract enforced by the parser.
        """

        instruction = self.__builder.build_system_instruction()

        self.assertIn("JSON", instruction)
        self.assertIn("aborted", instruction)
        self.assertIn("confidence", instruction)

    def test_system_instruction_blocks_ui_directive_false_positives(self) -> None:
        """
        Instruction must explicitly call out UI-directive phrases as non-abort cases.
        """

        instruction = self.__builder.build_system_instruction()

        self.assertIn("UI", instruction)
        self.assertIn("tap on stop", instruction)

    def test_system_instruction_is_stable_across_calls(self) -> None:
        """
        The instruction is a deterministic class attribute, not a per-call construct.
        """

        self.assertEqual(
            self.__builder.build_system_instruction(),
            self.__builder.build_system_instruction(),
        )

    def test_user_prompt_returns_response_unchanged(self) -> None:
        """
        User-prompt builder is the identity over the operator's response.
        """

        self.assertEqual(
            self.__builder.build_user_prompt(response="close the execution"),
            "close the execution",
        )
