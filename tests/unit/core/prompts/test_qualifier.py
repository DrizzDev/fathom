from __future__ import annotations

import unittest

from fathom.core.prompts.qualifier import GeminiQualifierPromptBuilder


class GeminiQualifierSystemInstructionTest(unittest.TestCase):
    """
    System instruction must teach the binary contract AND the outcome-verb /
    passive-declarative boundary that was missing from the original prompt.

    These assertions are deliberately string-level: they exist to keep future
    edits honest. If someone strips a section the model relies on, the test fails before the eval drift gets caught manually.
    """

    def setUp(self) -> None:
        """
        Build a fresh prompt builder for each test.
        """

        self.__builder = GeminiQualifierPromptBuilder()
        self.__system = self.__builder.build_system_instruction()

    def test_outcome_verb_imperative_section_present(self) -> None:
        """
        The outcome-verb imperative rule must be in the EXECUTABLE bullet list.
        Without this, intents like 'Complete the onboarding flow' get over-rejected.
        """

        self.assertIn("outcome-verb imperative", self.__system)
        self.assertIn("Complete the onboarding flow", self.__system)
        self.assertIn("Finish checkout", self.__system)

    def test_outcome_verb_constraint_carve_out_present(self) -> None:
        """
        Path constraints attached to the outcome must NARROW the route,
        not flip the verdict. The prompt must say so explicitly.
        """

        self.assertIn("NARROW", self.__system)
        self.assertIn("without a device", self.__system)

    def test_passive_declarative_rule_present_in_not_executable(self) -> None:
        """
        Passive declarative's (no imperative verb) must be the paired NOT_EXECUTABLE
        counterexample so the model has both sides of the boundary.
        """

        self.assertIn("passive declarative", self.__system.lower())
        self.assertIn("Login is complete", self.__system)
        self.assertIn("Order should be confirmed", self.__system)


class GeminiQualifierUserPromptTest(unittest.TestCase):
    """
    User prompt must include outcome-verb anchor examples (positive + paired
    passive-declarative negative) AND must carry the candidate intent verbatim.
    """

    def setUp(self) -> None:
        """
        Build a fresh prompt builder for each test.
        """

        self.__builder = GeminiQualifierPromptBuilder()

    def test_user_prompt_includes_outcome_verb_positive_anchor(self) -> None:
        """
        Positive anchor: 'Complete the onboarding flow' must appear as an
        EXECUTABLE example so the model learns the shape.
        """

        prompt = self.__builder.build_user_prompt(intent="anything")
        self.assertIn("Complete the onboarding flow", prompt)
        self.assertIn('"label": "EXECUTABLE"', prompt)

    def test_user_prompt_includes_constrained_outcome_anchor(self) -> None:
        """
        The exact false-positive that triggered this fix must be an explicit
        EXECUTABLE anchor in the prompt so the model never regresses on it.
        """

        prompt = self.__builder.build_user_prompt(intent="anything")
        self.assertIn("Complete the onboarding flow without a device", prompt)

    def test_user_prompt_includes_passive_declarative_negative_anchors(self) -> None:
        """
        Both passive-declarative anchors must be present so the model can
        distinguish them from outcome-verb imperatives.
        """

        prompt = self.__builder.build_user_prompt(intent="anything")
        self.assertIn("Login is complete", prompt)
        self.assertIn("Order should be confirmed", prompt)
        self.assertIn('"label": "NOT_EXECUTABLE"', prompt)

    def test_user_prompt_embeds_candidate_intent_verbatim(self) -> None:
        """
        Whatever the caller passes as intent must show up in the CANDIDATE block.
        Repr-quoted so adversarial inputs (quotes, newlines) survive transport.
        """

        prompt = self.__builder.build_user_prompt(intent="weird 'intent'\nwith newline")
        self.assertIn("CANDIDATE", prompt)
        self.assertIn("weird 'intent'", prompt)


if __name__ == "__main__":
    unittest.main()
