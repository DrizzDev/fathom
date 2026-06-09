from __future__ import annotations

import unittest

from fathom.core.prompts.rejection import RepeatedFailureRejectionPromptBuilder


class RepeatedFailureRejectionPromptBuilderInteractiveTest(unittest.TestCase):
    """
    Pins the interactive-mode rejection sentence shape; ask_user must be offered as a valid escape.
    """

    def test_interactive_prompt_offers_ask_user(self) -> None:
        """
        With HITL enabled the LLM must be told ask_user is available; otherwise it has no escape from the rejection loop.
        """

        text = RepeatedFailureRejectionPromptBuilder.build(
            action_descriptor="Swipe left on More on Swiggy",
            interactive=True,
        )

        self.assertIn("REJECTED", text)
        self.assertIn("Swipe left on More on Swiggy", text)
        self.assertIn("ask_user", text)

    def test_interactive_prompt_does_not_request_validate_completion(self) -> None:
        """
        The validate-with-sub_goal_completed escape belongs to the non-interactive surface only; including it interactively risks false-positive completion claims.
        """

        text = RepeatedFailureRejectionPromptBuilder.build(
            action_descriptor="Tap Continue",
            interactive=True,
        )

        self.assertNotIn("sub_goal_completed", text)


class RepeatedFailureRejectionPromptBuilderNonInteractiveTest(unittest.TestCase):
    """
    Pins the healing-context rejection sentence shape; ask_user must not be offered because the runtime cannot deliver it.
    """

    def test_non_interactive_prompt_does_not_offer_ask_user(self) -> None:
        """
        Fathom-healing runs without HITL; advertising ask_user would teach the LLM to emit an unsupported action.
        """

        text = RepeatedFailureRejectionPromptBuilder.build(
            action_descriptor="Swipe left on More on Swiggy",
            interactive=False,
        )

        self.assertIn("REJECTED", text)
        self.assertIn("Swipe left on More on Swiggy", text)
        self.assertNotIn("ask_user", text)

    def test_non_interactive_prompt_offers_navigate_back_escape(self) -> None:
        """
        Healing context needs a real escape: try a different action on this screen or navigate back; suggesting validate-with-sub_goal_completed risks false-positive completion claims so it must not appear.
        """

        text = RepeatedFailureRejectionPromptBuilder.build(
            action_descriptor="Tap Continue",
            interactive=False,
        )

        self.assertIn("navigate back", text)
        self.assertNotIn("sub_goal_completed", text)


class RepeatedFailureRejectionPromptBuilderNormalizationTest(unittest.TestCase):
    """
    Pins descriptor normalization so a blank or whitespace input never produces an unparseable prompt.
    """

    def test_blank_descriptor_normalizes_to_unknown_placeholder(self) -> None:
        """
        A blank or whitespace descriptor must still produce a stable, parseable rejection sentence.
        """

        text = RepeatedFailureRejectionPromptBuilder.build(
            action_descriptor="   ",
            interactive=False,
        )

        self.assertIn("REJECTED", text)
        self.assertIn("(unknown)", text)


if __name__ == "__main__":
    unittest.main()
