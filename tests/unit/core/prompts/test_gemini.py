from __future__ import annotations

import unittest

from fathom.constants.tools import BASE_TOOLS, ToolName
from fathom.core.prompts.gemini import GeminiPromptBuilder
from fathom.schemas.tools import AllowedTools


class GeminiPromptBuilderTest(unittest.TestCase):
    """Covers Gemini system prompt assembly."""

    @staticmethod
    def __tools() -> AllowedTools:
        """Build an HITL-capable allowed tool set for the prompt builder."""

        return AllowedTools(names=BASE_TOOLS | {ToolName.ASK_USER})

    def test_system_prompt_surfaces_progress_safety_block(self) -> None:
        """The assembled prompt must include the progress-safety rule."""

        system_prompt = GeminiPromptBuilder().build(tools=self.__tools())

        self.assertNotIn("request_replan", system_prompt)
        self.assertIn("PROGRESS SAFETY (MANDATORY)", system_prompt)

    def test_system_prompt_allows_bbox_fallback_when_manifest_lacks_target(self) -> None:
        """The assembled prompt must allow bbox fallback when manifest labels miss."""

        system_prompt = GeminiPromptBuilder().build(tools=self.__tools())

        self.assertNotIn(
            "MUST include 'label_id' from manifest for every interaction",
            system_prompt,
        )
        self.assertIn("Otherwise ground the action visually via bbox", system_prompt)

    def test_verifier_feedback_requires_corrective_action_before_completion(self) -> None:
        """Verifier feedback must require action before another completion claim."""

        user_context = GeminiPromptBuilder().build_user_context(
            history={
                "verifier_feedback": [
                    "Tap Yes, continue first",
                    {"reason": "modal still visible"},
                ]
            },
            sub_goal_info={"description": "Confirm SalarySe office address"},
        )

        self.assertIn("Take the next concrete UI action", user_context)
        self.assertIn("Do not claim completion again until that action has executed", user_context)
        self.assertIn("Continue working on the active sub-goal", user_context)
        self.assertIn("{'reason': 'modal still visible'}", user_context)
        self.assertNotIn("when the screen still needs a change", user_context)

    def test_durable_step_prompt_demands_visible_outcome_before_completion(self) -> None:
        """
        A durable sub-goal instructs the model not to claim completion until the result is visible.
        """

        durable = GeminiPromptBuilder().build_user_context(
            sub_goal_info={
                "index": 1,
                "total": 3,
                "description": "Add Diet Coke to the cart",
                "durable": True,
            },
        )
        transient = GeminiPromptBuilder().build_user_context(
            sub_goal_info={
                "index": 1,
                "total": 3,
                "description": "Open the cart",
                "durable": False,
            },
        )

        self.assertIn("VISIBLE on screen", durable)
        self.assertNotIn("VISIBLE on screen", transient)
