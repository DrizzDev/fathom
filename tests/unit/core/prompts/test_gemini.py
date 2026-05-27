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
