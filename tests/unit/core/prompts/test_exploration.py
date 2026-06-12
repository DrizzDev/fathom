from __future__ import annotations

import unittest

from fathom.core.prompts.exploration import ExplorationPromptBuilder


class TestExplorationPromptBuilder(unittest.TestCase):
    """The exploration system prompt carries the scan strategy, priority, and tool directive."""

    def setUp(self) -> None:
        self.__builder = ExplorationPromptBuilder()

    def test_system_prompt_includes_core_sections(self) -> None:
        prompt = self.__builder.build_system_prompt()

        for marker in (
            "screen mapper",
            "SCAN ORDER",
            "PRIORITY",
            "EXHAUSTION RULES",
            "explore_ui",
            "describe_screen",
            "NORMALIZED",
        ):
            self.assertIn(marker, prompt)

    def test_system_prompt_injects_goal(self) -> None:
        prompt = self.__builder.build_system_prompt(intent="Focus on checkout")

        self.assertIn("GOAL: Focus on checkout", prompt)

    def test_system_prompt_omits_goal_when_blank(self) -> None:
        self.assertNotIn("GOAL:", self.__builder.build_system_prompt())

    def test_prompt_has_no_emoji_symbols(self) -> None:
        prompt = self.__builder.build_system_prompt(intent="x")

        for symbol in ("⚠", "⋮", "⚙"):
            self.assertNotIn(symbol, prompt)

    def test_translation_prompt_requests_functional_fields(self) -> None:
        prompt = self.__builder.build_translation_prompt()

        for marker in (
            "describe_screen",
            "activity_name",
            "screen_purpose",
            "elements",
            "achievable_actions",
            "STABLE labels",
        ):
            self.assertIn(marker, prompt)
