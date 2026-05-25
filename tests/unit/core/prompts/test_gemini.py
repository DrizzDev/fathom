from __future__ import annotations

from fathom.core.prompts.gemini import GeminiPromptBuilder


class TestGeminiPromptBuilder:
    """
    Covers Gemini system prompt assembly.
    """

    def test_system_prompt_surfaces_progress_safety_block(self) -> None:
        """
        The assembled prompt must include the progress-safety rule.
        """

        system_prompt = GeminiPromptBuilder().build()

        assert "request_replan" not in system_prompt
        assert "PROGRESS SAFETY (MANDATORY)" in system_prompt

    def test_system_prompt_does_not_require_label_id_when_manifest_lacks_target(
        self,
    ) -> None:
        """
        The assembled prompt must allow bbox fallback when manifest labels miss.
        """

        system_prompt = GeminiPromptBuilder().build()

        assert "MUST include 'label_id' from manifest for every interaction" not in system_prompt
        assert "Otherwise ground the action visually via bbox" in system_prompt
