from __future__ import annotations

import unittest

from fathom.constants.llm import ModelProvider
from fathom.core.prompts.factory import PromptFactory


class PromptFactoryProviderTest(unittest.TestCase):
    """
    Prompt selection follows the explicitly declared provider.
    """

    def test_openai_uses_an_explicit_prompt_builder(self) -> None:
        """
        An OpenAI declaration selects the OpenAI prompt adapter explicitly.
        """

        builder = PromptFactory.get_builder(
            model_name="gpt-5",
            provider=ModelProvider.OPENAI,
        )

        self.assertEqual(type(builder).__name__, "OpenAIPromptBuilder")

    def test_vertex_ai_uses_gemini_prompt_dialect_explicitly(self) -> None:
        """
        Vertex AI selects the Gemini prompt dialect through an explicit mapping.
        """

        builder = PromptFactory.get_builder(
            model_name="gemini-3.5-flash",
            provider=ModelProvider.VERTEX_AI,
        )

        self.assertEqual(type(builder).__name__, "GeminiPromptBuilder")
