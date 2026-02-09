from __future__ import annotations

from typing import Dict, Optional, Type

from fathom.prompts.base import PromptBuilder
from fathom.prompts.gemini import GeminiPromptBuilder


class PromptFactory:
    """
    Factory for retrieving the appropriate PromptBuilder based on model type.
    """

    __builders: Dict[str, Type[PromptBuilder]] = {
        "gemini": GeminiPromptBuilder,
    }

    @classmethod
    def get_builder(cls, model_name: str) -> PromptBuilder:
        """
        Returns a concrete builder instance.
        """
        # Default to Gemini if not specified
        kind = "gemini"

        if "gpt" in model_name.lower():
            kind = "openai" # Placeholder

        elif "claude" in model_name.lower():
            kind = "anthropic" # Placeholder

        builder_class = cls.__builders.get(kind, GeminiPromptBuilder)
        return builder_class()
