from __future__ import annotations

from typing import Dict, Type

from fathom.core.prompts.base import PromptBuilder
from fathom.core.prompts.gemini import GeminiPromptBuilder


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

        kind = "gemini"

        if "gpt" in model_name.lower():
            kind = "openai"  # Placeholder

        elif "claude" in model_name.lower():
            kind = "anthropic"  # Placeholder

        builder_class = cls.__builders.get(kind, GeminiPromptBuilder)
        return builder_class()

    @staticmethod
    def resolve_version(model_name: str, use_xml: bool) -> str:
        """
        Determines the optimal prompt version based on model capabilities.
        """

        is_flash = "flash" in model_name.lower()

        tier = "flash" if is_flash else "pro"
        strategy = "xml" if use_xml else "vision"

        return f"{tier}_{strategy}"
