from __future__ import annotations

from typing import Dict, Type

from fathom.core.prompts.base import PromptBuilder
from fathom.core.prompts.decomposition import (
    DecompositionPromptBuilder,
    GeminiDecompositionPromptBuilder,
)
from fathom.core.prompts.export import ExportPromptBuilder, GeminiExportPromptBuilder
from fathom.core.prompts.gemini import GeminiPromptBuilder


class PromptFactory:
    """
    Factory for retrieving the appropriate PromptBuilder based on model type.
    """

    __builders: Dict[str, Type[PromptBuilder]] = {
        "gemini": GeminiPromptBuilder,
    }
    __export_builders: Dict[str, Type[ExportPromptBuilder]] = {
        "gemini": GeminiExportPromptBuilder,
    }
    __decomposition_builders: Dict[str, Type[DecompositionPromptBuilder]] = {
        "gemini": GeminiDecompositionPromptBuilder,
    }

    @classmethod
    def get_builder(cls, model_name: str) -> PromptBuilder:
        """
        Resolve the system-prompt builder for the model. Only Gemini is wired today;
        gpt and claude are placeholders that fall back to the Gemini builder.
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
        Pick the prompt variant "{tier}_{strategy}": flash or pro by model name, xml or vision by grounding.
        """

        is_flash = "flash" in model_name.lower()

        tier = "flash" if is_flash else "pro"
        strategy = "xml" if use_xml else "vision"

        return f"{tier}_{strategy}"

    @classmethod
    def get_export_builder(cls, model_name: str) -> ExportPromptBuilder:
        """
        Resolve the export-prompt builder for the model. Only Gemini is wired today;
        gpt and claude are placeholders that fall back to the Gemini builder.
        """

        kind = "gemini"

        if "gpt" in model_name.lower():
            kind = "openai"  # Placeholder

        elif "claude" in model_name.lower():
            kind = "anthropic"  # Placeholder

        builder_class = cls.__export_builders.get(kind, GeminiExportPromptBuilder)
        return builder_class()

    @classmethod
    def get_decomposition_builder(cls, model_name: str) -> DecompositionPromptBuilder:
        """
        Resolve the decomposition-prompt builder for the model. Only Gemini is wired today;
        gpt and claude are placeholders that fall back to the Gemini builder.
        """

        kind = "gemini"

        if "gpt" in model_name.lower():
            kind = "openai"  # Placeholder

        elif "claude" in model_name.lower():
            kind = "anthropic"  # Placeholder

        builder_class = cls.__decomposition_builders.get(kind, GeminiDecompositionPromptBuilder)
        return builder_class()
