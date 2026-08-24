from __future__ import annotations

from typing import Dict, Optional, Type

from fathom.constants.llm import ModelProvider
from fathom.core.exceptions import ConfigurationError
from fathom.core.prompts.base import PromptBuilder
from fathom.core.prompts.decomposition import (
    DecompositionPromptBuilder,
    GeminiDecompositionPromptBuilder,
)
from fathom.core.prompts.export import ExportPromptBuilder, GeminiExportPromptBuilder
from fathom.core.prompts.gemini import GeminiPromptBuilder
from fathom.core.prompts.openai import (
    OpenAIDecompositionPromptBuilder,
    OpenAIExportPromptBuilder,
    OpenAIPromptBuilder,
)


class PromptFactory:
    """
    Factory for retrieving the appropriate PromptBuilder based on model type.
    """

    __builders: Dict[str, Type[PromptBuilder]] = {
        "gemini": GeminiPromptBuilder,
        "openai": OpenAIPromptBuilder,
    }
    __export_builders: Dict[str, Type[ExportPromptBuilder]] = {
        "gemini": GeminiExportPromptBuilder,
        "openai": OpenAIExportPromptBuilder,
    }
    __decomposition_builders: Dict[str, Type[DecompositionPromptBuilder]] = {
        "gemini": GeminiDecompositionPromptBuilder,
        "openai": OpenAIDecompositionPromptBuilder,
    }

    @classmethod
    def get_builder(
        cls, model_name: str, *, provider: Optional[ModelProvider] = None
    ) -> PromptBuilder:
        """
        Resolve the system-prompt builder from the declared provider.

        ``model_name`` remains for prompt-tier selection and legacy callers. It is never used to
        silently select a different provider when one has been declared.
        """

        kind = cls.__provider_kind(provider=provider, model_name=model_name)
        builder_class = cls.__builders.get(kind)
        if builder_class is None:
            raise ConfigurationError(f"Prompt builder for provider {kind} is not implemented")
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
    def get_export_builder(
        cls, model_name: str, *, provider: Optional[ModelProvider] = None
    ) -> ExportPromptBuilder:
        """
        Resolve the export-prompt builder from the declared provider.
        """

        kind = cls.__provider_kind(provider=provider, model_name=model_name)
        builder_class = cls.__export_builders.get(kind)
        if builder_class is None:
            raise ConfigurationError(
                f"Export prompt builder for provider {kind} is not implemented"
            )
        return builder_class()

    @classmethod
    def get_decomposition_builder(
        cls, model_name: str, *, provider: Optional[ModelProvider] = None
    ) -> DecompositionPromptBuilder:
        """
        Resolve the decomposition prompt builder from the declared provider.
        """

        kind = cls.__provider_kind(provider=provider, model_name=model_name)
        builder_class = cls.__decomposition_builders.get(kind)
        if builder_class is None:
            raise ConfigurationError(
                f"Decomposition prompt builder for provider {kind} is not implemented"
            )
        return builder_class()

    @staticmethod
    def __provider_kind(*, provider: Optional[ModelProvider], model_name: str) -> str:
        """
        Resolve a provider without pretending that a model name is an adapter declaration.

        Legacy callers with no declared provider retain Gemini behavior unless the model name
        explicitly identifies an unsupported provider; those calls fail closed instead of using a
        Gemini prompt for an OpenAI or Anthropic model.
        """

        if provider is not None and provider is not ModelProvider.UNKNOWN:
            return "gemini" if provider is ModelProvider.VERTEX_AI else provider.value

        lowered = model_name.lower()
        if "gpt" in lowered or "claude" in lowered:
            raise ConfigurationError(
                f"Provider must be declared for non-Gemini model {model_name!r}"
            )

        return "gemini"
