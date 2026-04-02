from __future__ import annotations

from typing import Dict, Type

from fathom.core.prompts.base import PromptBuilder
from fathom.core.prompts.decomposition import DecompositionPromptBuilder
from fathom.core.prompts.export import ExportPromptBuilder


class PromptFactory:
    """
    Registry-based factory for retrieving PromptBuilders by model kind.

    Concrete builders must be registered at composition time (e.g. in
    ``runtime/factories.py``) via the ``register_*`` class methods.
    """

    __builders: Dict[str, Type[PromptBuilder]] = {}
    __export_builders: Dict[str, Type[ExportPromptBuilder]] = {}
    __decomposition_builders: Dict[str, Type[DecompositionPromptBuilder]] = {}

    # ------------------------------------------------------------------
    # Registration API (called from the composition root)
    # ------------------------------------------------------------------

    @classmethod
    def register_builder(cls, kind: str, builder_class: Type[PromptBuilder]) -> None:
        cls.__builders[kind] = builder_class

    @classmethod
    def register_export_builder(cls, kind: str, builder_class: Type[ExportPromptBuilder]) -> None:
        cls.__export_builders[kind] = builder_class

    @classmethod
    def register_decomposition_builder(
        cls, kind: str, builder_class: Type[DecompositionPromptBuilder]
    ) -> None:
        cls.__decomposition_builders[kind] = builder_class

    # ------------------------------------------------------------------
    # Lookup API
    # ------------------------------------------------------------------

    @classmethod
    def resolve_kind(cls, model_name: str) -> str:
        """
        Map a model name to a registry key.

        Override or extend at registration time if more providers are added.
        """

        lower = model_name.lower()

        if "gpt" in lower:
            return "openai"

        if "claude" in lower:
            return "anthropic"

        return "gemini"

    @classmethod
    def get_builder(cls, model_name: str) -> PromptBuilder:
        kind = cls.resolve_kind(model_name)
        builder_class = cls.__builders.get(kind)

        if builder_class is None:
            raise ValueError(
                f"No PromptBuilder registered for kind={kind!r} (model={model_name!r}). "
                "Register one via PromptFactory.register_builder()."
            )

        return builder_class()

    @classmethod
    def get_export_builder(cls, model_name: str) -> ExportPromptBuilder:
        kind = cls.resolve_kind(model_name)
        builder_class = cls.__export_builders.get(kind)

        if builder_class is None:
            raise ValueError(
                f"No ExportPromptBuilder registered for kind={kind!r} (model={model_name!r}). "
                "Register one via PromptFactory.register_export_builder()."
            )

        return builder_class()

    @classmethod
    def get_decomposition_builder(cls, model_name: str) -> DecompositionPromptBuilder:
        kind = cls.resolve_kind(model_name)
        builder_class = cls.__decomposition_builders.get(kind)

        if builder_class is None:
            raise ValueError(
                f"No DecompositionPromptBuilder registered for kind={kind!r} "
                f"(model={model_name!r}). "
                "Register one via PromptFactory.register_decomposition_builder()."
            )

        return builder_class()
