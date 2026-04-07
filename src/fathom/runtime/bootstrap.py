"""Explicit composition-root wiring for provider registries.

Core services look up builders and other provider-specific implementations
via registries (e.g. :class:`fathom.core.prompts.factory.PromptFactory`).
Those registries must be populated *before* any service that depends on
them is instantiated.

Historically, registration happened as a side effect of importing
``fathom.runtime.factories`` — which is implicit and fragile: anything that
imports that module (tests, tooling, editor autocomplete) would trigger
wiring. This module replaces that pattern with an explicit, idempotent
``register_default_prompt_builders()`` function that the composition root
(CLI, test fixtures, long-running services) must call during startup.

Typical usage::

    from fathom.runtime.bootstrap import register_default_prompt_builders

    register_default_prompt_builders()   # once, at process startup
    ...
    service = IntentDecomposer(llm=...)  # now resolves a builder

Calling the function more than once is safe — it overwrites the existing
registration with the same class, which is a no-op in practice.
"""

from __future__ import annotations

from logging import getLogger

from fathom.adapters.prompts import (
    GeminiDecompositionPromptBuilder,
    GeminiExportPromptBuilder,
    GeminiPromptBuilder,
)
from fathom.core.prompts.factory import PromptFactory

__all__ = ["register_default_prompt_builders"]

logger = getLogger(__name__)

_REGISTERED = False


def register_default_prompt_builders() -> None:
    """Register the default prompt builders for every supported provider.

    This is the explicit composition-root hook that replaces the previous
    import-time side effects in :mod:`fathom.runtime.factories`. Callers
    are expected to invoke it once during process startup; repeated calls
    are safe and cheap.
    """

    global _REGISTERED
    if _REGISTERED:
        return

    PromptFactory.register_builder("gemini", GeminiPromptBuilder)
    PromptFactory.register_export_builder("gemini", GeminiExportPromptBuilder)
    PromptFactory.register_decomposition_builder("gemini", GeminiDecompositionPromptBuilder)

    _REGISTERED = True
    logger.debug("PromptFactory default builders registered (gemini)")
