"""Gemini-provider prompt adapters.

These classes are thin shims over the provider-neutral policy that lives
in ``fathom.core.prompts.*``. Adding a new LLM provider means creating a
sibling subpackage (e.g. ``adapters/prompts/anthropic/``) with the same
three modules — never duplicating policy.
"""

from fathom.adapters.prompts.gemini.builder import GeminiPromptBuilder
from fathom.adapters.prompts.gemini.decomposition import GeminiDecompositionPromptBuilder
from fathom.adapters.prompts.gemini.export import GeminiExportPromptBuilder

__all__ = [
    "GeminiPromptBuilder",
    "GeminiDecompositionPromptBuilder",
    "GeminiExportPromptBuilder",
]
