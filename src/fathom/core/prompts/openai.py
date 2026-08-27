from __future__ import annotations

from fathom.core.prompts.decomposition import GeminiDecompositionPromptBuilder
from fathom.core.prompts.export import GeminiExportPromptBuilder
from fathom.core.prompts.gemini import GeminiPromptBuilder


class OpenAIPromptBuilder(GeminiPromptBuilder):
    """
    OpenAI planner prompt builder using provider-neutral semantic instructions.
    """


class OpenAIDecompositionPromptBuilder(GeminiDecompositionPromptBuilder):
    """
    OpenAI decomposition prompt builder using shared semantic instructions.
    """


class OpenAIExportPromptBuilder(GeminiExportPromptBuilder):
    """
    OpenAI export prompt builder using shared semantic instructions.
    """
