"""Detect whether an LLM adapter supports embeddings."""

from __future__ import annotations

from fathom.interfaces.embedding import EmbeddingCapable
from fathom.interfaces.llm import LLMPort


class EmbeddingCapability:
    """
    Single rule for whether an LLM adapter can back a sibling embedding adapter.
    """

    @staticmethod
    def available(*, llm: LLMPort) -> bool:
        """
        Whether the adapter declares an embedding-capable client and has initialized one.
        """

        return isinstance(llm, EmbeddingCapable) and llm.client is not None
