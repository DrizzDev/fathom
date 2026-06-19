from __future__ import annotations

from typing import Tuple

from fathom.constants.embedding import EmbeddingProvider
from fathom.interfaces.embedding import EmbeddingPort
from fathom.schemas.embedding import EmbeddingResult, EmbeddingVector


class NoopEmbeddingAdapter(EmbeddingPort):
    """
    :class:`EmbeddingPort` that returns deterministic zero vectors.
    Used when embedding-backed similarity is disabled in configuration.
    """

    def __init__(self, *, dimensions: int = 8) -> None:
        """
        Bind the noop adapter to a fixed vector dimensionality.
        """

        if dimensions <= 0:
            raise ValueError("Noop embedding dimensionality must be positive")

        self.__dimensions = dimensions

    async def embed(self, *, texts: Tuple[str, ...]) -> EmbeddingResult:
        """
        Return one zero vector per input text without any provider call.
        """

        if not texts:
            raise ValueError("No texts supplied to noop embedding adapter")

        zero = EmbeddingVector(values=tuple(0.0 for _ in range(self.__dimensions)))

        return EmbeddingResult(
            duration=0,
            model="noop",
            provider=EmbeddingProvider.NOOP,
            vectors=tuple(zero for _ in texts),
        )
