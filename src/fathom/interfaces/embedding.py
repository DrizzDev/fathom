from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from fathom.schemas.embedding import EmbeddingResult


class EmbeddingPort(ABC):
    """
    Domain-facing port for producing dense embedding vectors over text.
    Implementations adapt third-party providers and must honor their
    own retry / timeout policy so callers see a uniform contract.
    """

    @abstractmethod
    async def embed(self, *, texts: Tuple[str, ...]) -> EmbeddingResult:
        """
        Embed every input text and return a tuple of vectors in request order.
        Implementations must raise :class:`EmbeddingError` on terminal
        failure (timeout exhaustion, validation error, provider error).
        """
