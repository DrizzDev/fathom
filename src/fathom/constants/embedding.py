from __future__ import annotations

from enum import StrEnum
from typing import Final


class EmbeddingProvider(StrEnum):
    """
    Identifies the backend that produced an embedding vector.
    """

    NOOP = "noop"
    GEMINI = "gemini"


class EmbeddingTaskName(StrEnum):
    """
    Stable asyncio.Task ``name`` identifiers for embedding subsystems.
    """

    CACHE_WARMUP = "embedding.cache.warmup"


DEFAULT_EMBEDDING_ATTEMPTS: Final[int] = 3
DEFAULT_EMBEDDING_TIMEOUT: Final[int] = 30_000
DEFAULT_EMBEDDING_RETRY_BACKOFF: Final[float] = 1.5
DEFAULT_EMBEDDING_MODEL: Final[str] = "text-embedding-004"
