from __future__ import annotations

import asyncio
from logging import getLogger
from typing import Dict, Optional, Tuple

from fathom.core.exceptions import EmbeddingError
from fathom.interfaces.embedding import EmbeddingPort
from fathom.schemas.embedding import EmbeddingVector

logger = getLogger(__name__)


class EmbeddingCache:
    """
    Async pre-warmed text-to-vector cache backed by an :class:`EmbeddingPort`.
    Generic over any text payload; awaits the in-flight warmup once and lazily embeds on miss.
    Returns ``None`` only when every recovery path has been exhausted.
    """

    def __init__(self, *, embedder: EmbeddingPort) -> None:
        """
        Bind the cache to its embedding port; entries populate lazily.
        """

        self.__embedder = embedder
        self.__cache: Dict[str, EmbeddingVector] = {}
        self.__warmup_task: Optional[asyncio.Task[None]] = None

    def warm(self, *, texts: Tuple[str, ...]) -> None:
        """
        Kick off a background batched embed call covering every supplied text.
        """

        unique = self.__deduplicate(texts=texts)
        if not unique:
            return

        self.__warmup_task = asyncio.create_task(
            self.__warmup(texts=unique),
            name="embedding_cache_warmup",
        )

    async def get(self, *, text: str) -> Optional[EmbeddingVector]:
        """
        Return the cached vector for ``text``; await warmup once, then lazy-embed on miss.
        """

        key = self.__key(text=text)
        if not key:
            return None
        if key in self.__cache:
            return self.__cache[key]

        if self.__warmup_task is not None and not self.__warmup_task.done():
            try:
                await self.__warmup_task
            except Exception as exception:  # noqa: BLE001 - logged below
                logger.warning(
                    "Embedding cache warmup raised; falling back to lazy embed",
                    extra={
                        "component": "core.embedding.cache",
                        "event": "embedding.cache.warmup.failed",
                        "error.kind": type(exception).__name__,
                    },
                )

        if key in self.__cache:
            return self.__cache[key]

        return await self.__lazy(text=text)

    @staticmethod
    def __key(*, text: str) -> str:
        """
        Stable cache key derived from the input text.
        """

        return (text or "").strip().lower()

    @classmethod
    def __deduplicate(cls, *, texts: Tuple[str, ...]) -> Tuple[str, ...]:
        """
        Return non-empty input texts deduplicated by normalised key.
        """

        seen: Dict[str, str] = {}
        for text in texts:
            key = cls.__key(text=text)
            if not key or key in seen:
                continue
            seen[key] = text
        return tuple(seen.values())

    async def __warmup(self, *, texts: Tuple[str, ...]) -> None:
        """
        Embed every text in one batched call and populate the cache.
        """

        try:
            result = await self.__embedder.embed(texts=texts)
        except EmbeddingError as exception:
            logger.warning(
                "Embedding cache warmup failed",
                extra={
                    "component": "core.embedding.cache",
                    "event": "embedding.cache.warmup.error",
                    "error.kind": type(exception).__name__,
                    "text.count": len(texts),
                },
            )
            return

        if len(result.vectors) != len(texts):
            logger.warning(
                "Embedding cache warmup vector count mismatch",
                extra={
                    "component": "core.embedding.cache",
                    "event": "embedding.cache.warmup.mismatch",
                    "expected.count": len(texts),
                    "actual.count": len(result.vectors),
                },
            )
            return

        for text, vector in zip(texts, result.vectors, strict=True):
            self.__cache[self.__key(text=text)] = vector

    async def __lazy(self, *, text: str) -> Optional[EmbeddingVector]:
        """
        On cache miss embed one text and store the result; return None on failure.
        """

        sanitized = (text or "").strip()
        if not sanitized:
            return None

        try:
            result = await self.__embedder.embed(texts=(sanitized,))
        except EmbeddingError as exception:
            logger.warning(
                "Lazy embedding failed",
                extra={
                    "component": "core.embedding.cache",
                    "event": "embedding.cache.lazy.error",
                    "error.kind": type(exception).__name__,
                },
            )
            return None

        if not result.vectors:
            return None

        vector = result.vectors[0]
        self.__cache[self.__key(text=sanitized)] = vector
        return vector
