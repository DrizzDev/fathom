from __future__ import annotations

import asyncio
import unittest
from typing import List, Tuple

from fathom.constants.embedding import EmbeddingProvider
from fathom.core.embedding.cache import EmbeddingCache
from fathom.core.exceptions import EmbeddingError
from fathom.interfaces.embedding import EmbeddingPort
from fathom.schemas.embedding import EmbeddingResult, EmbeddingVector


class _StubEmbedder(EmbeddingPort):
    """
    Test double that records every batch and yields seeded vectors or a one-shot failure.
    """

    def __init__(
        self,
        *,
        responses: List[Tuple[EmbeddingVector, ...]],
        raise_first: bool = False,
        delay_seconds: float = 0.0,
    ) -> None:
        """
        Pre-seed responses, optional one-shot failure, and an artificial delay.
        """

        self.__responses = list(responses)
        self.__raise_first = raise_first
        self.__delay = delay_seconds
        self.calls: List[Tuple[str, ...]] = []

    async def embed(self, *, texts: Tuple[str, ...]) -> EmbeddingResult:
        """
        Record the call and yield the next staged vector tuple.
        """

        self.calls.append(texts)
        if self.__raise_first:
            self.__raise_first = False
            raise EmbeddingError("stubbed failure")
        if self.__delay:
            await asyncio.sleep(self.__delay)
        if not self.__responses:
            raise AssertionError("StubEmbedder ran out of staged responses")
        vectors = self.__responses.pop(0)
        return EmbeddingResult(
            duration=0,
            model="stub",
            provider=EmbeddingProvider.GEMINI,
            vectors=vectors,
        )


class EmbeddingCacheTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers warm-then-get, in-flight await, warmup failure fallback, lazy
    failure, empty-set short-circuit, and key deduplication for
    :class:`EmbeddingCache`.
    """

    async def test_warm_then_get_returns_cached_vector(self) -> None:
        """
        After warmup completes ``get`` hits the cache without a second provider call.
        """

        vector = EmbeddingVector(values=(0.1, 0.2))
        embedder = _StubEmbedder(responses=[(vector,)])
        cache = EmbeddingCache(embedder=embedder)

        cache.warm(texts=("Open the Posh app",))
        result = await cache.get(text="Open the Posh app")

        self.assertEqual(result, vector)
        self.assertEqual(len(embedder.calls), 1)

    async def test_get_awaits_inflight_warmup(self) -> None:
        """
        ``get`` issued before warmup completes blocks on the same task.
        """

        vector = EmbeddingVector(values=(0.5, 0.5))
        embedder = _StubEmbedder(responses=[(vector,)], delay_seconds=0.05)
        cache = EmbeddingCache(embedder=embedder)

        cache.warm(texts=("Login",))
        result = await cache.get(text="Login")

        self.assertEqual(result, vector)
        self.assertEqual(len(embedder.calls), 1)

    async def test_warmup_failure_falls_back_to_lazy_embed(self) -> None:
        """
        Warmup ``EmbeddingError`` is logged and the lookup falls through to lazy embed.
        """

        vector = EmbeddingVector(values=(0.3,))
        embedder = _StubEmbedder(responses=[(vector,)], raise_first=True)
        cache = EmbeddingCache(embedder=embedder)

        cache.warm(texts=("Pay the bill",))
        result = await cache.get(text="Pay the bill")

        self.assertEqual(result, vector)
        self.assertGreaterEqual(len(embedder.calls), 2)

    async def test_lazy_failure_returns_none(self) -> None:
        """
        When every recovery path fails the cache returns ``None`` for graceful degradation.
        """

        class _AlwaysFailing(EmbeddingPort):
            async def embed(self, *, texts: Tuple[str, ...]) -> EmbeddingResult:
                _ = texts
                raise EmbeddingError("always down")

        cache = EmbeddingCache(embedder=_AlwaysFailing())

        result = await cache.get(text="Search for biryani")

        self.assertIsNone(result)

    async def test_warm_skips_empty_texts(self) -> None:
        """
        An empty input tuple short-circuits and never consults the provider.
        """

        embedder = _StubEmbedder(responses=[])
        cache = EmbeddingCache(embedder=embedder)

        cache.warm(texts=())

        self.assertEqual(embedder.calls, [])

    async def test_duplicate_texts_deduplicated_in_batch(self) -> None:
        """
        Texts with the same normalised key are embedded once and served twice.
        """

        vector = EmbeddingVector(values=(0.9, 0.1))
        embedder = _StubEmbedder(responses=[(vector,)])
        cache = EmbeddingCache(embedder=embedder)

        cache.warm(texts=("Open the app", "open the app"))
        first = await cache.get(text="Open the app")
        second = await cache.get(text="open the app")

        self.assertEqual(first, vector)
        self.assertEqual(second, vector)
        self.assertEqual(len(embedder.calls), 1)


if __name__ == "__main__":
    unittest.main()
