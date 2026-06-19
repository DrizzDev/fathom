from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any, List, Tuple
from unittest.mock import MagicMock

from fathom.adapters.embedding.gemini import GeminiEmbeddingAdapter
from fathom.constants.embedding import EmbeddingProvider
from fathom.core.exceptions import EmbeddingError
from fathom.schemas.embedding import EmbeddingConfiguration, EmbeddingRetryPolicy


class _FakeClient:
    """
    Stand-in for the Gemini client capturing every embed_content call.
    """

    def __init__(self, *, responses: List[Any]) -> None:
        """
        Pre-seed the queue of responses or exceptions to yield per call.
        """

        self.__responses = list(responses)
        self.calls: List[Tuple[str, Tuple[str, ...]]] = []
        self.models = SimpleNamespace(embed_content=self.__embed)

    def __embed(self, *, model: str, contents: List[str]) -> Any:
        """
        Pop the next staged response or raise the staged exception.
        """

        self.calls.append((model, tuple(contents)))
        if not self.__responses:
            raise AssertionError("FakeClient ran out of staged responses")
        nxt = self.__responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _DelayedClient:
    """
    Stand-in that sleeps before returning so the adapter timeout fires.
    """

    def __init__(self, *, delay_seconds: float) -> None:
        """
        Bind the client to a delay longer than the configured timeout.
        """

        self.__delay = delay_seconds
        self.models = SimpleNamespace(embed_content=self.__embed)

    def __embed(self, *, model: str, contents: List[str]) -> Any:
        """
        Sleep then return one zero vector per input text.
        """

        _ = model, contents
        import time

        time.sleep(self.__delay)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])


class GeminiEmbeddingAdapterTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the Gemini embedding adapter against happy paths, timeouts,
    retries, and provider/payload errors.
    """

    @staticmethod
    def __response(*, vectors: Tuple[Tuple[float, ...], ...]) -> Any:
        """
        Build a stand-in Gemini embedding response object.
        """

        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=list(values)) for values in vectors]
        )

    async def test_happy_path_returns_one_vector_per_input(self) -> None:
        """
        A successful embed call projects every provider vector into typed form.
        """

        client = _FakeClient(responses=[self.__response(vectors=((0.1, 0.2), (0.3, 0.4)))])
        adapter = GeminiEmbeddingAdapter(client=client)

        result = await adapter.embed(texts=("hello", "world"))

        self.assertEqual(len(result.vectors), 2)
        self.assertEqual(result.vectors[0].values, (0.1, 0.2))
        self.assertEqual(result.vectors[1].values, (0.3, 0.4))
        self.assertEqual(result.provider, EmbeddingProvider.GEMINI)

    async def test_empty_text_input_raises_embedding_error(self) -> None:
        """
        Whitespace-only inputs cannot be embedded and must raise early.
        """

        client = _FakeClient(responses=[])
        adapter = GeminiEmbeddingAdapter(client=client)

        with self.assertRaises(EmbeddingError):
            await adapter.embed(texts=("   ",))

    async def test_no_texts_raises_embedding_error(self) -> None:
        """
        An empty tuple is rejected without consulting the provider.
        """

        adapter = GeminiEmbeddingAdapter(client=MagicMock())
        with self.assertRaises(EmbeddingError):
            await adapter.embed(texts=())

    async def test_provider_payload_missing_embeddings_raises(self) -> None:
        """
        A response without an ``embeddings`` field is a terminal failure.
        """

        client = _FakeClient(responses=[SimpleNamespace(embeddings=None)])
        adapter = GeminiEmbeddingAdapter(client=client)

        with self.assertRaises(EmbeddingError):
            await adapter.embed(texts=("hello",))

    async def test_provider_entry_missing_values_raises(self) -> None:
        """
        A response entry without ``values`` is a terminal failure.
        """

        client = _FakeClient(responses=[SimpleNamespace(embeddings=[SimpleNamespace(values=None)])])
        adapter = GeminiEmbeddingAdapter(client=client)

        with self.assertRaises(EmbeddingError):
            await adapter.embed(texts=("hello",))

    async def test_timeout_retries_then_succeeds(self) -> None:
        """
        Transient timeouts retry per the configured policy and succeed once a
        valid response is staged.
        """

        client = _FakeClient(
            responses=[
                asyncio.TimeoutError("first"),
                self.__response(vectors=((0.5, 0.5),)),
            ]
        )
        adapter = GeminiEmbeddingAdapter(
            client=client,
            configuration=EmbeddingConfiguration(
                timeout=10,
                retry=EmbeddingRetryPolicy(attempts=2, backoff=1.0),
            ),
        )

        result = await adapter.embed(texts=("hello",))

        self.assertEqual(len(result.vectors), 1)
        self.assertEqual(client.calls[0][1], ("hello",))

    async def test_timeout_exhausted_raises_embedding_error(self) -> None:
        """
        The adapter raises an :class:`EmbeddingError` after exhausting attempts.
        """

        client = _FakeClient(responses=[asyncio.TimeoutError("a"), asyncio.TimeoutError("b")])
        adapter = GeminiEmbeddingAdapter(
            client=client,
            configuration=EmbeddingConfiguration(
                timeout=10,
                retry=EmbeddingRetryPolicy(attempts=2, backoff=1.0),
            ),
        )

        with self.assertRaises(EmbeddingError):
            await adapter.embed(texts=("hello",))

    async def test_unexpected_exception_wrapped_as_embedding_error(self) -> None:
        """
        Non-timeout provider exceptions are wrapped as ``EmbeddingError``.
        """

        client = _FakeClient(responses=[RuntimeError("provider 500")])
        adapter = GeminiEmbeddingAdapter(client=client)

        with self.assertRaises(EmbeddingError):
            await adapter.embed(texts=("hello",))


if __name__ == "__main__":
    unittest.main()
