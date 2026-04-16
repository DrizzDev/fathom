from __future__ import annotations

import asyncio
import unittest
from typing import List, Optional
from unittest.mock import AsyncMock, Mock

from fathom.adapters.llm.cache import DEFAULT_BUCKET, CacheService


class _FakeCachedContent:
    """
    Test double for the provider's ``cachedContents`` resource.

    ``unittest.mock.Mock`` reserves ``name`` as a constructor argument,
    so we use a plain class to guarantee ``.name`` returns the intended
    string instead of a nested Mock.
    """

    def __init__(self, *, name: str) -> None:
        self.name = name


def _build_fake_client(cached_content_names: Optional[List[str]] = None) -> Mock:
    """
    Build a Gemini-SDK-shaped client double with async cache endpoints.

    ``caches.create`` returns successive ``_FakeCachedContent`` objects
    from the provided name list, rotating back to the last name once
    exhausted so tests can issue more creates than they preallocated.
    """

    names = list(cached_content_names or ["cache-0", "cache-1", "cache-2", "cache-3", "cache-4"])
    created: List[_FakeCachedContent] = []
    index = {"value": 0}

    async def create_side_effect(**_: object) -> _FakeCachedContent:
        slot = min(index["value"], len(names) - 1)
        content = _FakeCachedContent(name=names[slot])
        created.append(content)
        index["value"] += 1
        return content

    client = Mock()
    client.aio = Mock()
    client.aio.caches = Mock()
    client.aio.caches.create = AsyncMock(side_effect=create_side_effect)
    client.aio.caches.delete = AsyncMock()
    client.__created_contents__ = created  # type: ignore[attr-defined]
    return client


class CacheServiceBucketIsolationTest(unittest.IsolatedAsyncioTestCase):
    """
    Verify that buckets evict independently.
    """

    async def test_filling_one_bucket_does_not_evict_another(self) -> None:
        client = _build_fake_client()
        service = CacheService(
            client=client,
            model_name="test-model",
            max_entries=2,
            bucket_max_entries={"vision_planner": 2},
        )

        # Saturate the vision_planner bucket (limit = 2).
        await service.get_cached_content(
            system_instruction="planner-prompt-a", bucket="vision_planner"
        )
        await service.get_cached_content(
            system_instruction="planner-prompt-b", bucket="vision_planner"
        )

        # Hitting the default bucket must not evict vision_planner.
        await service.get_cached_content(system_instruction="classifier-prompt")

        self.assertEqual(client.aio.caches.delete.await_count, 0)
        self.assertEqual(service.bucket_stats("vision_planner").evictions, 0)
        self.assertEqual(service.bucket_stats(DEFAULT_BUCKET).evictions, 0)
        self.assertEqual(service.bucket_stats("vision_planner").creates, 2)
        self.assertEqual(service.bucket_stats(DEFAULT_BUCKET).creates, 1)

    async def test_exceeding_bucket_limit_evicts_oldest_entry_in_same_bucket(self) -> None:
        client = _build_fake_client(cached_content_names=["cache-0", "cache-1", "cache-2"])
        service = CacheService(
            client=client,
            model_name="test-model",
            max_entries=2,
        )

        await service.get_cached_content(system_instruction="prompt-a")
        await service.get_cached_content(system_instruction="prompt-b")
        await service.get_cached_content(system_instruction="prompt-c")

        # Third insert should have evicted the oldest entry (cache-0).
        client.aio.caches.delete.assert_awaited_once_with(name="cache-0")
        self.assertEqual(service.bucket_stats(DEFAULT_BUCKET).evictions, 1)
        self.assertEqual(service.bucket_stats(DEFAULT_BUCKET).creates, 3)


class CacheServiceBackCompatTest(unittest.IsolatedAsyncioTestCase):
    """
    Verify callers that omit ``bucket`` keep working.
    """

    async def test_default_bucket_is_used_when_kwarg_omitted(self) -> None:
        client = _build_fake_client()
        service = CacheService(client=client, model_name="test-model", max_entries=2)

        returned = await service.get_cached_content(system_instruction="prompt")

        self.assertEqual(returned, "cache-0")
        self.assertEqual(service.bucket_stats(DEFAULT_BUCKET).creates, 1)
        self.assertEqual(service.bucket_stats(DEFAULT_BUCKET).misses, 1)
        self.assertEqual(service.bucket_stats(DEFAULT_BUCKET).hits, 0)

    async def test_cache_hit_returns_existing_name_without_remote_create(self) -> None:
        client = _build_fake_client()
        service = CacheService(client=client, model_name="test-model", max_entries=4)

        first = await service.get_cached_content(system_instruction="prompt")
        second = await service.get_cached_content(system_instruction="prompt")

        self.assertEqual(first, second)
        self.assertEqual(client.aio.caches.create.await_count, 1)
        self.assertEqual(service.bucket_stats(DEFAULT_BUCKET).hits, 1)
        self.assertEqual(service.bucket_stats(DEFAULT_BUCKET).misses, 1)
        self.assertEqual(service.bucket_stats(DEFAULT_BUCKET).creates, 1)


class CacheServiceUndersizedHandlingTest(unittest.IsolatedAsyncioTestCase):
    """
    Verify sub-threshold content is remembered globally across buckets.
    """

    async def test_undersized_hash_short_circuits_any_bucket(self) -> None:
        client = _build_fake_client()
        client.aio.caches.create = AsyncMock(
            side_effect=RuntimeError("cachedContent is below the minimum token count of 2048")
        )
        service = CacheService(client=client, model_name="test-model", max_entries=2)

        first = await service.get_cached_content(
            system_instruction="tiny-prompt", bucket="bucket_a"
        )
        second = await service.get_cached_content(
            system_instruction="tiny-prompt", bucket="bucket_b"
        )

        self.assertIsNone(first)
        self.assertIsNone(second)
        # Only one attempted remote call; the second is short-circuited.
        self.assertEqual(client.aio.caches.create.await_count, 1)


class CacheServiceGlobalOperationsTest(unittest.IsolatedAsyncioTestCase):
    """
    Verify delete_cache and invalidate_cache_name span every bucket.
    """

    async def test_delete_cache_clears_every_bucket(self) -> None:
        client = _build_fake_client(cached_content_names=["c0", "c1", "c2"])
        service = CacheService(client=client, model_name="test-model", max_entries=4)

        await service.get_cached_content(system_instruction="a", bucket="alpha")
        await service.get_cached_content(system_instruction="b", bucket="beta")
        await service.get_cached_content(system_instruction="c", bucket="alpha")

        await service.delete_cache()

        deleted_names = sorted(
            call.kwargs["name"] for call in client.aio.caches.delete.await_args_list
        )
        self.assertEqual(deleted_names, ["c0", "c1", "c2"])

        # Subsequent gets are misses; state was fully cleared.
        await service.get_cached_content(system_instruction="a", bucket="alpha")
        self.assertEqual(service.bucket_stats("alpha").misses, 3)  # 2 creates + 1 post-delete miss

    async def test_invalidate_cache_name_finds_entry_across_buckets(self) -> None:
        client = _build_fake_client(cached_content_names=["first", "target", "third"])
        service = CacheService(client=client, model_name="test-model", max_entries=4)

        await service.get_cached_content(system_instruction="a", bucket="alpha")
        await service.get_cached_content(system_instruction="b", bucket="beta")
        await service.get_cached_content(system_instruction="c", bucket="alpha")

        await service.invalidate_cache_name("target")

        client.aio.caches.delete.assert_awaited_once_with(name="target")

        # The invalidated entry should not be found on a repeat lookup.
        await service.get_cached_content(system_instruction="b", bucket="beta")
        self.assertEqual(service.bucket_stats("beta").creates, 2)
        self.assertEqual(service.bucket_stats("beta").hits, 0)


class CacheServiceStatsAggregationTest(unittest.IsolatedAsyncioTestCase):
    """
    Verify per-bucket stats isolation and back-compat aggregate view.
    """

    async def test_aggregate_stats_sum_every_bucket(self) -> None:
        client = _build_fake_client()
        service = CacheService(client=client, model_name="test-model", max_entries=4)

        # alpha: 1 miss + create, 1 hit
        await service.get_cached_content(system_instruction="prompt-a", bucket="alpha")
        await service.get_cached_content(system_instruction="prompt-a", bucket="alpha")
        # beta: 2 misses + creates
        await service.get_cached_content(system_instruction="prompt-b1", bucket="beta")
        await service.get_cached_content(system_instruction="prompt-b2", bucket="beta")

        alpha_stats = service.bucket_stats("alpha")
        beta_stats = service.bucket_stats("beta")
        aggregate = service.stats

        self.assertEqual((alpha_stats.hits, alpha_stats.misses, alpha_stats.creates), (1, 1, 1))
        self.assertEqual((beta_stats.hits, beta_stats.misses, beta_stats.creates), (0, 2, 2))
        self.assertEqual(aggregate.hits, 1)
        self.assertEqual(aggregate.misses, 3)
        self.assertEqual(aggregate.creates, 3)
        self.assertEqual(sorted(service.all_bucket_stats().keys()), ["alpha", "beta"])


class CacheServiceConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    """
    Verify the per-bucket lock serializes concurrent writers.
    """

    async def test_concurrent_same_key_creates_remote_content_only_once(self) -> None:
        create_in_flight = asyncio.Event()
        allow_create = asyncio.Event()
        create_calls = {"count": 0}

        async def slow_create(**_: object) -> _FakeCachedContent:
            create_calls["count"] += 1
            create_in_flight.set()
            await allow_create.wait()
            return _FakeCachedContent(name="only-one")

        client = Mock()
        client.aio = Mock()
        client.aio.caches = Mock()
        client.aio.caches.create = AsyncMock(side_effect=slow_create)
        client.aio.caches.delete = AsyncMock()

        service = CacheService(client=client, model_name="test-model", max_entries=4)

        # Start both callers; the second should park on the bucket lock
        # until the first completes and publishes the entry.
        first_task = asyncio.create_task(
            service.get_cached_content(system_instruction="shared-prompt", bucket="alpha")
        )
        await create_in_flight.wait()
        second_task = asyncio.create_task(
            service.get_cached_content(system_instruction="shared-prompt", bucket="alpha")
        )
        # Give the second task a tick to attempt the lock.
        await asyncio.sleep(0)

        allow_create.set()
        first_result, second_result = await asyncio.gather(first_task, second_task)

        self.assertEqual(first_result, "only-one")
        self.assertEqual(second_result, "only-one")
        self.assertEqual(create_calls["count"], 1)
        self.assertEqual(service.bucket_stats("alpha").creates, 1)
        self.assertEqual(service.bucket_stats("alpha").hits, 1)
        self.assertEqual(service.bucket_stats("alpha").misses, 1)


class GeminiLLMCacheBucketPlumbingTest(unittest.IsolatedAsyncioTestCase):
    """
    Verify GeminiLLM.generate forwards its cache_bucket kwarg through
    to CacheService.get_cached_content.
    """

    async def test_generate_forwards_cache_bucket(self) -> None:
        from fathom.adapters.llm.gemini import GeminiLLM

        # Build a GeminiLLM instance without going through __init__
        # (which requires credentials + SDK client). We stitch in the
        # minimum collaborators we need via name-mangled access.
        llm = GeminiLLM.__new__(GeminiLLM)
        llm._GeminiLLM__client = Mock()  # non-None passes the readiness check
        llm._GeminiLLM__configuration = Mock()
        llm._GeminiLLM__configuration.model = "test-model"
        llm._GeminiLLM__configuration.max_retries = 0
        llm._GeminiLLM__configuration.retry_delay = 0
        llm._GeminiLLM__configuration.temperature = 0.0
        llm._GeminiLLM__configuration.media_resolution = "low"
        llm._GeminiLLM__configuration.thinking_level = "low"
        llm._GeminiLLM__configuration.include_thoughts = False

        cache = Mock()
        cache.get_cached_content = AsyncMock(return_value="cache-name")
        llm._GeminiLLM__cache = cache

        # Stub the SDK generate_content call to return a minimal response.
        fake_response = Mock()
        fake_response.candidates = []
        fake_response.usage_metadata = None
        llm._GeminiLLM__client.aio = Mock()
        llm._GeminiLLM__client.aio.models = Mock()
        llm._GeminiLLM__client.aio.models.generate_content = AsyncMock(return_value=fake_response)

        await llm.generate(
            use_cache=True,
            prompt=["hello"],
            system_instruction="instr",
            tools={"function_declarations": [{"name": "noop"}]},
            cache_bucket="vision_planner",
        )

        cache.get_cached_content.assert_awaited_once()
        call_kwargs = cache.get_cached_content.await_args.kwargs
        self.assertEqual(call_kwargs["bucket"], "vision_planner")
        self.assertEqual(call_kwargs["system_instruction"], "instr")
        self.assertEqual(call_kwargs["tools"], [{"name": "noop"}])
