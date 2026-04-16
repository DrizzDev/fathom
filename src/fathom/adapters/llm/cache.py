from __future__ import annotations

import asyncio
import hashlib
import json
from logging import getLogger
from typing import Any, Dict, List, Mapping, Optional, Set

from google.genai.types import Content, Part

from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.schemas.statistics import CacheStats

logger = getLogger(__name__)

DEFAULT_BUCKET = "default"


class CacheService:
    """
    Service for managing LLM context caching with key hashing and stats.

    Storage is partitioned into named buckets so each caller subsystem
    (e.g. vision planner, intent classifier) owns an independent slot
    budget and cannot evict another caller's entries.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        *,
        ttl_minutes: int = 60,
        max_entries: int = 2,
        bucket_max_entries: Optional[Mapping[str, int]] = None,
    ) -> None:
        """
        Initialize CacheService.

        Args:
            client: The GenAI client instance.
            model_name: The model name to cache for.
            ttl_minutes: Time-to-live for cached content in minutes.
            max_entries: Default per-bucket slot budget for buckets not
                present in ``bucket_max_entries``.
            bucket_max_entries: Optional per-bucket slot overrides. Any
                bucket not listed here uses ``max_entries``.
        """

        self.__client = client
        self.__model_name = model_name
        self.__ttl_minutes = ttl_minutes
        self.__default_max_entries = max(1, max_entries)
        self.__bucket_max_entries: Dict[str, int] = {
            name: max(1, limit) for name, limit in (bucket_max_entries or {}).items()
        }

        # bucket -> (hash -> cached_content)
        self.__buckets: Dict[str, Dict[str, Any]] = {}
        # bucket -> CacheStats
        self.__bucket_stats: Dict[str, CacheStats] = {}
        # bucket -> lock (guards the miss-create-insert critical section)
        self.__bucket_locks: Dict[str, asyncio.Lock] = {}

        # Hashes known to be below the provider's minimum token threshold.
        # Tracked globally (content property, not bucket property) so a
        # retry from a different bucket short-circuits without another
        # provider round-trip.
        self.__undersized_hashes: Set[str] = set()

    @property
    def stats(self) -> CacheStats:
        """
        Aggregate cache statistics summed across every bucket.

        Preserved for back-compat with callers that read ``cache.stats``
        without knowing about buckets. Mutations on the returned
        snapshot are not persisted; use ``bucket_stats(name)`` to
        inspect or compose per-bucket numbers.
        """

        aggregate = CacheStats()
        for bucket_stats in self.__bucket_stats.values():
            aggregate.hits += bucket_stats.hits
            aggregate.misses += bucket_stats.misses
            aggregate.creates += bucket_stats.creates
            aggregate.evictions += bucket_stats.evictions
        return aggregate

    def bucket_stats(self, bucket: str) -> CacheStats:
        """
        Return the ``CacheStats`` for a specific bucket, creating a
        zeroed entry if the bucket has not been touched yet.
        """

        return self.__ensure_bucket_stats(bucket=bucket)

    def all_bucket_stats(self) -> Dict[str, CacheStats]:
        """
        Return a snapshot mapping bucket name → ``CacheStats`` for every
        bucket that has been observed so far.
        """

        return dict(self.__bucket_stats)

    async def get_cached_content(
        self,
        system_instruction: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        bucket: str = DEFAULT_BUCKET,
    ) -> Optional[str]:
        """
        Create or retrieve a cached content object within a bucket.

        Each bucket owns an independent slot budget so that a single
        caller cannot evict another caller's entries.

        Args:
            system_instruction: The system prompt/instruction to cache.
            tools: Optional list of tool declarations to include in cache.
            bucket: Caller-supplied namespace that isolates eviction.
                Defaults to ``"default"`` for legacy callers.

        Returns:
            The name of the cached content object, or ``None`` if caching
            failed or was skipped (content too small, provider error).
        """

        current_hash = self.__compute_hash(instruction=system_instruction, tools=tools)

        # Short-circuit: content is known to be below the provider minimum
        # token count and can never cache successfully, regardless of bucket.
        if current_hash in self.__undersized_hashes:
            return None

        lock = self.__ensure_bucket_lock(bucket=bucket)
        async with lock:
            bucket_entries = self.__ensure_bucket(bucket=bucket)
            bucket_stats = self.__ensure_bucket_stats(bucket=bucket)

            # Cache hit: matching entry exists within this bucket.
            cached_entry = bucket_entries.get(current_hash)
            if cached_entry:
                bucket_stats.hits += 1
                logger.debug(f"Cache hit (bucket={bucket}, hash={current_hash[:8]})")
                return str(cached_entry.name)

            # Cache miss.
            bucket_stats.misses += 1

            if not (hasattr(self.__client, "aio") and hasattr(self.__client.aio, "caches")):
                return None

            try:
                tool_list: List[Dict[str, Any]] = []
                if tools:
                    tool_list.append({"function_declarations": tools})

                config = {
                    "tools": tool_list,
                    "ttl": f"{self.__ttl_minutes * 60}s",
                    "tool_config": {"function_calling_config": {"mode": "ANY"}},
                    "contents": [
                        Content(role="user", parts=[Part.from_text(text=system_instruction)])
                    ],
                }

                cached_content = await self.__client.aio.caches.create(
                    model=self.__model_name, config=config
                )

            except Exception as exception:
                if "minimum token count" in str(exception):
                    self.__undersized_hashes.add(current_hash)
                    logger.debug(
                        f"Skipping cache (content below minimum token threshold): {exception}"
                    )
                else:
                    logger.warning(f"Failed to create cache: {exception}")

                return None

            # Bucket-local eviction: only this bucket's entries are at risk.
            max_entries = self.__bucket_limit(bucket=bucket)
            while len(bucket_entries) >= max_entries:
                oldest_hash = next(iter(bucket_entries))
                await self.__evict_hash(bucket=bucket, content_hash=oldest_hash)
                bucket_stats.evictions += 1

            bucket_stats.creates += 1
            bucket_entries[current_hash] = cached_content

            logger.info(
                f"Created context cache: {cached_content.name} "
                f"(bucket={bucket}, hash={current_hash[:8]})"
            )
            return str(cached_content.name)

    async def delete_cache(self) -> None:
        """
        Delete every cache entry across every bucket.
        """

        await self.__evict_all()

    async def invalidate_cache_name(self, cache_name: str) -> None:
        """
        Evict a cache entry by remote cache name across every bucket.
        """

        for bucket, entries in list(self.__buckets.items()):
            for content_hash, cached_content in list(entries.items()):
                if str(getattr(cached_content, "name", "")) == cache_name:
                    await self.__evict_hash(bucket=bucket, content_hash=content_hash)
                    return

    async def __evict_all(self) -> None:
        """
        Evict all current cache entries across every bucket.
        """

        for bucket, entries in list(self.__buckets.items()):
            for content_hash in list(entries.keys()):
                await self.__evict_hash(bucket=bucket, content_hash=content_hash)

    async def __evict_hash(self, *, bucket: str, content_hash: str) -> None:
        bucket_entries = self.__buckets.get(bucket)
        if not bucket_entries:
            return

        cached_content = bucket_entries.get(content_hash)
        if not cached_content:
            return

        try:
            await self.__client.aio.caches.delete(name=cached_content.name)
            logger.info(f"Deleted cache: {cached_content.name} (bucket={bucket})")
        except Exception as exception:
            logger.warning(f"Failed to delete cache: {exception}")
        finally:
            bucket_entries.pop(content_hash, None)

    def __ensure_bucket(self, *, bucket: str) -> Dict[str, Any]:
        entries = self.__buckets.get(bucket)
        if entries is None:
            entries = {}
            self.__buckets[bucket] = entries
        return entries

    def __ensure_bucket_stats(self, *, bucket: str) -> CacheStats:
        stats = self.__bucket_stats.get(bucket)
        if stats is None:
            stats = CacheStats()
            self.__bucket_stats[bucket] = stats
        return stats

    def __ensure_bucket_lock(self, *, bucket: str) -> asyncio.Lock:
        lock = self.__bucket_locks.get(bucket)
        if lock is None:
            lock = asyncio.Lock()
            self.__bucket_locks[bucket] = lock
        return lock

    def __bucket_limit(self, *, bucket: str) -> int:
        return self.__bucket_max_entries.get(bucket, self.__default_max_entries)

    @staticmethod
    def __compute_hash(instruction: str, tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Compute a deterministic hash of the cache key content.

        The hash is a content fingerprint: bucket name is intentionally
        excluded so identical (instruction, tools) in two buckets still
        produce two distinct cache entries. If two buckets want to share
        a single cached content they can route through the same bucket.

        Args:
            instruction: System instruction text.
            tools: Tool declarations list.

        Returns:
            Hex digest of the hash.
        """

        payload = instruction

        if tools:
            payload += json.dumps(tools, sort_keys=True)

        return hashlib.sha256(payload.encode(), usedforsecurity=False).hexdigest()[
            :VISUAL_HASH_LENGTH
        ]
