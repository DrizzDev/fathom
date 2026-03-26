from __future__ import annotations

import hashlib
import json
from logging import getLogger
from typing import Any, Dict, List, Optional

from google.genai.types import Content, Part

from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.schemas.statistics import CacheStats

logger = getLogger(__name__)


class CacheService:
    """
    Service for managing LLM context caching with key hashing and stats.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        *,
        ttl_minutes: int = 60,
        max_entries: int = 2,
    ) -> None:
        """
        Initialize CacheService.

        Args:
            client: The GenAI client instance.
            model_name: The model name to cache for.
            ttl_minutes: Time-to-live for cached content in minutes.
            max_entries: Maximum number of distinct cache entries retained.
        """

        self.__client = client
        self.__model_name = model_name
        self.__ttl_minutes = ttl_minutes
        self.__max_entries = max(1, max_entries)

        self.__cache_entries: Dict[str, Any] = {}
        # Hashes that are known to be below the provider's minimum token threshold.
        # Tracked to avoid redundant API calls for content that will never cache successfully.
        self.__undersized_hashes: set[str] = set()

        self.stats = CacheStats()

    async def get_cached_content(
        self, system_instruction: str, tools: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """
        Creates or retrieves a cached content object.
        Invalidates the cache if the content hash changes.

        Args:
            system_instruction: The system prompt/instruction to cache.
            tools: Optional list of tool declarations to include in cache.

        Returns:
            The name of the cached content object, or None if caching failed/skipped.
        """

        current_hash = self.__compute_hash(instruction=system_instruction, tools=tools)

        # Short-circuit: content is known to be below the provider minimum token count.
        if current_hash in self.__undersized_hashes:
            return None

        cached_entry = self.__cache_entries.get(current_hash)

        # Cache hit: matching entry exists
        if cached_entry:
            self.stats.hits += 1
            logger.debug(f"Cache hit (hash={current_hash[:8]})")
            return str(cached_entry.name)

        # Cache miss: hash not present yet
        self.stats.misses += 1

        try:
            if hasattr(self.__client, "aio") and hasattr(self.__client.aio, "caches"):
                tool_list = []
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

                # Bounded in-memory tracking to avoid unbounded remote cache accumulation.
                while len(self.__cache_entries) >= self.__max_entries:
                    oldest_hash = next(iter(self.__cache_entries))
                    await self.__evict_hash(content_hash=oldest_hash)
                    self.stats.evictions += 1

                self.stats.creates += 1
                self.__cache_entries[current_hash] = cached_content

                logger.info(
                    f"Created context cache: {cached_content.name} (hash={current_hash[:8]})"
                )
                return str(cached_content.name)

        except Exception as exception:
            if "minimum token count" in str(exception):
                self.__undersized_hashes.add(current_hash)
                logger.debug(f"Skipping cache (content below minimum token threshold): {exception}")
            else:
                logger.warning(f"Failed to create cache: {exception}")

            return None

        return None

    async def delete_cache(self) -> None:
        """
        Deletes the current cache if it exists.
        """

        await self.__evict_all()

    async def invalidate_cache_name(self, cache_name: str) -> None:
        """
        Evict a cache entry by remote cache name if tracked locally.
        """

        for content_hash, cached_content in list(self.__cache_entries.items()):
            if str(getattr(cached_content, "name", "")) == cache_name:
                await self.__evict_hash(content_hash=content_hash)
                break

    async def __evict_all(self) -> None:
        """
        Evicts all current cache entries.
        """

        for content_hash in list(self.__cache_entries.keys()):
            await self.__evict_hash(content_hash=content_hash)

    async def __evict_hash(self, *, content_hash: str) -> None:
        cached_content = self.__cache_entries.get(content_hash)
        if not cached_content:
            return

        try:
            await self.__client.aio.caches.delete(name=cached_content.name)
            logger.info(f"Deleted cache: {cached_content.name}")
        except Exception as exception:
            logger.warning(f"Failed to delete cache: {exception}")
        finally:
            self.__cache_entries.pop(content_hash, None)

    @staticmethod
    def __compute_hash(instruction: str, tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Computes a deterministic hash of the cache key content.

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
