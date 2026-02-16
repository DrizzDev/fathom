from __future__ import annotations

import hashlib
import json
from logging import getLogger
from typing import Any, Dict, List, Optional

from google.genai.types import Content

from fathom.schemas.statistics import CacheStats

logger = getLogger(__name__)


class CacheService:
    """
    Service for managing LLM context caching with key hashing and stats.
    """

    def __init__(self, client: Any, model_name: str, *, ttl_minutes: int = 60) -> None:
        """
        Initialize CacheService.

        Args:
            client: The GenAI client instance.
            model_name: The model name to cache for.
            ttl_minutes: Time-to-live for cached content in minutes.
        """
        self.__client = client
        self.__model_name = model_name
        self.__ttl_minutes = ttl_minutes

        self.__cached_content: Optional[Any] = None
        self.__content_hash: Optional[str] = None

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

        # Cache hit: same content, cache exists
        if self.__cached_content and self.__content_hash == current_hash:
            self.stats.hits += 1
            logger.debug(f"Cache hit (hash={current_hash[:8]})")
            return str(self.__cached_content.name)

        # Cache miss: content changed or no cache
        self.stats.misses += 1

        # Evict stale cache if content changed
        if self.__cached_content and self.__content_hash != current_hash:
            self.stats.evictions += 1
            logger.info(
                f"Cache invalidated (old={self.__content_hash[:8] if self.__content_hash else '?'}, new={current_hash[:8]})"
            )
            await self.__evict()

        try:
            if hasattr(self.__client, "aio") and hasattr(self.__client.aio, "caches"):
                tool_list = []
                if tools:
                    tool_list.append({"function_declarations": tools})

                config = {
                    "tools": tool_list,
                    "ttl": f"{self.__ttl_minutes * 60}s",
                    "tool_config": {"function_calling_config": {"mode": "ANY"}},
                    "contents": [Content(role="user", parts=[{"text": system_instruction}])],
                }

                self.__cached_content = await self.__client.aio.caches.create(
                    model=self.__model_name, config=config
                )
                self.__content_hash = current_hash
                self.stats.creates += 1

                logger.info(
                    f"Created context cache: {self.__cached_content.name} (hash={current_hash[:8]})"
                )
                return str(self.__cached_content.name)

        except Exception as exception:
            if "minimum token count" in str(exception):
                logger.debug(f"Skipping cache: {exception}")
            else:
                logger.warning(f"Failed to create cache: {exception}")

            return None

        return None

    async def delete_cache(self) -> None:
        """
        Deletes the current cache if it exists.
        """

        await self.__evict()

    async def __evict(self) -> None:
        """
        Evicts the current cache entry.
        """

        if self.__cached_content:
            try:
                await self.__client.aio.caches.delete(name=self.__cached_content.name)
                logger.info(f"Deleted cache: {self.__cached_content.name}")
            except Exception as exception:
                logger.warning(f"Failed to delete cache: {exception}")
            finally:
                self.__cached_content = None
                self.__content_hash = None

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

        return hashlib.sha256(payload.encode(), usedforsecurity=False).hexdigest()[:16]
