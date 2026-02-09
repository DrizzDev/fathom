from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional

from google.genai.types import Content

logger = getLogger(__name__)


class CacheService:
    """
    Service for managing LLM context caching.
    """

    def __init__(self, client: Any, model_name: str) -> None:
        self.__client = client
        self.__ttl_minutes = 60
        self.__model_name = model_name
        self.__cached_content: Optional[Any] = None

    async def get_cached_content(
        self, system_instruction: str, tools: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """
        Creates or retrieves a cached content object.
        """

        if self.__cached_content:
            return str(self.__cached_content.name)

        try:
            # Accessing the async client property
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

                # Correctly using the async create method
                self.__cached_content = await self.__client.aio.caches.create(
                    model=self.__model_name, config=config
                )
                logger.info(f"Created context cache: {self.__cached_content.name}")
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

        if self.__cached_content:
            try:
                await self.__client.aio.caches.delete(name=self.__cached_content.name)
                logger.info(f"Deleted cache: {self.__cached_content.name}")
                self.__cached_content = None
            except Exception as exception:
                logger.warning(f"Failed to delete cache: {exception}")
