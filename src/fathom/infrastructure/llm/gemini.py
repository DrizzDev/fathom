from __future__ import annotations

import asyncio
import contextlib
import os
import random
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from google.oauth2 import service_account

from fathom.exceptions import VisionError
from fathom.interfaces import IVisionProvider
from fathom.schemas.configuration import GeminiConfig
from fathom.schemas.results import AnalysisResult
from fathom.services.cache import CacheService
from fathom.services.parsing import ToolResponseParser

logger = getLogger(__name__)


class GeminiLLMClient(IVisionProvider):
    """
    Infrastructure client for Gemini API.
    Implements IVisionProvider for plug-and-play usage.
    """

    def __init__(self, configuration: GeminiConfig) -> None:
        self.__configuration = configuration
        self.__client: Optional[Any] = None

        self.__credentials: Optional[Any] = None
        self.__cache: Optional[CacheService] = None

        self.__parser = ToolResponseParser()

        self.__initialize()

    def __initialize(self) -> None:
        """
        Initialize client.
        """

        project = self.__configuration.project_id
        location = self.__configuration.location or "global"

        if self.__configuration.credentials_path:
            with contextlib.suppress(Exception):
                path = Path(self.__configuration.credentials_path)
                if path.exists():
                    self.__credentials = service_account.Credentials.from_service_account_file(
                        str(path),
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                    if not project:
                        project = getattr(self.__credentials, "project_id", None)

        if not project:
            project = os.environ.get("GEMINI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")

        try:
            if self.__configuration.api_key:
                self.__client = genai.Client(api_key=self.__configuration.api_key)
            else:
                self.__client = genai.Client(
                    vertexai=True,
                    project=project,
                    location=location,
                    credentials=self.__credentials,
                )

            self.__cache = CacheService(self.__client, self.__configuration.model)
        except Exception as exception:
            raise VisionError(f"Init failed: {exception}") from exception

    @property
    def credentials(self) -> Any:
        """
        Returns credentials.
        """

        return self.__credentials

    async def analyze(
        self,
        system_instruction: str,
        user_content: List[Any],
        tools: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Main handler for LLM interaction.
        """

        if not self.__client:
            raise VisionError("Client not ready")

        cache_name = None
        if self.__cache:
            cache_name = await self.__cache.get_cached_content(
                system_instruction=system_instruction,
                tools=tools.get("function_declarations") if tools else None,
            )

        # Wrap content parts correctly for SDK
        parts = []
        for item in user_content:
            if isinstance(item, bytes):  # It's an image
                parts.append(types.Part.from_bytes(data=item, mime_type="image/jpeg"))
            elif isinstance(item, str):
                parts.append({"text": item})
            else:
                parts.append(item)

        config_args: Dict[str, Any] = {
            "candidate_count": 1,
            "temperature": self.__configuration.temperature,
        }

        if not cache_name:
            config_args["system_instruction"] = [{"text": system_instruction}]
            if tools:
                config_args["tools"] = [tools]
                config_args["tool_config"] = {"function_calling_config": {"mode": "ANY"}}
        else:
            config_args["cached_content"] = cache_name

        config = types.GenerateContentConfig(**config_args)

        max_retries = self.__configuration.max_retries
        for attempt in range(max_retries + 1):
            try:
                response = await self.__client.aio.models.generate_content(
                    config=config,
                    model=self.__configuration.model,
                    contents=[types.Content(role="user", parts=parts)],
                )
                return self.__parser.parse(response)
            except Exception as exception:
                if attempt == max_retries:
                    raise VisionError(f"LLM fail: {exception}") from exception

                delay = (self.__configuration.retry_delay * (2**attempt)) + (random.random() * 0.5)  # nosec
                await asyncio.sleep(delay)

        raise VisionError("Unreachable")

    async def cleanup(self) -> None:
        """
        Cleanup resources.
        """

        if self.__cache:
            await self.__cache.delete_cache()
