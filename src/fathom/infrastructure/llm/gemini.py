from __future__ import annotations

import asyncio
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
            path = Path(self.__configuration.credentials_path)
            if path.exists():
                self.__credentials = service_account.Credentials.from_service_account_file(
                    str(path),
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                if not project:
                    project = getattr(self.__credentials, "project_id", None)
            else:
                logger.warning(f"Credential file not found at: {path}")

        if not project:
            project = os.environ.get("GEMINI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")


        http_options = {"timeout": self.__configuration.timeout * 1000}  # ms

        try:
            if self.__configuration.api_key:
                self.__client = genai.Client(
                    http_options=http_options,
                    api_key=self.__configuration.api_key,
                )
            else:
                self.__client = genai.Client(
                    vertexai=True,
                    project=project,
                    location=location,
                    http_options=http_options,
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
                if not item:
                    raise VisionError("Received empty image data for analysis")
                mime_type = self.__detect_mime(data=item)
                parts.append(types.Part.from_bytes(data=item, mime_type=mime_type))
            elif isinstance(item, str):
                parts.append({"text": item})
            else:
                parts.append(item)

        config_args: Dict[str, Any] = {
            "candidate_count": 1,
            "temperature": self.__configuration.temperature,
            "automatic_function_calling": {"disable": True},
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
                result = self.__parser.parse(response)

                # Extract token usage from response
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    result.metrics["prompt_tokens"] = getattr(usage, "prompt_token_count", 0) or 0
                    result.metrics["completion_tokens"] = getattr(usage, "candidates_token_count", 0) or 0
                    result.metrics["cached_tokens"] = getattr(usage, "cached_content_token_count", 0) or 0

                return result
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

    @staticmethod
    def __detect_mime(data: bytes) -> str:
        """
        Detect common image formats from file signatures.
        """

        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
            return "image/gif"
        if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"
