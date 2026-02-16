"""Gemini LLM adapter - wraps existing Gemini client logic."""

from __future__ import annotations

import asyncio
import random
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from google.oauth2 import service_account

from fathom.core.services.parsing import ToolResponseParser
from fathom.exceptions import VisionError
from fathom.interfaces.llm import LLMPort
from fathom.schemas.configuration import GeminiConfig
from fathom.schemas.results import GenerateResult
from fathom.adapters.llm.cache import CacheService

logger = getLogger(__name__)


class GeminiLLM(LLMPort):
    """
    Gemini adapter for LLM interactions.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash-exp",
        configuration: Optional[GeminiConfig] = None,
    ) -> None:
        """Initialize Gemini LLM adapter."""
        if configuration:
            self.__configuration = configuration
        else:
            self.__configuration = GeminiConfig(api_key=api_key, model=model)

        self.__client: Optional[Any] = None
        self.__credentials: Optional[Any] = None
        self.__cache: Optional[CacheService] = None
        self.__parser = ToolResponseParser()

        self.__initialize()

    @property
    def model_name(self) -> str:
        """Name of the model being used."""
        return self.__configuration.model

    def __initialize(self) -> None:
        """Initialize client."""
        project = self.__configuration.project_id
        location = self.__configuration.location or "global"

        if self.__configuration.credentials_path:
            path = Path(self.__configuration.credentials_path)
            if path.exists():
                self.__credentials = service_account.Credentials.from_service_account_file(
                    filename=str(path),
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                if not project:
                    project = getattr(self.__credentials, "project_id", None)
            else:
                logger.warning(f"Credential file not found at: {path}")

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

            self.__cache = CacheService(client=self.__client, model_name=self.__configuration.model)
        except Exception as exception:
            raise VisionError(f"Init failed: {exception}") from exception

    async def generate(
        self,
        *,
        prompt: List[Any],
        system_instruction: Optional[str] = None,
        tools: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        """Main handler for LLM interaction."""
        if not self.__client:
            raise VisionError("Client not ready")

        cache_name = None
        if self.__cache and system_instruction:
            cache_name = await self.__cache.get_cached_content(
                system_instruction=system_instruction,
                tools=tools.get("function_declarations") if tools else None,
            )

        # Wrap content parts correctly for SDK
        parts = []
        for item in prompt:
            if isinstance(item, bytes):  # It's an image
                if not item:
                    raise VisionError("Received empty image data for analysis")
                mime_type = self.__detect_mime(data=item)
                parts.append(types.Part.from_bytes(data=item, mime_type=mime_type))
            elif isinstance(item, str):
                parts.append(types.Part.from_text(text=item))
            else:
                parts.append(item)

        config_args: Dict[str, Any] = {
            "candidate_count": 1,
            "temperature": self.__configuration.temperature,
            "automatic_function_calling": {"disable": True},
        }

        if not cache_name:
            if system_instruction:
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

                # Extract content
                content = ""
                tool_calls = []
                if response.candidates:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.text:
                                content += part.text
                            if part.function_call:
                                tool_calls.append(part.function_call)

                # Extract token usage
                metrics = {}
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    metrics["prompt_tokens"] = float(getattr(usage, "prompt_token_count", 0) or 0)
                    metrics["completion_tokens"] = float(
                        getattr(usage, "candidates_token_count", 0) or 0
                    )
                    metrics["cached_tokens"] = float(
                        getattr(usage, "cached_content_token_count", 0) or 0
                    )

                return GenerateResult(content=content, tool_calls=tool_calls, metrics=metrics)

            except Exception as exception:
                error_msg = str(exception)
                is_quota_error = "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg

                if attempt == max_retries:
                    raise VisionError(f"LLM fail: {exception}") from exception

                if is_quota_error:
                    logger.warning(
                        f"Quota exceeded (429). Pausing before retry {attempt + 1}/{max_retries}..."
                    )
                    jitter = random.random() * 5.0  # nosec
                    delay = 30.0 + jitter
                else:
                    jitter = random.random() * 0.5  # nosec
                    delay = (1.0 * (2**attempt)) + jitter  # Using 1.0s base delay

                await asyncio.sleep(delay=delay)

        raise VisionError("Unreachable")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.__cache:
            await self.__cache.delete_cache()

    @staticmethod
    def __detect_mime(data: bytes) -> str:
        """Detect common image formats from file signatures."""
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
            return "image/gif"
        if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"
