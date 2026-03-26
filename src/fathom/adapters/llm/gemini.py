from __future__ import annotations

import asyncio
import random
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union, cast

from google.genai import Client, types
from google.oauth2 import service_account

from fathom.adapters.llm.cache import CacheService
from fathom.core.exceptions import VisionError
from fathom.core.services.parsing import ToolResponseParser
from fathom.interfaces.llm import LLMPort
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.conversation import ConversationTurn, TurnPart
from fathom.schemas.results import GenerateResult

logger = getLogger(__name__)


class GeminiLLM(LLMPort):
    """
    Gemini adapter for LLM interactions.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gemini-3.1-flash-preview",
        configuration: Optional[LLMConfiguration] = None,
    ) -> None:
        """
        Initialize Gemini LLM adapter.
        """

        if configuration:
            self.__configuration = configuration
        else:
            self.__configuration = LLMConfiguration(api_key=api_key, model=model, use_cache=True)

        self.__client: Optional[Any] = None
        self.__credentials: Optional[Any] = None

        self.__parser = ToolResponseParser()
        self.__cache: Optional[CacheService] = None

        self.__initialize()

    @property
    def model_name(self) -> str:
        """
        Name of the model being used.
        """

        return self.__configuration.model

    def __initialize(self) -> None:
        """
        Initialize client.
        """

        project = self.__configuration.project_id
        location = self.__configuration.location or "global"

        if self.__configuration.credentials:
            if isinstance(self.__configuration.credentials, dict):
                self.__credentials = service_account.Credentials.from_service_account_info(
                    info=self.__configuration.credentials,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                if not project:
                    project = getattr(self.__credentials, "project_id", None)

            elif isinstance(self.__configuration.credentials, str):
                path = Path(self.__configuration.credentials)
                if path.exists():
                    self.__credentials = service_account.Credentials.from_service_account_file(
                        filename=str(path),
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                    if not project:
                        project = getattr(self.__credentials, "project_id", None)
                else:
                    logger.warning(f"Credential file not found at: {path}")

        http_options = {"timeout": int(self.__configuration.timeout * 1000)}

        try:
            if self.__configuration.api_key:
                self.__client = Client(
                    http_options=cast("Any", http_options),
                    api_key=self.__configuration.api_key,
                )
            else:
                self.__client = Client(
                    vertexai=True,
                    project=project,
                    location=location,
                    http_options=cast("Any", http_options),
                    credentials=self.__credentials,
                )

            self.__cache = CacheService(client=self.__client, model_name=self.__configuration.model)
        except Exception as exception:
            if not self.__configuration.api_key and not self.__configuration.credentials:
                raise VisionError("Init failed: Missing Gemini authentication") from exception

            raise VisionError(f"Init failed: {exception}") from exception

    def __get_generation_configuration(
        self,
        cache_name: Optional[str] = None,
        tools: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
    ) -> types.GenerateContentConfig:
        """
        Constructs the GenerateContentConfig using current configuration.
        """

        media_resolution_map = {
            "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
            "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
            "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        }
        configured_resolution = str(self.__configuration.media_resolution).lower()

        thinking_level_map = {
            "minimal": types.ThinkingLevel.MINIMAL,
            "low": types.ThinkingLevel.LOW,
            "medium": types.ThinkingLevel.MEDIUM,
            "high": types.ThinkingLevel.HIGH,
        }
        configured_thinking = getattr(self.__configuration, "thinking_level", "low")
        if isinstance(configured_thinking, str):
            configured_thinking = configured_thinking.lower()
        else:
            configured_thinking = "low"

        config_args: Dict[str, Any] = {
            "candidate_count": 1,
            "automatic_function_calling": {"disable": True},
            "temperature": self.__configuration.temperature,
            "media_resolution": media_resolution_map.get(
                configured_resolution,
                types.MediaResolution.MEDIA_RESOLUTION_LOW,
            ),
        }

        # Add thinking configuration for Gemini 3 series
        if "gemini-3" in self.model_name:
            config_args["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level_map.get(
                    configured_thinking,
                    types.ThinkingLevel.LOW,
                ),
                include_thoughts=getattr(self.__configuration, "include_thoughts", True),
            )

        if not cache_name:
            if system_instruction:
                config_args["system_instruction"] = [{"text": system_instruction}]

            if tools:
                config_args["tools"] = [tools]
                config_args["tool_config"] = {"function_calling_config": {"mode": "ANY"}}
        else:
            config_args["cached_content"] = cache_name

        return types.GenerateContentConfig(**config_args)

    async def generate(
        self,
        *,
        use_cache: bool,
        prompt: Sequence[Union[str, bytes, Dict[str, str]]],
        tools: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
    ) -> GenerateResult:
        """
        Main handler for LLM interaction.
        """

        if not self.__client:
            raise VisionError("Client not ready")

        cache_name = None
        if use_cache and self.__cache and system_instruction:
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
                # Dict[str, str] prompt parts — convert to SDK Part
                text = item.get("text", "")
                if text:
                    parts.append(types.Part.from_text(text=text))
                else:
                    parts.append(types.Part(inline_data=item))

        max_retries = self.__configuration.max_retries
        active_cache_name = cache_name

        for attempt in range(max_retries + 1):
            config = self.__get_generation_configuration(
                tools=tools,
                cache_name=active_cache_name,
                system_instruction=system_instruction,
            )
            try:
                # Build contents: convert provider-neutral turns to Gemini SDK types,
                # then append current user turn.
                if conversation_history:
                    contents = [
                        self.__to_gemini_content(turn) for turn in conversation_history
                    ] + [types.Content(role="user", parts=parts)]
                else:
                    contents = [types.Content(role="user", parts=parts)]

                response = await self.__client.aio.models.generate_content(
                    config=config,
                    model=self.__configuration.model,
                    contents=contents,
                )

                # Extract content
                content = ""
                tool_calls = []

                if response.candidates:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            # Skip thought parts for final content but log them if necessary
                            if hasattr(part, "thought") and part.thought:
                                continue

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
                lower_error = error_msg.lower()
                stale_cached_content = (
                    active_cache_name is not None
                    and (
                        "cached content" in lower_error
                        or "cached_content" in lower_error
                        or "cachedcontent" in lower_error
                    )
                    and ("not found" in lower_error or "invalid" in lower_error)
                )

                if stale_cached_content:
                    logger.warning(
                        "Stale cached content detected (%s); retrying without cache.",
                        active_cache_name,
                    )
                    if self.__cache and active_cache_name is not None:
                        await self.__cache.invalidate_cache_name(cache_name=active_cache_name)
                    active_cache_name = None
                    continue

                if attempt == max_retries:
                    raise VisionError(f"LLM fail: {exception}") from exception

                if is_quota_error:
                    logger.warning(
                        f"Quota exceeded (429). Pausing before retry {attempt + 1}/{max_retries}..."
                    )
                    jitter = random.random() * 2.0  # nosec
                    # Use configured backoff for rate limits
                    delay = (self.__configuration.rate_limit_backoff * (attempt + 1)) + jitter
                else:
                    jitter = random.random() * 0.5  # nosec
                    delay = (self.__configuration.retry_delay * (2**attempt)) + jitter

                await asyncio.sleep(delay=delay)

        raise VisionError("Unreachable")

    async def cleanup(self) -> None:
        """
        Cleanup resources.
        """

        if self.__cache:
            await self.__cache.delete_cache()

    @staticmethod
    def __to_gemini_content(turn: ConversationTurn) -> types.Content:
        """
        Convert a provider-neutral ConversationTurn to a Gemini SDK Content object.

        This is the adapter boundary where domain models are translated to
        provider-specific types.
        """

        sdk_parts: list[types.Part] = []
        for part in turn.parts:
            if part.function_call:
                sdk_parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=part.function_call.name,
                            args=dict(part.function_call.args),
                        )
                    )
                )
            elif part.image_data is not None:
                sdk_parts.append(
                    types.Part.from_bytes(
                        data=part.image_data,
                        mime_type=part.mime_type or "image/png",
                    )
                )
            elif part.text is not None:
                sdk_parts.append(types.Part.from_text(text=part.text))

        return types.Content(role=turn.role, parts=sdk_parts)

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
