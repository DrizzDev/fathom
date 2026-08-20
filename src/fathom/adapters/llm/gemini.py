from __future__ import annotations

import asyncio
import random
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from google.genai import Client, types
from google.oauth2 import service_account

from fathom.adapters.llm.cache import CacheService
from fathom.constants.llm import (
    GEMINI_CANCELLED_ERROR_MARKERS,
    GEMINI_CANCELLED_STATUS_CODE,
    GEMINI_GENERIC_RETRY_JITTER_SECONDS,
    GEMINI_MAX_TRANSIENT_RETRY_DELAY_SECONDS,
    GEMINI_PRIORITY_SERVICE_TIER_VALUE,
    GEMINI_PRIORITY_TRAFFIC_TYPE,
    GEMINI_PROVIDER_OVERLOAD_ERROR_MARKERS,
    GEMINI_PROVIDER_OVERLOADED_STATUS_CODE,
    GEMINI_RATE_LIMIT_ERROR_MARKERS,
    GEMINI_RATE_LIMIT_STATUS_CODE,
    GEMINI_RETRY_AFTER_JITTER_SECONDS,
    GEMINI_STALE_CACHE_NAME_MARKERS,
    GEMINI_STALE_CACHE_STATE_MARKERS,
    GEMINI_STALE_CACHE_STATUS_CODE,
    GEMINI_TRANSIENT_RETRY_JITTER_SECONDS,
    GEMINI_TRANSIENT_STATUS_CODES,
    GEMINI_VERTEX_PRIORITY_HEADER,
    GEMINI_VERTEX_PRIORITY_REQUEST_TYPE,
    GEMINI_VERTEX_REQUEST_TYPE_HEADER,
    GEMINI_VERTEX_SHARED_REQUEST_TYPE,
    InferenceTier,
    StructuredOutputMediaType,
)
from fathom.core.exceptions import VisionError
from fathom.core.inference.priority import PriorityInferencePolicy
from fathom.core.services.parsing import ToolResponseParser
from fathom.interfaces.llm import LLMPort
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.llm import (
    GeminiExceptionKind,
    GeminiExceptionMetadata,
    PriorityInferenceSignal,
    PriorityInferenceTransition,
    StructuredOutput,
)
from fathom.schemas.results import GenerateResult

logger = getLogger(__name__)


class GeminiLLM(LLMPort):
    """
    Gemini implementation of :class:`LLMPort`.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gemini-3.5-flash",
        configuration: Optional[LLMConfiguration] = None,
    ) -> None:
        """
        Resolve configuration, set up the priority policy and parser, then build the SDK client and cache.
        """

        if configuration:
            self.__configuration = configuration
        else:
            self.__configuration = LLMConfiguration(api_key=api_key, model=model, use_cache=True)

        self.__client: Optional[Any] = None
        self.__credentials: Optional[Any] = None
        self.__priority = PriorityInferencePolicy(configuration=self.__configuration.priority)

        self.__parser = ToolResponseParser()
        self.__cache: Optional[CacheService] = None

        self.__initialize()

    @property
    def model_name(self) -> str:
        """
        Name of the model being used.
        """

        return self.__configuration.model

    def derive(self, *, overrides: LLMConfiguration) -> LLMPort:
        """
        Spawn a sibling adapter sharing credentials and model, applying only the
        explicitly-set fields of ``overrides``.
        """

        return GeminiLLM(
            configuration=self.__configuration.model_copy(
                update=overrides.model_dump(exclude_unset=True)
            )
        )

    @property
    def client(self) -> Any:
        """
        Underlying ``google.genai`` client; consumed by sibling adapters that share auth.
        """

        return self.__client

    def __initialize(self) -> None:
        """
        Resolve service-account credentials when configured, then build the SDK client and context cache.
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
                    logger.warning(
                        "Gemini credentials file path does not exist",
                        extra={
                            "credentials.path": str(path),
                            "component": "adapters.llm.gemini",
                            "event": "gemini.credentials.path.missing",
                        },
                    )

        try:
            self.__client = self.__build_client(project=project, location=location)

            self.__cache = CacheService(client=self.__client, model_name=self.__configuration.model)
        except Exception as exception:
            if not self.__configuration.api_key and not self.__configuration.credentials:
                raise VisionError("Init failed: Missing Gemini authentication") from exception

            raise VisionError(f"Init failed: {exception}") from exception

    def __build_client(self, *, project: Optional[str], location: str) -> Client:
        """
        Build the base Gemini client for all SDK operations.
        """

        if self.__configuration.api_key:
            return Client(
                api_key=self.__configuration.api_key,
                http_options=self.__base_http_options(),
            )

        return Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=self.__credentials,
            http_options=self.__base_http_options(),
        )

    def __base_http_options(self) -> types.HttpOptions:
        """
        Return SDK HTTP options shared by all requests.
        """

        return types.HttpOptions(
            api_version="v1",
            timeout=int(self.__configuration.timeout * 1000),
        )

    def __get_generation_configuration(
        self,
        cache_name: Optional[str] = None,
        tools: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        tier: InferenceTier = InferenceTier.STANDARD,
        structured_output: Optional[StructuredOutput] = None,
    ) -> types.GenerateContentConfig:
        """
        Assemble the SDK ``GenerateContentConfig`` from the current inference settings.
        """

        media_resolution_map = {
            "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
            "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        }
        configured_resolution = str(self.__configuration.media_resolution).lower()

        thinking_level_map = {
            "low": types.ThinkingLevel.LOW,
            "high": types.ThinkingLevel.HIGH,
            "medium": getattr(types.ThinkingLevel, "MEDIUM", types.ThinkingLevel.HIGH),
            "minimal": getattr(types.ThinkingLevel, "MINIMAL", types.ThinkingLevel.LOW),
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

        if structured_output is not None:
            config_args["response_schema"] = structured_output.payload
            config_args["response_mime_type"] = StructuredOutputMediaType.JSON.value

        if self.__uses_service_tier_config(tier=tier):
            config_args["service_tier"] = GEMINI_PRIORITY_SERVICE_TIER_VALUE

        if http_options := self.__priority_http_options(tier=tier):
            config_args["http_options"] = http_options

        return types.GenerateContentConfig(**config_args)

    def __uses_service_tier_config(self, *, tier: InferenceTier) -> bool:
        """
        Return whether Gemini API priority should be sent in generation config.
        """

        return tier == InferenceTier.PRIORITY and bool(self.__configuration.api_key)

    def __priority_http_options(self, *, tier: InferenceTier) -> Optional[types.HttpOptions]:
        """
        Return the per-request SDK header patch for Vertex priority routing.
        """

        if tier != InferenceTier.PRIORITY or self.__configuration.api_key:
            return None

        return types.HttpOptions(
            headers={
                GEMINI_VERTEX_PRIORITY_HEADER: GEMINI_VERTEX_PRIORITY_REQUEST_TYPE,
                GEMINI_VERTEX_REQUEST_TYPE_HEADER: GEMINI_VERTEX_SHARED_REQUEST_TYPE,
            },
        )

    def __is_transient_failure(
        self,
        *,
        exception: Exception,
        metadata: GeminiExceptionMetadata,
    ) -> bool:
        """
        Return whether a Gemini failure should feed adaptive priority escalation.
        """

        if isinstance(exception, asyncio.TimeoutError):
            return True

        if metadata.kind in {
            GeminiExceptionKind.RATE_LIMITED,
            GeminiExceptionKind.PROVIDER_OVERLOADED,
        }:
            return True

        return metadata.status_code in GEMINI_TRANSIENT_STATUS_CODES

    @staticmethod
    def __traffic_type(*, usage: Optional[object]) -> Optional[str]:
        """
        Normalize provider-reported traffic type when the SDK returns it.
        """

        if usage is None:
            return None

        traffic_type = getattr(usage, "traffic_type", None)
        if traffic_type is None:
            return None

        name = getattr(traffic_type, "name", None)
        if isinstance(name, str):
            return name

        value = getattr(traffic_type, "value", None)
        if isinstance(value, str):
            return value

        return str(traffic_type).removeprefix("TrafficType.")

    async def generate(
        self,
        *,
        use_cache: bool,
        prompt: Sequence[Union[str, bytes, Dict[str, str]]],
        tools: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        structured_output: Optional[StructuredOutput] = None,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
    ) -> GenerateResult:
        """
        Run one generation: reuse or skip the context cache, build SDK contents from the prompt and
        history, then call Gemini under a per-attempt timeout with adaptive-priority retry.
        """

        if not self.__client:
            raise VisionError("Client not ready")

        if structured_output is not None and tools:
            raise VisionError("Structured-output mode cannot be combined with tool calling")

        if use_cache and self.__cache and system_instruction and structured_output is None:
            cache_name = await self.__cache.get_cached_content(
                system_instruction=system_instruction,
                tools=tools.get("function_declarations") if tools else None,
            )
        else:
            cache_name = None

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

        active_cache_name = cache_name
        max_retries = self.__configuration.max_retries

        for attempt in range(max_retries + 1):
            tier = self.__priority.select()

            config = self.__get_generation_configuration(
                tier=tier,
                tools=tools,
                cache_name=active_cache_name,
                structured_output=structured_output,
                system_instruction=system_instruction,
            )
            try:
                # Build contents: convert provider-neutral turns to Gemini SDK types,
                # then append current user turn.
                if conversation_history:
                    contents = [self.__to_gemini_content(turn) for turn in conversation_history] + [
                        types.Content(role="user", parts=parts)
                    ]
                else:
                    contents = [types.Content(role="user", parts=parts)]

                # Per-attempt timeout — caps a single Gemini call so a slow tail latency event (preview-model variance, regional slow path)
                # cannot stall the caller for the full HTTP timeout. On expiry, asyncio.TimeoutError is raised, classified as GENERIC
                # by __build_exception_metadata, and the retry path engages with backoff + jitter just like any other transient failure.
                started = time.perf_counter()
                response = await asyncio.wait_for(
                    self.__client.aio.models.generate_content(
                        config=config,
                        contents=contents,
                        model=self.__configuration.model,
                    ),
                    timeout=self.__configuration.timeout,
                )
                latency = time.perf_counter() - started
                usage = getattr(response, "usage_metadata", None)
                observed_traffic = self.__traffic_type(usage=usage)

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

                metrics = self.__collect_usage_metrics(response=response, tier=tier)
                logger.info(
                    "Gemini request completed",
                    extra={
                        "component": "adapters.llm.gemini",
                        "event": "gemini.request.completed",
                        "latency": latency,
                        "llm.tier.requested": tier.value,
                        "llm.traffic.type": observed_traffic,
                        "llm.model": self.__configuration.model,
                    },
                )
                transition = self.__priority.record(
                    signal=PriorityInferenceSignal(
                        tier=tier,
                        success=True,
                        latency=latency,
                    ),
                )
                self.__log_priority_transition(transition=transition)

                return GenerateResult(content=content, tool_calls=tool_calls, metrics=metrics)

            except Exception as exception:
                metadata = self.__build_exception_metadata(
                    exception=exception,
                    cache_name=active_cache_name,
                )
                self.__log_generation_exception(
                    tier=tier,
                    attempt=attempt,
                    metadata=metadata,
                    max_retries=max_retries,
                    cache_name=active_cache_name,
                )

                if metadata.kind == GeminiExceptionKind.STALE_CACHED_CONTENT:
                    active_cache_name = await self.__reset_stale_cache(cache_name=active_cache_name)
                    continue

                transition = self.__priority.record(
                    signal=PriorityInferenceSignal(
                        tier=tier,
                        success=False,
                        transient=self.__is_transient_failure(
                            metadata=metadata, exception=exception
                        ),
                    ),
                )
                self.__log_priority_transition(transition=transition)

                if metadata.kind == GeminiExceptionKind.CANCELLED:
                    raise VisionError(f"LLM cancelled: {exception}") from exception

                if attempt == max_retries:
                    raise VisionError(f"LLM fail: {exception}") from exception

                delay = self.__retry_delay(attempt=attempt, metadata=metadata)

                if delay is not None:
                    logger.warning(
                        "Gemini transient failure; scheduling retry",
                        extra={
                            "component": "adapters.llm.gemini",
                            "event": "gemini.request.retry.scheduled",
                            "retry.delay_seconds": delay,
                            "retry.attempt": attempt + 1,
                            "llm.tier.requested": tier.value,
                            "retry.max_attempts": max_retries,
                            "exception.kind": metadata.kind.value,
                            "exception.status_code": metadata.status_code,
                        },
                    )
                else:
                    jitter = random.random() * GEMINI_GENERIC_RETRY_JITTER_SECONDS  # nosec
                    delay = (self.__configuration.retry_delay * (2**attempt)) + jitter

                await asyncio.sleep(delay=delay)

        raise VisionError("Unreachable")

    def __log_priority_transition(
        self, *, transition: Optional[PriorityInferenceTransition]
    ) -> None:
        """
        Emit structured diagnostics when adaptive priority changes tier.
        """

        if transition is None:
            return

        evidence = transition.evidence
        logger.info(
            "LLM priority tier changed",
            extra={
                "component": "adapters.llm.gemini",
                "event": "llm.priority.transition",
                "llm.model": self.__configuration.model,
                "llm.tier.previous": transition.previous.value,
                "llm.tier.current": transition.current.value,
                "llm.priority.reason": transition.reason.value,
                "llm.priority.window": evidence.window,
                "llm.priority.failures": evidence.failures,
                "llm.priority.slows": evidence.slows,
                "llm.priority.healthy": evidence.healthy,
                "llm.priority.threshold.failures": evidence.threshold.failures,
                "llm.priority.threshold.slows": evidence.threshold.slows,
                "llm.priority.threshold.latency": evidence.threshold.latency,
                "llm.priority.threshold.recovery": evidence.threshold.recovery,
            },
        )

    @classmethod
    def __collect_usage_metrics(
        cls,
        *,
        response: Any,
        tier: Optional[InferenceTier] = None,
    ) -> Dict[str, float]:
        """
        Capture token usage from the SDK response and emit a structured usage event.
        """

        metrics: Dict[str, float] = {}
        usage = getattr(response, "usage_metadata", None)

        if tier is not None:
            observed_traffic = cls.__traffic_type(usage=usage)
            metrics["priority_used"] = 1.0 if tier == InferenceTier.PRIORITY else 0.0
            metrics["priority_observed"] = (
                1.0 if observed_traffic == GEMINI_PRIORITY_TRAFFIC_TYPE else 0.0
            )

        if usage is None:
            return metrics

        for metric_name, attribute in (
            ("total_tokens", "total_token_count"),
            ("prompt_tokens", "prompt_token_count"),
            ("thoughts_tokens", "thoughts_token_count"),
            ("completion_tokens", "candidates_token_count"),
            ("cached_tokens", "cached_content_token_count"),
            ("tool_use_prompt_tokens", "tool_use_prompt_token_count"),
        ):
            metrics[metric_name] = float(getattr(usage, attribute, 0) or 0)

        logger.info(
            "Gemini usage recorded",
            extra={
                "component": "adapters.llm.gemini",
                "event": "gemini.usage.recorded",
                "tokens.total": metrics["total_tokens"],
                "tokens.prompt": metrics["prompt_tokens"],
                "tokens.cached": metrics["cached_tokens"],
                "tokens.thoughts": metrics["thoughts_tokens"],
                "tokens.completion": metrics["completion_tokens"],
                "traffic.type": getattr(usage, "traffic_type", None),
                "tokens.tool_use_prompt": metrics["tool_use_prompt_tokens"],
            },
        )

        return metrics

    async def __reset_stale_cache(self, *, cache_name: Optional[str]) -> Optional[str]:
        """
        Invalidate the active cached content and continue uncached.
        """

        logger.warning(
            "Stale Gemini cached content detected; retrying uncached",
            extra={
                "component": "adapters.llm.gemini",
                "event": "gemini.cache.stale",
                "cache.name": cache_name,
            },
        )

        if self.__cache and cache_name is not None:
            await self.__cache.invalidate_cache_name(cache_name=cache_name)

        return None

    def __retry_delay(self, *, attempt: int, metadata: GeminiExceptionMetadata) -> Optional[float]:
        """
        Resolve the retry delay for transient provider failures.
        """

        if metadata.kind not in {
            GeminiExceptionKind.RATE_LIMITED,
            GeminiExceptionKind.PROVIDER_OVERLOADED,
        }:
            return None

        return self.__compute_transient_retry_delay(
            attempt=attempt,
            retry_after_seconds=metadata.retry_after_seconds,
            rate_limited=metadata.kind == GeminiExceptionKind.RATE_LIMITED,
        )

    async def prewarm(
        self,
        *,
        system_instruction: Optional[str],
        tools: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Prewarm provider-side prompt cache before the first LLM call.
        """

        if not system_instruction or not self.__cache:
            return

        if not self.__configuration.use_cache:
            return

        declarations = tools.get("function_declarations") if tools is not None else None

        await self.__cache.get_cached_content(
            tools=declarations,
            system_instruction=system_instruction,
        )

    async def cleanup(self) -> None:
        """
        Delete the provider cache, then close the SDK client.
        """

        try:
            await self.__delete_cache()
        finally:
            await self.__close_client()

    async def __delete_cache(self) -> None:
        """
        Delete provider cache once when the adapter owns one.
        """

        if cache := self.__cache:
            self.__cache = None
            await cache.delete_cache()

    async def __close_client(self) -> None:
        """
        Close async and sync SDK client resources once.
        """

        if client := self.__client:
            self.__client = None
            try:
                await client.aio.aclose()
            finally:
                client.close()

    def __build_exception_metadata(
        self, *, exception: Exception, cache_name: Optional[str]
    ) -> GeminiExceptionMetadata:
        """
        Normalize a Gemini exception into retry and cache-recovery metadata.
        """

        message = str(exception)
        status_code = self.__extract_status_code(exception=exception)
        retry_after_seconds = self.__extract_retry_after_seconds(exception=exception)

        text = message.casefold()
        kind = GeminiExceptionKind.GENERIC

        if self.__is_stale_cached_content_error(
            text=text,
            cache_name=cache_name,
            status_code=status_code,
        ):
            kind = GeminiExceptionKind.STALE_CACHED_CONTENT
        elif self.__is_cancelled_error(status_code=status_code, text=text):
            kind = GeminiExceptionKind.CANCELLED

        elif self.__is_rate_limit_error(status_code=status_code, text=text):
            kind = GeminiExceptionKind.RATE_LIMITED

        elif self.__is_provider_overloaded_error(status_code=status_code, text=text):
            kind = GeminiExceptionKind.PROVIDER_OVERLOADED

        return GeminiExceptionMetadata(
            kind=kind,
            message=message,
            status_code=status_code,
            exception_type=type(exception).__name__,
            retry_after_seconds=retry_after_seconds,
        )

    def __log_generation_exception(
        self,
        *,
        attempt: int,
        max_retries: int,
        tier: InferenceTier,
        cache_name: Optional[str],
        metadata: GeminiExceptionMetadata,
    ) -> None:
        """
        Log the full provider exception with normalized metadata for later diagnosis.
        """

        logger.warning(
            "Gemini request failed",
            extra={
                "component": "adapters.llm.gemini",
                "event": "gemini.request.failed",
                "cache.name": cache_name,
                "retry.attempt": attempt + 1,
                "llm.tier.requested": tier.value,
                "exception.kind": metadata.kind.value,
                "exception.message": metadata.message,
                "retry.max_attempts": max_retries + 1,
                "exception.type": metadata.exception_type,
                "exception.status_code": metadata.status_code,
                "retry.after_seconds": metadata.retry_after_seconds,
            },
            exc_info=True,
        )

    @staticmethod
    def __extract_status_code(*, exception: Exception) -> Optional[int]:
        """
        Extract an HTTP-style status code from a Gemini SDK exception when available.
        """

        status_code = getattr(exception, "status_code", None)
        if isinstance(status_code, int):
            return status_code

        code = getattr(exception, "code", None)
        if isinstance(code, int):
            return code

        response = getattr(exception, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status

        return None

    @staticmethod
    def __extract_retry_after_seconds(*, exception: Exception) -> Optional[float]:
        """
        Extract a Retry-After delay from the provider response when present.
        """

        response = getattr(exception, "response", None)
        headers = getattr(response, "headers", None)

        if not isinstance(headers, Mapping):
            return None

        if (retry_after := headers.get("retry-after")) is None:
            return None

        try:
            return max(float(retry_after), 0.0)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def __contains_error_marker(*, text: str, markers: Sequence[str]) -> bool:
        """
        Determine whether a normalized provider message contains any known marker.
        """

        return any(marker in text for marker in markers)

    @staticmethod
    def __is_rate_limit_error(*, status_code: Optional[int], text: str) -> bool:
        """
        Determine whether the provider rejected the request due to rate limiting.
        """

        if status_code is not None:
            return status_code == GEMINI_RATE_LIMIT_STATUS_CODE

        return GeminiLLM.__contains_error_marker(
            text=text,
            markers=GEMINI_RATE_LIMIT_ERROR_MARKERS,
        )

    @staticmethod
    def __is_provider_overloaded_error(
        *,
        text: str,
        status_code: Optional[int],
    ) -> bool:
        """
        Determine whether the provider is temporarily overloaded.
        """

        if status_code is not None:
            return status_code == GEMINI_PROVIDER_OVERLOADED_STATUS_CODE

        return GeminiLLM.__contains_error_marker(
            text=text,
            markers=GEMINI_PROVIDER_OVERLOAD_ERROR_MARKERS,
        )

    @staticmethod
    def __is_cancelled_error(*, status_code: Optional[int], text: str) -> bool:
        """
        Determine whether the provider request was cancelled upstream.
        """

        if status_code is not None:
            return status_code == GEMINI_CANCELLED_STATUS_CODE

        return GeminiLLM.__contains_error_marker(
            text=text,
            markers=GEMINI_CANCELLED_ERROR_MARKERS,
        )

    def __compute_transient_retry_delay(
        self,
        *,
        attempt: int,
        rate_limited: bool,
        retry_after_seconds: Optional[float],
    ) -> float:
        """
        Compute a backoff delay for provider throttling or overload conditions.
        """

        if retry_after_seconds is not None:
            jitter = random.random() * GEMINI_RETRY_AFTER_JITTER_SECONDS  # nosec
            return float(retry_after_seconds + jitter)

        backoff_base = (
            float(self.__configuration.rate_limit_backoff)
            if rate_limited
            else max(float(self.__configuration.rate_limit_backoff) / 2.0, 1.0)
        )
        jitter = random.random() * GEMINI_TRANSIENT_RETRY_JITTER_SECONDS  # nosec

        return float(
            min(
                backoff_base * (2**attempt),
                GEMINI_MAX_TRANSIENT_RETRY_DELAY_SECONDS,
            )
            + jitter
        )

    @staticmethod
    def __is_stale_cached_content_error(
        *, text: str, cache_name: Optional[str], status_code: Optional[int]
    ) -> bool:
        """
        Determine whether the provider rejected the currently attached cached content.
        """

        if cache_name is None:
            return False

        if status_code is not None and status_code != GEMINI_STALE_CACHE_STATUS_CODE:
            return False

        return GeminiLLM.__contains_error_marker(
            text=text,
            markers=GEMINI_STALE_CACHE_NAME_MARKERS,
        ) and GeminiLLM.__contains_error_marker(
            text=text,
            markers=GEMINI_STALE_CACHE_STATE_MARKERS,
        )

    @staticmethod
    def __to_gemini_content(turn: ConversationTurn) -> types.Content:
        """
        Convert a provider-neutral ConversationTurn to a Gemini SDK Content object.

        This is the adapter boundary where domain models are translated to provider-specific types.
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
