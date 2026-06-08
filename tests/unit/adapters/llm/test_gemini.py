from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Dict, Optional
from unittest.mock import patch

from google.genai import types

from fathom.adapters.llm.gemini import GeminiLLM
from fathom.core.exceptions import VisionError
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.llm import GeminiExceptionKind, StructuredOutput
from fathom.schemas.localization import VisionLocalizationPayload


class FakeResponse:
    """
    Simple response double for Gemini error classification tests.
    """

    def __init__(
        self,
        *,
        status_code: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Store response metadata exposed by the SDK exception.
        """

        self.headers = headers or {}
        self.status_code = status_code


class FakeGeminiException(Exception):
    """
    Simple exception double exposing Gemini-like status metadata.
    """

    def __init__(
        self,
        *,
        message: str,
        status_code: Optional[int] = None,
        code: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize exception metadata for classifier helpers.
        """

        super().__init__(message)

        self.code = code
        self.status_code = status_code
        self.response = FakeResponse(status_code=status_code, headers=headers)


class GeminiLLMTest(unittest.TestCase):
    """
    Cover Gemini cache error classification.
    """

    def test_stale_cache_detection_matches_invalid_resource_state(self) -> None:
        """
        Treat invalid cache resource state as a stale cache error.
        """

        result = GeminiLLM._GeminiLLM__is_stale_cached_content_error(
            cache_name="projects/x/locations/global/cachedContents/123",
            status_code=400,
            text=(
                "400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': "
                "'Invalid resource state for cache content 123.', 'status': 'INVALID_ARGUMENT'}}"
            ).casefold(),
        )

        self.assertTrue(result)

    def test_stale_cache_detection_requires_active_cache(self) -> None:
        """
        Do not classify the error as stale cache when no cache is attached.
        """

        result = GeminiLLM._GeminiLLM__is_stale_cached_content_error(
            cache_name=None,
            status_code=400,
            text="Invalid resource state for cache content 123.".casefold(),
        )

        self.assertFalse(result)

    def test_extract_status_code_prefers_structured_exception_fields(self) -> None:
        """
        Read provider status from structured exception metadata when present.
        """

        exception = FakeGeminiException(message="rate limited", status_code=429)

        result = GeminiLLM._GeminiLLM__extract_status_code(exception=exception)

        self.assertEqual(result, 429)

    def test_extract_retry_after_seconds_reads_response_header(self) -> None:
        """
        Use Retry-After when the provider returns it.
        """

        exception = FakeGeminiException(
            status_code=429,
            message="rate limited",
            headers={"retry-after": "7"},
        )

        result = GeminiLLM._GeminiLLM__extract_retry_after_seconds(exception=exception)

        self.assertEqual(result, 7.0)

    def test_build_exception_metadata_sets_exception_kind(self) -> None:
        """
        Normalize provider status and classify the exception in one place.
        """

        gemini = object.__new__(GeminiLLM)
        exception = FakeGeminiException(
            status_code=429,
            headers={"retry-after": "5"},
            message="429 RESOURCE_EXHAUSTED",
        )

        metadata = gemini._GeminiLLM__build_exception_metadata(
            cache_name=None,
            exception=exception,
        )

        self.assertEqual(metadata.status_code, 429)
        self.assertEqual(metadata.retry_after_seconds, 5.0)
        self.assertEqual(metadata.kind, GeminiExceptionKind.RATE_LIMITED)

    def test_provider_overload_detection_uses_status_code(self) -> None:
        """
        Classify 529 as transient provider overload.
        """

        result = GeminiLLM._GeminiLLM__is_provider_overloaded_error(
            status_code=529,
            text="Service overloaded.".casefold(),
        )

        self.assertTrue(result)

    def test_cancelled_detection_uses_status_code(self) -> None:
        """
        Classify 499 as an upstream cancellation signal.
        """

        result = GeminiLLM._GeminiLLM__is_cancelled_error(
            status_code=499,
            text="The operation was cancelled.".casefold(),
        )

        self.assertTrue(result)

    def test_generation_config_maps_unavailable_thinking_levels(self) -> None:
        """Falls back to LOW when the SDK does not expose the requested thinking level."""

        gemini = object.__new__(GeminiLLM)
        gemini._GeminiLLM__configuration = LLMConfiguration(
            model="gemini-3-flash-preview",
            thinking_level="minimal",
        )

        sdk_stub = SimpleNamespace(LOW=types.ThinkingLevel.LOW, HIGH=types.ThinkingLevel.HIGH)
        with patch("fathom.adapters.llm.gemini.types.ThinkingLevel", sdk_stub):
            config = gemini._GeminiLLM__get_generation_configuration()

        self.assertIsNotNone(config.thinking_config)
        self.assertEqual(config.thinking_config.thinking_level, types.ThinkingLevel.LOW)

    def test_async_timeout_does_not_match_cancelled_classifier(self) -> None:
        """Regression: asyncio.wait_for raises asyncio.TimeoutError when the per-attempt budget expires."""

        timeout_message = str(asyncio.TimeoutError()).casefold()
        result = GeminiLLM._GeminiLLM__is_cancelled_error(status_code=None, text=timeout_message)

        self.assertFalse(result)

    def test_async_timeout_does_not_match_rate_limit_classifier(self) -> None:
        """
        asyncio.TimeoutError must NOT classify as RATE_LIMITED either; it should
        fall through to GENERIC so the retry path uses the default backoff
        (not the rate-limit Retry-After path which expects status_code=429).
        """

        timeout_message = str(asyncio.TimeoutError()).casefold()
        result = GeminiLLM._GeminiLLM__is_rate_limit_error(status_code=None, text=timeout_message)

        self.assertFalse(result)


class GeminiStructuredOutputTest(unittest.TestCase):
    """
    Pins how the Gemini adapter translates the production structured-output spec.
    """

    @staticmethod
    def __gemini() -> GeminiLLM:
        """
        Build an unconnected adapter instance with a minimal configuration.
        """

        gemini = object.__new__(GeminiLLM)
        gemini._GeminiLLM__configuration = LLMConfiguration(
            model="gemini-3-flash-preview",
            thinking_level="low",
        )
        return gemini

    def test_structured_output_binds_application_json_and_real_payload(self) -> None:
        """
        The adapter emits the JSON media type and binds the vision-localizer payload.
        """

        gemini = self.__gemini()
        config = gemini._GeminiLLM__get_generation_configuration(
            structured_output=StructuredOutput(payload=VisionLocalizationPayload),
        )

        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIs(config.response_schema, VisionLocalizationPayload)

    def test_no_structured_output_leaves_schema_unset(self) -> None:
        """
        The adapter does not add response_schema when no structured-output spec is supplied.
        """

        gemini = self.__gemini()
        config = gemini._GeminiLLM__get_generation_configuration()

        self.assertIsNone(config.response_mime_type)
        self.assertIsNone(config.response_schema)

    def test_structured_output_with_tools_raises_vision_error(self) -> None:
        """
        Gemini cannot combine structured output with tool calling — fail-fast at the boundary.
        """

        gemini = object.__new__(GeminiLLM)
        gemini._GeminiLLM__configuration = LLMConfiguration(model="gemini-3-flash-preview")
        gemini._GeminiLLM__client = object()
        gemini._GeminiLLM__cache = None

        async def call() -> None:
            """
            Drive a structured-output request that must collide with tool calling.
            """

            await gemini.generate(
                use_cache=False,
                prompt=["hi"],
                tools={"function_declarations": [{"name": "f"}]},
                structured_output=StructuredOutput(payload=VisionLocalizationPayload),
            )

        with self.assertRaises(VisionError):
            asyncio.run(call())


class GeminiUsageMetricsTest(unittest.TestCase):
    """
    Pins how the Gemini adapter captures token usage from SDK responses.
    """

    @staticmethod
    def __response(**usage_overrides: object) -> SimpleNamespace:
        """
        Build a response double whose usage_metadata mirrors the SDK shape.
        """

        usage = SimpleNamespace(
            prompt_token_count=0,
            candidates_token_count=0,
            cached_content_token_count=0,
            thoughts_token_count=0,
            tool_use_prompt_token_count=0,
            total_token_count=0,
            traffic_type=None,
        )
        for attribute, value in usage_overrides.items():
            setattr(usage, attribute, value)
        return SimpleNamespace(usage_metadata=usage)

    def test_collect_captures_every_known_token_field(self) -> None:
        """
        Every documented token counter on the SDK lands in the result metrics.
        """

        response = self.__response(
            prompt_token_count=120,
            candidates_token_count=24,
            cached_content_token_count=8,
            thoughts_token_count=300,
            tool_use_prompt_token_count=16,
            total_token_count=468,
        )

        metrics = GeminiLLM._GeminiLLM__collect_usage_metrics(response=response)

        self.assertEqual(metrics["prompt_tokens"], 120.0)
        self.assertEqual(metrics["completion_tokens"], 24.0)
        self.assertEqual(metrics["cached_tokens"], 8.0)
        self.assertEqual(metrics["thoughts_tokens"], 300.0)
        self.assertEqual(metrics["tool_use_prompt_tokens"], 16.0)
        self.assertEqual(metrics["total_tokens"], 468.0)

    def test_collect_defaults_missing_counters_to_zero(self) -> None:
        """
        Missing or None-valued counters degrade to 0 without raising.
        """

        response = SimpleNamespace(usage_metadata=SimpleNamespace())

        metrics = GeminiLLM._GeminiLLM__collect_usage_metrics(response=response)

        for key in (
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "thoughts_tokens",
            "tool_use_prompt_tokens",
            "total_tokens",
        ):
            self.assertEqual(metrics[key], 0.0)

    def test_collect_returns_empty_when_usage_absent(self) -> None:
        """
        Responses without usage_metadata produce an empty metrics dict.
        """

        response = SimpleNamespace(usage_metadata=None)

        metrics = GeminiLLM._GeminiLLM__collect_usage_metrics(response=response)

        self.assertEqual(metrics, {})
