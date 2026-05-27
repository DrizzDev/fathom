from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Dict, Optional
from unittest.mock import patch

from google.genai import types

from fathom.adapters.llm.gemini import GeminiLLM
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.llm import GeminiExceptionKind


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
