from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import AsyncMock, patch

from fathom.adapters.llm.gemini import GeminiLLM
from fathom.constants.llm import (
    GEMINI_PRIORITY_TRAFFIC_TYPE,
    GEMINI_RATE_LIMIT_STATUS_CODE,
    GEMINI_VERTEX_PRIORITY_HEADER,
    GEMINI_VERTEX_PRIORITY_REQUEST_TYPE,
    GEMINI_VERTEX_REQUEST_TYPE_HEADER,
    GEMINI_VERTEX_SHARED_REQUEST_TYPE,
    InferencePriorityMode,
)
from fathom.schemas.base.common import ThresholdConfiguration
from fathom.schemas.configuration import (
    AdaptivePriorityConfiguration,
    LLMConfiguration,
    PriorityInferenceConfiguration,
)


class FakeGeminiModels:
    """
    SDK models double that records generateContent configs.
    """

    def __init__(self, *, outcomes: Optional[List[Any]] = None) -> None:
        """
        Initialize recorded call storage.
        """

        self.configs: List[Any] = []
        self.__outcomes = outcomes or []

    async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        """
        Record request input and return a Gemini-shaped response.
        """

        self.configs.append(config)
        _ = model, contents

        if self.__outcomes:
            outcome = self.__outcomes.pop(0)

            if isinstance(outcome, Exception):
                raise outcome

        return SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text='{"ok": true}',
                                function_call=None,
                                thought=False,
                            )
                        ],
                    ),
                )
            ],
            usage_metadata=SimpleNamespace(
                total_token_count=10,
                prompt_token_count=8,
                candidates_token_count=2,
                cached_content_token_count=0,
                thoughts_token_count=0,
                tool_use_prompt_token_count=0,
                traffic_type=SimpleNamespace(name=GEMINI_PRIORITY_TRAFFIC_TYPE),
            ),
        )


class FakeGeminiTransientException(Exception):
    """
    SDK exception double exposing a retryable status code.
    """

    def __init__(self) -> None:
        """
        Initialize retryable Gemini-like exception metadata.
        """

        super().__init__("429 RESOURCE_EXHAUSTED")

        self.status_code = GEMINI_RATE_LIMIT_STATUS_CODE
        self.response = SimpleNamespace(
            status_code=GEMINI_RATE_LIMIT_STATUS_CODE,
            headers={},
        )


class FakeGeminiAsyncClient:
    """
    SDK async client double exposing models.
    """

    def __init__(self, *, models: FakeGeminiModels) -> None:
        """
        Bind fake models endpoint.
        """

        self.models = models


class FakeGeminiClient:
    """
    SDK client double exposing aio models.
    """

    def __init__(self, *, models: FakeGeminiModels) -> None:
        """
        Bind fake async client.
        """

        self.aio = FakeGeminiAsyncClient(models=models)


class GeminiPriorityIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """
    Integration seam tests for Gemini priority request routing.
    """

    async def test_generate_sends_vertex_priority_headers_and_records_metrics(self) -> None:
        """
        The adapter must attach Vertex Priority headers on the actual generate request.
        """

        configuration = LLMConfiguration(
            credentials="/fake/credentials.json",
            model="gemini-3-flash-preview",
            max_retries=0,
            use_cache=False,
        )
        models = FakeGeminiModels()
        fake_client = FakeGeminiClient(models=models)

        with (
            patch.object(
                GeminiLLM,
                "_GeminiLLM__build_client",
                return_value=fake_client,
            ),
            patch("fathom.adapters.llm.gemini.CacheService"),
        ):
            gemini = GeminiLLM(configuration=configuration)

            result = await gemini.generate(
                use_cache=False,
                prompt=["Reply with JSON."],
            )

        self.assertEqual(result.content, '{"ok": true}')
        self.assertEqual(result.metrics["priority_used"], 1.0)
        self.assertEqual(result.metrics["priority_observed"], 1.0)
        self.assertEqual(len(models.configs), 1)
        config = models.configs[0]
        self.assertIsNotNone(config.http_options)
        assert config.http_options is not None
        self.assertEqual(
            config.http_options.headers[GEMINI_VERTEX_REQUEST_TYPE_HEADER],
            GEMINI_VERTEX_SHARED_REQUEST_TYPE,
        )
        self.assertEqual(
            config.http_options.headers[GEMINI_VERTEX_PRIORITY_HEADER],
            GEMINI_VERTEX_PRIORITY_REQUEST_TYPE,
        )

    async def test_generate_logs_adaptive_priority_transition_and_retries_priority(
        self,
    ) -> None:
        """
        Adaptive mode must log why it escalated and retry the next attempt as priority.
        """

        configuration = LLMConfiguration(
            credentials="/fake/credentials.json",
            model="gemini-3-flash-preview",
            max_retries=2,
            retry_delay=0.0,
            rate_limit_backoff=0.0,
            use_cache=False,
            priority=PriorityInferenceConfiguration(
                mode=InferencePriorityMode.ADAPTIVE,
                adaptive=AdaptivePriorityConfiguration(
                    threshold=ThresholdConfiguration(
                        failures=2,
                        slows=3,
                        latency=15.0,
                        recovery=5,
                    ),
                ),
            ),
        )
        models = FakeGeminiModels(
            outcomes=[
                FakeGeminiTransientException(),
                FakeGeminiTransientException(),
            ],
        )
        fake_client = FakeGeminiClient(models=models)

        with (
            patch.object(
                GeminiLLM,
                "_GeminiLLM__build_client",
                return_value=fake_client,
            ),
            patch("fathom.adapters.llm.gemini.CacheService"),
            patch("fathom.adapters.llm.gemini.asyncio.sleep", new_callable=AsyncMock),
            patch("fathom.adapters.llm.gemini.random.random", return_value=0.0),
            patch("fathom.adapters.llm.gemini.logger") as logger,
        ):
            gemini = GeminiLLM(configuration=configuration)

            result = await gemini.generate(
                use_cache=False,
                prompt=["Reply with JSON."],
            )

        self.assertEqual(result.content, '{"ok": true}')
        self.assertEqual(len(models.configs), 3)
        self.assertIsNone(models.configs[0].http_options)
        self.assertIsNone(models.configs[1].http_options)
        self.assertIsNotNone(models.configs[2].http_options)
        assert models.configs[2].http_options is not None
        self.assertEqual(
            models.configs[2].http_options.headers[GEMINI_VERTEX_PRIORITY_HEADER],
            GEMINI_VERTEX_PRIORITY_REQUEST_TYPE,
        )

        transition_logs = [
            call.kwargs["extra"]
            for call in logger.info.call_args_list
            if call.kwargs.get("extra", {}).get("event") == "llm.priority.transition"
        ]
        self.assertEqual(len(transition_logs), 1)
        self.assertEqual(transition_logs[0]["llm.tier.previous"], "standard")
        self.assertEqual(transition_logs[0]["llm.tier.current"], "priority")
        self.assertEqual(transition_logs[0]["llm.priority.reason"], "transient_failures")
        self.assertEqual(transition_logs[0]["llm.priority.failures"], 2)
        self.assertEqual(transition_logs[0]["llm.priority.threshold.failures"], 2)


if __name__ == "__main__":
    unittest.main()
