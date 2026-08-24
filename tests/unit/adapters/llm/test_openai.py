from __future__ import annotations

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase

from pydantic import BaseModel

from fathom.adapters.llm.openai import OpenAILLM
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.llm import StructuredOutput


class _FakeCompletions:
    """
    Minimal async Chat Completions surface used to test adapter normalization without network I/O.
    """

    def __init__(self, response: object) -> None:
        self.response = response
        self.request = None

    async def create(self, **request: object) -> object:
        self.request = request
        return self.response


class _FakeClient:
    """
    Minimal client surface matching the adapter's async SDK usage.
    """

    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)

    async def close(self) -> None:
        return None


class OpenAILLMTest(IsolatedAsyncioTestCase):
    """
    OpenAI adapter contract tests over a fake SDK client.
    """

    @staticmethod
    def __adapter(*, response: object) -> tuple[OpenAILLM, _FakeCompletions]:
        """
        Construct the adapter and replace only its transport with a deterministic fake.
        """

        adapter = OpenAILLM(
            configuration=LLMConfiguration(
                provider="openai",
                api_key="test-key",
                model="gpt-4.1-mini",
                max_retries=0,
            )
        )
        completions = _FakeCompletions(response=response)
        adapter._OpenAILLM__client = _FakeClient(completions)  # type: ignore[attr-defined]
        return adapter, completions

    async def test_tool_calls_and_usage_are_normalized(self) -> None:
        """
        OpenAI function calls become the provider-neutral name/args shape consumed by the parser.
        """

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="execute_ui",
                                    arguments='{"action_type":"tap"}',
                                )
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=4,
                prompt_tokens_details=SimpleNamespace(cached_tokens=2),
            ),
        )
        adapter, completions = self.__adapter(response=response)

        result = await adapter.generate(
            use_cache=False,
            prompt=["Open the app"],
            tools={
                "tools": [
                    {
                        "name": "execute_ui",
                        "description": "Execute UI action",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ]
            },
        )

        self.assertEqual(result.tool_calls[0].name, "execute_ui")
        self.assertEqual(result.tool_calls[0].args, {"action_type": "tap"})
        self.assertEqual(result.metrics["total_tokens"], 14)
        self.assertEqual(completions.request["tools"][0]["type"], "function")
        self.assertEqual(completions.request["tool_choice"], "required")

    async def test_structured_output_is_mapped_to_json_schema(self) -> None:
        """
        Fathom's Pydantic structured-output contract becomes OpenAI's JSON-schema response format.
        """

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"ok":true}', tool_calls=[]))
            ],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
        )
        adapter, completions = self.__adapter(response=response)

        # Use a real structured-output Pydantic model while keeping this test local.
        class ResultPayload(BaseModel):
            ok: bool

        result = await adapter.generate(
            use_cache=False,
            prompt=["Return JSON"],
            structured_output=StructuredOutput(payload=ResultPayload),
        )

        self.assertEqual(result.content, '{"ok":true}')
        self.assertEqual(
            completions.request["response_format"]["type"],
            "json_schema",
        )

    async def test_structured_output_removes_pydantic_keywords_rejected_by_openai(self) -> None:
        """Strict schemas contain only the OpenAI-supported structural constraints."""

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"ok":true}', tool_calls=[]))
            ],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
        )
        adapter, completions = self.__adapter(response=response)

        class ResultPayload(BaseModel):
            ok: bool = True

        await adapter.generate(
            use_cache=False,
            prompt=["Return JSON"],
            structured_output=StructuredOutput(payload=ResultPayload),
        )

        schema = completions.request["response_format"]["json_schema"]["schema"]
        self.assertNotIn("default", schema["properties"]["ok"])
        self.assertEqual(schema["required"], ["ok"])

    async def test_structured_output_converts_const_and_removes_titles(self) -> None:
        """OpenAI strict schemas use enum for fixed values and omit Pydantic titles."""

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"kind":"fixed"}', tool_calls=[]))
            ],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
        )
        adapter, completions = self.__adapter(response=response)

        class ResultPayload(BaseModel):
            kind: str = "fixed"

        await adapter.generate(
            use_cache=False,
            prompt=["Return JSON"],
            structured_output=StructuredOutput(payload=ResultPayload),
        )

        schema = completions.request["response_format"]["json_schema"]["schema"]
        self.assertNotIn("title", schema)

    async def test_structured_output_makes_refs_standalone(self) -> None:
        """OpenAI rejects descriptions emitted beside JSON Schema references."""

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"child":{"value":"ok"}}', tool_calls=[])
                )
            ],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
        )
        adapter, completions = self.__adapter(response=response)

        class Child(BaseModel):
            value: str

        class ResultPayload(BaseModel):
            child: Child

        await adapter.generate(
            use_cache=False,
            prompt=["Return JSON"],
            structured_output=StructuredOutput(payload=ResultPayload),
        )

        schema = completions.request["response_format"]["json_schema"]["schema"]
        self.assertEqual(list(schema["properties"]["child"]), ["$ref"])

    async def test_reasoning_models_do_not_receive_temperature(self) -> None:
        """
        GPT-5.2-compatible requests omit the unsupported temperature parameter.
        """

        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )
        adapter = OpenAILLM(
            configuration=LLMConfiguration(
                provider="openai",
                api_key="test-key",
                model="gpt-5.2",
                max_retries=0,
            )
        )
        completions = _FakeCompletions(response=response)
        adapter._OpenAILLM__client = _FakeClient(completions)  # type: ignore[attr-defined]

        await adapter.generate(use_cache=False, prompt=["Hello"])

        self.assertNotIn("temperature", completions.request)

    def test_reasoning_identity_reports_only_settings_sent_to_openai(self) -> None:
        """
        Reasoning-model identity must not claim controls the adapter does not transmit.
        """

        adapter = OpenAILLM(
            configuration=LLMConfiguration(
                provider="openai",
                api_key="test-key",
                model="gpt-5.2",
                temperature=0.7,
                use_cache=True,
            )
        )

        self.assertIsNone(adapter.identity.settings.temperature)
        self.assertIsNone(adapter.identity.settings.cache)

    def test_non_reasoning_identity_reports_transmitted_temperature(self) -> None:
        """
        Non-reasoning identity reports temperature while leaving provider-managed cache unknown.
        """

        adapter = OpenAILLM(
            configuration=LLMConfiguration(
                provider="openai",
                api_key="test-key",
                model="gpt-4.1-mini",
                temperature=0.4,
                use_cache=True,
            )
        )

        self.assertEqual(adapter.identity.settings.temperature, 0.4)
        self.assertIsNone(adapter.identity.settings.cache)
