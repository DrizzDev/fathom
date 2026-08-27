from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

from fathom.adapters.llm.gateway import GatewayLLM
from fathom.core.exceptions import VisionError
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.conversation.turns import ConversationTurn, TurnPart

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
JPEG = b"\xff\xd8\xff" + b"\x00" * 8


def _response(
    *,
    content: str = "",
    tool_calls: Optional[List[Any]] = None,
    usage: Optional[Any] = None,
) -> SimpleNamespace:
    """
    Build a gateway response double shaped like the OpenAI contract.
    """

    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _call(*, name: str, arguments: Any) -> SimpleNamespace:
    """
    Build a gateway tool-call double.
    """

    return SimpleNamespace(
        id="call_0",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class GatewayAdapterTest(unittest.IsolatedAsyncioTestCase):
    """
    Translation between Fathom's neutral types and the gateway contract.
    """

    def setUp(self) -> None:
        self.configuration = LLMConfiguration(
            provider="anthropic",
            model="claude-sonnet-5",
            max_retries=2,
        )

    def __adapter(self, *, configuration: Optional[LLMConfiguration] = None) -> GatewayLLM:
        """
        Build an adapter with the gateway import stubbed out.
        """

        with patch.dict("sys.modules", {"litellm": SimpleNamespace()}):
            return GatewayLLM(configuration=configuration or self.configuration)

    async def __generate(self, *, adapter: GatewayLLM, outcome: Any, **kwargs: Any) -> Any:
        """
        Run one generation against a recorded gateway double.
        """

        completion = AsyncMock(
            side_effect=outcome if isinstance(outcome, list) else [outcome]
        )
        adapter._GatewayLLM__client = SimpleNamespace(acompletion=completion)  # type: ignore[attr-defined]
        result = await adapter.generate(use_cache=False, **kwargs)
        return result, completion

    # -- model identity ----------------------------------------------------

    def test_model_is_qualified_with_its_provider(self) -> None:
        """LiteLLM routes on a provider/model identifier."""

        self.assertEqual(self.__adapter().model_name, "anthropic/claude-sonnet-5")

    def test_an_already_qualified_model_is_not_double_prefixed(self) -> None:
        """An explicitly routed model must pass through untouched."""

        adapter = self.__adapter(
            configuration=LLMConfiguration(
                provider="anthropic",
                model="bedrock/anthropic.claude-v2",
            )
        )

        self.assertEqual(adapter.model_name, "bedrock/anthropic.claude-v2")

    # -- prompt translation ------------------------------------------------

    async def test_text_and_images_become_content_blocks(self) -> None:
        adapter = self.__adapter()

        _, completion = await self.__generate(
            adapter=adapter,
            outcome=_response(content="ok"),
            prompt=["look at this", PNG, JPEG],
        )

        content = completion.await_args.kwargs["messages"][-1]["content"]

        self.assertEqual(content[0], {"type": "text", "text": "look at this"})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertTrue(content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    async def test_an_empty_image_is_rejected(self) -> None:
        """
        An empty capture is a bug upstream; sending it would earn a confusing
        provider error instead of naming the real problem.
        """

        adapter = self.__adapter()
        adapter._GatewayLLM__client = SimpleNamespace(acompletion=AsyncMock())  # type: ignore[attr-defined]

        with self.assertRaises(VisionError):
            await adapter.generate(use_cache=False, prompt=[b""])

    async def test_system_instruction_leads_the_messages(self) -> None:
        adapter = self.__adapter()

        _, completion = await self.__generate(
            adapter=adapter,
            outcome=_response(content="ok"),
            prompt=["go"],
            system_instruction="be precise",
        )

        messages = completion.await_args.kwargs["messages"]

        self.assertEqual(messages[0], {"role": "system", "content": "be precise"})

    async def test_history_is_translated_with_model_mapped_to_assistant(self) -> None:
        adapter = self.__adapter()
        history = [
            ConversationTurn(role="user", parts=[TurnPart.from_text("first")]),
            ConversationTurn(
                role="model",
                parts=[TurnPart.from_function_call(name="execute_ui", args={"label_id": "4"})],
            ),
        ]

        _, completion = await self.__generate(
            adapter=adapter,
            outcome=_response(content="ok"),
            prompt=["next"],
            conversation_history=history,
        )

        messages = completion.await_args.kwargs["messages"]

        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        # Arguments cross the wire as a JSON string, not a mapping.
        self.assertEqual(
            json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]),
            {"label_id": "4"},
        )

    # -- tool translation --------------------------------------------------

    async def test_declarations_are_wrapped_in_the_tool_envelope(self) -> None:
        adapter = self.__adapter()
        declaration: Dict[str, Any] = {
            "name": "execute_ui",
            "description": "do a thing",
            "parameters": {"type": "object", "properties": {}},
        }

        _, completion = await self.__generate(
            adapter=adapter,
            outcome=_response(content="ok"),
            prompt=["go"],
            tools={"function_declarations": [declaration]},
        )

        self.assertEqual(
            completion.await_args.kwargs["tools"],
            [{"type": "function", "function": declaration}],
        )

    async def test_no_tools_key_when_none_are_declared(self) -> None:
        adapter = self.__adapter()

        _, completion = await self.__generate(
            adapter=adapter,
            outcome=_response(content="ok"),
            prompt=["go"],
            tools={"function_declarations": []},
        )

        self.assertNotIn("tools", completion.await_args.kwargs)

    # -- tool-call translation (the join ToolResponseParser depends on) ----

    async def test_tool_calls_expose_name_and_parsed_args(self) -> None:
        """
        ToolResponseParser reads `.name` and `.args` and expects a mapping,
        while the gateway returns arguments as a JSON string.
        """

        adapter = self.__adapter()

        result, _ = await self.__generate(
            adapter=adapter,
            outcome=_response(
                tool_calls=[_call(name="execute_ui", arguments='{"label_id": "7"}')]
            ),
            prompt=["go"],
        )

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "execute_ui")
        self.assertEqual(result.tool_calls[0].args, {"label_id": "7"})

    async def test_a_mapping_argument_is_accepted_unchanged(self) -> None:
        """Some providers hand back a decoded mapping already."""

        adapter = self.__adapter()

        result, _ = await self.__generate(
            adapter=adapter,
            outcome=_response(tool_calls=[_call(name="execute_ui", arguments={"label_id": "7"})]),
            prompt=["go"],
        )

        self.assertEqual(result.tool_calls[0].args, {"label_id": "7"})

    async def test_one_unparseable_call_does_not_discard_the_others(self) -> None:
        """
        A malformed call is dropped, not raised — losing the whole response
        would turn one bad call into a failed step.
        """

        adapter = self.__adapter()

        result, _ = await self.__generate(
            adapter=adapter,
            outcome=_response(
                tool_calls=[
                    _call(name="broken", arguments="{not json"),
                    _call(name="execute_ui", arguments='{"label_id": "7"}'),
                ]
            ),
            prompt=["go"],
        )

        self.assertEqual([call.name for call in result.tool_calls], ["execute_ui"])

    async def test_a_nameless_call_is_dropped(self) -> None:
        adapter = self.__adapter()

        result, _ = await self.__generate(
            adapter=adapter,
            outcome=_response(tool_calls=[_call(name="", arguments="{}")]),
            prompt=["go"],
        )

        self.assertEqual(result.tool_calls, [])

    # -- structured output -------------------------------------------------

    async def test_structured_output_and_tools_cannot_be_combined(self) -> None:
        from fathom.schemas.llm import StructuredOutput
        from pydantic import BaseModel

        class Payload(BaseModel):
            verdict: str

        adapter = self.__adapter()
        adapter._GatewayLLM__client = SimpleNamespace(acompletion=AsyncMock())  # type: ignore[attr-defined]

        with self.assertRaises(VisionError):
            await adapter.generate(
                use_cache=False,
                prompt=["go"],
                tools={"function_declarations": [{"name": "x"}]},
                structured_output=StructuredOutput(payload=Payload),
            )

    # -- results and metrics -----------------------------------------------

    async def test_usage_is_reported_without_a_cost_figure(self) -> None:
        """
        The gateway prices from its own table, which drifts from the provider
        invoice most on image tokens — which every step carries.
        """

        adapter = self.__adapter()

        result, _ = await self.__generate(
            adapter=adapter,
            outcome=_response(
                content="ok",
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
            ),
            prompt=["go"],
        )

        self.assertEqual(result.metrics["total_tokens"], 14.0)
        self.assertNotIn("cost", result.metrics)

    async def test_a_response_without_choices_is_rejected(self) -> None:
        adapter = self.__adapter()
        adapter._GatewayLLM__client = SimpleNamespace(  # type: ignore[attr-defined]
            acompletion=AsyncMock(return_value=SimpleNamespace(choices=[]))
        )

        with self.assertRaises(VisionError):
            await adapter.generate(use_cache=False, prompt=["go"])

    # -- retry -------------------------------------------------------------

    async def test_a_transient_failure_is_retried_then_succeeds(self) -> None:
        adapter = self.__adapter()

        with patch("asyncio.sleep", new=AsyncMock()):
            result, completion = await self.__generate(
                adapter=adapter,
                outcome=[RuntimeError("boom"), _response(content="recovered")],
                prompt=["go"],
            )

        self.assertEqual(result.content, "recovered")
        self.assertEqual(completion.await_count, 2)

    async def test_retries_are_bounded_by_the_configuration(self) -> None:
        adapter = self.__adapter()
        completion = AsyncMock(side_effect=RuntimeError("boom"))
        adapter._GatewayLLM__client = SimpleNamespace(acompletion=completion)  # type: ignore[attr-defined]

        with patch("asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(VisionError):
                await adapter.generate(use_cache=False, prompt=["go"])

        # max_retries=2 means three attempts in total.
        self.assertEqual(completion.await_count, 3)

    # -- key handling ------------------------------------------------------

    async def test_a_configured_key_is_forwarded(self) -> None:
        adapter = self.__adapter(
            configuration=LLMConfiguration(
                provider="anthropic",
                model="claude-sonnet-5",
                api_key="sk-test",
            )
        )

        _, completion = await self.__generate(
            adapter=adapter, outcome=_response(content="ok"), prompt=["go"]
        )

        self.assertEqual(completion.await_args.kwargs["api_key"], "sk-test")

    async def test_no_key_is_sent_when_none_is_configured(self) -> None:
        """
        Omitting it lets the gateway read the ambient provider environment,
        which is how the Gemini path already gets its credentials.
        """

        adapter = self.__adapter()

        _, completion = await self.__generate(
            adapter=adapter, outcome=_response(content="ok"), prompt=["go"]
        )

        self.assertNotIn("api_key", completion.await_args.kwargs)

    # -- lifecycle ---------------------------------------------------------

    async def test_prewarm_and_cleanup_are_no_ops(self) -> None:
        adapter = self.__adapter()

        self.assertIsNone(await adapter.prewarm(system_instruction="x"))
        self.assertIsNone(await adapter.cleanup())

    def test_derive_returns_a_sibling_bound_to_the_overrides(self) -> None:
        adapter = self.__adapter()

        with patch.dict("sys.modules", {"litellm": SimpleNamespace()}):
            derived = adapter.derive(
                overrides=LLMConfiguration(provider="openai", model="gpt-5.2")
            )

        self.assertIsInstance(derived, GatewayLLM)
        self.assertEqual(derived.model_name, "openai/gpt-5.2")


if __name__ == "__main__":
    unittest.main()
