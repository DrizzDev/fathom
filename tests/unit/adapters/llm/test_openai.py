from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from fathom.adapters.llm.openai import OpenAILLM, OpenAIToolCall
from fathom.core.exceptions import VisionError
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.conversation import ConversationTurn, TurnPart
from fathom.schemas.llm import StructuredOutput

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff" + b"\x00" * 32

GEMINI_TOOL = {
    "function_declarations": [
        {
            "name": "explore_ui",
            "description": "Tap the next untried element.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "assistant_message": {"type": "STRING", "description": "Reasoning."},
                    "action": {
                        "type": "OBJECT",
                        "properties": {
                            "action_type": {"type": "STRING", "enum": ["tap", "back"]},
                            "tap_target": {
                                "type": "OBJECT",
                                "properties": {
                                    "x": {"type": "INTEGER"},
                                    "y": {"type": "INTEGER"},
                                },
                            },
                            "labels": {"type": "ARRAY", "items": {"type": "STRING"}},
                        },
                        "required": ["action_type"],
                    },
                },
                "required": ["action"],
            },
        }
    ]
}


class Localization(BaseModel):
    """Structured-output payload double."""

    surface: str = Field(description="Surface name")
    confidence: float = Field(description="Confidence")
    note: Optional[str] = Field(default=None, description="Optional note")


def build(**overrides: Any) -> OpenAILLM:
    """An adapter over a configuration that never reaches the network."""

    settings: Dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-5-mini",
        "api_key": "test-key",
    }
    settings.update(overrides)

    return OpenAILLM(configuration=LLMConfiguration(**settings))


def completion(
    *,
    content: Optional[str] = "ok",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A Chat Completions response body."""

    message: Dict[str, Any] = {"role": "assistant", "content": content}

    if tool_calls is not None:
        message["tool_calls"] = tool_calls

    return {
        "model": "gpt-5-mini-2025-08-07",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage
        or {
            "total_tokens": 30,
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 6},
        },
    }


class TestIdentity(unittest.TestCase):
    """The adapter reports the model it was configured with."""

    def test_model_name_is_the_configured_model(self) -> None:
        self.assertEqual(build(model="gpt-5").model_name, "gpt-5")

    def test_a_missing_api_key_is_refused_at_construction(self) -> None:
        with self.assertRaises(VisionError):
            OpenAILLM(configuration=LLMConfiguration(provider="openai", api_key=None))


class TestToolTranslation(unittest.TestCase):
    """Gemini declares tools in its own dialect; OpenAI does not speak it.

    The types are uppercase, the payload is wrapped in `function_declarations`,
    and OpenAI wants each entry under a `function` key with lowercase JSON-Schema
    types. Getting this wrong produces a 400 from the provider rather than a
    quietly wrong answer, but only once a call is actually made.
    """

    def test_declarations_are_wrapped_in_openai_function_envelopes(self) -> None:
        translated = OpenAILLM.translate_tools(tools=GEMINI_TOOL)

        self.assertEqual(len(translated), 1)
        self.assertEqual(translated[0]["type"], "function")
        self.assertEqual(translated[0]["function"]["name"], "explore_ui")
        self.assertEqual(translated[0]["function"]["description"], "Tap the next untried element.")

    def test_schema_types_are_lowercased_recursively(self) -> None:
        parameters = OpenAILLM.translate_tools(tools=GEMINI_TOOL)[0]["function"]["parameters"]

        self.assertEqual(parameters["type"], "object")
        self.assertEqual(parameters["properties"]["action"]["type"], "object")
        action = parameters["properties"]["action"]["properties"]
        self.assertEqual(action["action_type"]["type"], "string")
        self.assertEqual(action["tap_target"]["properties"]["x"]["type"], "integer")
        self.assertEqual(action["labels"]["type"], "array")
        self.assertEqual(action["labels"]["items"]["type"], "string")

    def test_enums_and_required_lists_survive_translation(self) -> None:
        parameters = OpenAILLM.translate_tools(tools=GEMINI_TOOL)[0]["function"]["parameters"]
        action = parameters["properties"]["action"]

        self.assertEqual(action["properties"]["action_type"]["enum"], ["tap", "back"])
        self.assertEqual(action["required"], ["action_type"])
        self.assertEqual(parameters["required"], ["action"])

    def test_tools_are_not_marked_strict(self) -> None:
        """Strict mode would force every declared property to be required.

        Fathom's tool schemas have genuinely optional fields — `text` matters
        only when `action_type` is `type` — so strict mode would change the
        contract rather than enforce it. Structured output is strict; tools are
        not, and the asymmetry is deliberate.
        """
        translated = OpenAILLM.translate_tools(tools=GEMINI_TOOL)[0]["function"]

        self.assertNotIn("strict", translated)

    def test_no_tools_translates_to_none(self) -> None:
        self.assertIsNone(OpenAILLM.translate_tools(tools=None))
        self.assertIsNone(OpenAILLM.translate_tools(tools={}))


class TestStructuredOutput(unittest.TestCase):
    """A Pydantic payload becomes an OpenAI `json_schema` response format."""

    def test_the_payload_becomes_a_strict_json_schema(self) -> None:
        contract = StructuredOutput(payload=Localization)

        response_format = OpenAILLM.structured_response_format(structured_output=contract)

        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(response_format["json_schema"]["name"], "Localization")

    def test_strict_mode_requires_every_property_and_forbids_extras(self) -> None:
        """OpenAI rejects a strict schema whose objects allow extra keys or omit
        a property from `required`. Pydantic emits neither by default."""
        contract = StructuredOutput(payload=Localization)

        schema = OpenAILLM.structured_response_format(structured_output=contract)
        body = schema["json_schema"]["schema"]

        self.assertFalse(body["additionalProperties"])
        self.assertEqual(sorted(body["required"]), ["confidence", "note", "surface"])

    def test_structured_output_and_tools_together_are_refused(self) -> None:
        """The same rule the Gemini adapter enforces, for the same reason: the
        two are different response channels and asking for both is ambiguous."""
        adapter = build()

        with self.assertRaises(VisionError):
            OpenAILLM.validate_request(
                tools=GEMINI_TOOL,
                structured_output=StructuredOutput(payload=Localization),
            )

        self.assertEqual(adapter.model_name, "gpt-5-mini")


class TestMessageConstruction(unittest.TestCase):
    """Prompt parts, images, and history become OpenAI chat messages."""

    def test_a_system_instruction_becomes_the_first_message(self) -> None:
        messages = OpenAILLM.build_messages(
            prompt=["hello"],
            system_instruction="be terse",
            conversation_history=None,
            detail="high",
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "be terse")

    def test_image_bytes_become_a_data_url_with_the_detected_mime_type(self) -> None:
        messages = OpenAILLM.build_messages(
            prompt=["look", PNG], system_instruction=None,
            conversation_history=None, detail="high",
        )

        parts = messages[-1]["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_jpeg_is_detected_separately_from_png(self) -> None:
        messages = OpenAILLM.build_messages(
            prompt=[JPEG], system_instruction=None,
            conversation_history=None, detail="high",
        )

        url = messages[-1]["content"][0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))

    def test_the_image_detail_is_carried_from_media_resolution(self) -> None:
        """`media_resolution` exists because grounding accuracy on tall mobile
        screenshots depends on vision token density. Dropping it on the floor
        would silently degrade exactly what the setting is for."""
        messages = OpenAILLM.build_messages(
            prompt=[PNG], system_instruction=None,
            conversation_history=None, detail="low",
        )

        self.assertEqual(messages[-1]["content"][0]["image_url"]["detail"], "low")

    def test_empty_image_bytes_are_refused(self) -> None:
        with self.assertRaises(VisionError):
            OpenAILLM.build_messages(
                prompt=[b""], system_instruction=None,
                conversation_history=None, detail="high",
            )

    def test_history_maps_the_model_role_onto_assistant(self) -> None:
        history = [
            ConversationTurn(role="user", parts=[TurnPart.from_text("first")]),
            ConversationTurn(role="model", parts=[TurnPart.from_text("reply")]),
        ]

        messages = OpenAILLM.build_messages(
            prompt=["second"], system_instruction=None,
            conversation_history=history, detail="high",
        )

        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])

    def test_a_function_call_in_history_is_rendered_as_assistant_text(self) -> None:
        """A deliberate deviation from Gemini, and it is a protocol constraint
        rather than a shortcut. OpenAI requires every assistant `tool_calls`
        entry to be answered by a matching `tool` message; replaying history that
        carries calls without results is a 400. Rendering the call as text keeps
        the turn's information without inventing a tool result that never existed.
        """
        history = [
            ConversationTurn(
                role="model",
                parts=[TurnPart.from_function_call("explore_ui", {"x": 1})],
            )
        ]

        messages = OpenAILLM.build_messages(
            prompt=["next"], system_instruction=None,
            conversation_history=history, detail="high",
        )

        self.assertEqual(messages[0]["role"], "assistant")
        self.assertIn("explore_ui", messages[0]["content"])
        self.assertNotIn("tool_calls", messages[0])


class TestResponseParsing(unittest.TestCase):
    """The provider's payload becomes a `GenerateResult`."""

    def test_plain_text_content_is_returned(self) -> None:
        result = OpenAILLM.parse_completion(payload=completion(content="a reply"))

        self.assertEqual(result.content, "a reply")
        self.assertEqual(result.tool_calls, [])

    def test_a_null_content_becomes_an_empty_string(self) -> None:
        """OpenAI returns `content: null` alongside tool calls; `GenerateResult`
        declares `content` a `str`."""
        result = OpenAILLM.parse_completion(payload=completion(content=None))

        self.assertEqual(result.content, "")

    def test_tool_calls_expose_name_and_parsed_arguments(self) -> None:
        """`ToolResponseParser` reads `.name` and `dict(.args)`, so the adapter
        owes it objects shaped like Gemini's `FunctionCall`, not raw JSON."""
        payload = completion(
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "explore_ui",
                        "arguments": json.dumps({"action": {"action_type": "tap"}}),
                    },
                }
            ],
        )

        result = OpenAILLM.parse_completion(payload=payload)

        self.assertEqual(len(result.tool_calls), 1)
        call = result.tool_calls[0]
        self.assertEqual(call.name, "explore_ui")
        self.assertEqual(dict(call.args), {"action": {"action_type": "tap"}})

    def test_a_tool_call_with_unparsable_arguments_is_dropped_not_raised(self) -> None:
        """One malformed call should not lose the rest of the turn. The parser
        falls back to text when no call survives, which is a better outcome than
        an exception from inside the adapter."""
        payload = completion(
            content="fallback text",
            tool_calls=[
                {"id": "1", "type": "function",
                 "function": {"name": "explore_ui", "arguments": "{not json"}},
                {"id": "2", "type": "function",
                 "function": {"name": "verify_goal", "arguments": "{}"}},
            ],
        )

        result = OpenAILLM.parse_completion(payload=payload)

        self.assertEqual([c.name for c in result.tool_calls], ["verify_goal"])
        self.assertEqual(result.content, "fallback text")

    def test_usage_metrics_use_the_same_names_the_gemini_adapter_emits(self) -> None:
        """Callers read `GenerateResult.metrics` without knowing the provider, so
        a differently-named key here would silently zero a dashboard."""
        result = OpenAILLM.parse_completion(payload=completion())

        self.assertEqual(result.metrics["total_tokens"], 30.0)
        self.assertEqual(result.metrics["prompt_tokens"], 20.0)
        self.assertEqual(result.metrics["completion_tokens"], 10.0)
        self.assertEqual(result.metrics["cached_tokens"], 4.0)
        self.assertEqual(result.metrics["thoughts_tokens"], 6.0)

    def test_a_response_with_no_choices_is_an_error_not_an_empty_answer(self) -> None:
        with self.assertRaises(VisionError):
            OpenAILLM.parse_completion(payload={"choices": []})


class TestRequestShaping(unittest.TestCase):
    """Reasoning models do not take the same parameters as chat models."""

    def test_a_reasoning_model_carries_effort_and_omits_temperature(self) -> None:
        """`gpt-5` rejects a non-default `temperature`, and Fathom's default is
        1.0 with a configurable override. Sending it is a 400; mapping
        `thinking_level` onto `reasoning_effort` preserves the knob's intent."""
        body = build(model="gpt-5", thinking_level="high", temperature=0.2).request_body(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            response_format=None,
        )

        self.assertNotIn("temperature", body)
        self.assertEqual(body["reasoning_effort"], "high")

    def test_a_chat_model_carries_temperature_and_omits_effort(self) -> None:
        body = build(model="gpt-4o", temperature=0.2).request_body(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            response_format=None,
        )

        self.assertEqual(body["temperature"], 0.2)
        self.assertNotIn("reasoning_effort", body)

    def test_tool_choice_is_required_when_tools_are_supplied(self) -> None:
        """Gemini is called with `function_calling_config.mode = ANY`, which
        obliges the model to call a tool. `tool_choice: required` is the same
        instruction, and without it the explorer gets prose where it expects an
        action."""
        body = build().request_body(
            messages=[{"role": "user", "content": "hi"}],
            tools=OpenAILLM.translate_tools(tools=GEMINI_TOOL),
            response_format=None,
        )

        self.assertEqual(body["tool_choice"], "required")


class TestRetryClassification(unittest.TestCase):
    """Which failures are worth retrying, and for how long."""

    def test_rate_limits_and_server_errors_are_retryable(self) -> None:
        for status in (429, 500, 502, 503, 529):
            self.assertTrue(OpenAILLM.is_retryable(status_code=status), status)

    def test_client_errors_are_not_retryable(self) -> None:
        """A 400 is a malformed request and a 401 is a bad key. Retrying either
        burns the budget to arrive at the same answer."""
        for status in (400, 401, 403, 404, 422):
            self.assertFalse(OpenAILLM.is_retryable(status_code=status), status)

    def test_a_retry_after_header_is_honoured_over_backoff(self) -> None:
        delay = build().retry_delay(attempt=0, retry_after_seconds=12.0)

        self.assertGreaterEqual(delay, 12.0)

    def test_backoff_grows_with_the_attempt(self) -> None:
        adapter = build(retry_delay=1.0)

        first = adapter.retry_delay(attempt=0, retry_after_seconds=None)
        third = adapter.retry_delay(attempt=2, retry_after_seconds=None)

        self.assertGreater(third, first)

    def test_the_delay_is_capped(self) -> None:
        adapter = build(retry_delay=1.0)

        self.assertLessEqual(adapter.retry_delay(attempt=20, retry_after_seconds=None), 60.0)


class TestToolCallValue(unittest.TestCase):
    """The provider-neutral tool call the parser consumes."""

    def test_it_exposes_name_and_args(self) -> None:
        call = OpenAIToolCall(name="explore_ui", args={"a": 1})

        self.assertEqual(call.name, "explore_ui")
        self.assertEqual(dict(call.args), {"a": 1})
