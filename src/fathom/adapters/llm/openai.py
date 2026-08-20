from __future__ import annotations

import asyncio
import base64
import json
import random
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import httpx

from fathom.core.exceptions import VisionError
from fathom.interfaces.llm import LLMPort
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import GenerateResult

logger = getLogger(__name__)

OPENAI_BASE_URL = "https://api.openai.com/v1"

REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
"""Models that reject a non-default ``temperature`` and take ``reasoning_effort``
instead. Matched by prefix because the dated aliases (``gpt-5-2025-08-07``)
behave identically to their bare names."""

RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

MAX_RETRY_DELAY_SECONDS = 60.0
RETRY_JITTER_SECONDS = 1.0

MEDIA_RESOLUTION_TO_DETAIL = {"low": "low", "medium": "auto", "high": "high"}
"""``media_resolution`` exists because grounding accuracy on tall mobile
screenshots depends on vision token density; OpenAI spells the same knob
``image_url.detail``. Mapping ``medium`` to ``auto`` rather than to ``high`` is
the honest translation — OpenAI offers two densities and a chooser, not three."""


@dataclass(frozen=True)
class OpenAIToolCall:
    """A tool call, shaped like the one the rest of Fathom already consumes.

    `ToolResponseParser` reads `getattr(call, "name")` and `dict(call.args)` off
    whatever the adapter puts in `GenerateResult.tool_calls`, having been written
    against `google.genai.types.FunctionCall`. Returning OpenAI's own object —
    whose arguments are a JSON *string* under `.function.arguments` — would parse
    as an empty action rather than fail, so the translation happens here.
    """

    name: str
    args: Dict[str, Any] = field(default_factory=dict)


class OpenAILLM(LLMPort):
    """
    OpenAI adapter for LLM interactions.

    Written against the REST API through `httpx` rather than the `openai` SDK,
    which is a dependency decision rather than a stylistic one: `httpx` is
    already a dependency, the surface used here is one endpoint, and adding an
    SDK to run one POST would widen the install for every Fathom consumer.

    Three things do not map cleanly from the Gemini adapter, and each is handled
    explicitly rather than approximated:

    1. **Tool dialect.** Fathom declares tools in Gemini's shape — a
       `function_declarations` list with uppercase JSON-Schema types. They are
       translated, and deliberately *not* marked `strict`: Fathom's schemas have
       genuinely optional fields and strict mode would force all of them required.
    2. **Conversation history carrying tool calls.** OpenAI requires every
       assistant `tool_calls` entry to be answered by a matching `tool` message.
       Replayed history has calls without results, so a call in history is
       rendered as assistant text. See `build_messages`.
    3. **Provider-side prompt caching.** Gemini's is explicit and `prewarm`
       creates the entry. OpenAI's is automatic on long prompts, so `prewarm`
       stays the interface's no-op and cached tokens are *reported* rather than
       requested.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-5-mini",
        configuration: Optional[LLMConfiguration] = None,
    ) -> None:
        """
        Initialize the OpenAI LLM adapter.
        """

        if configuration:
            self.__configuration = configuration
        else:
            self.__configuration = LLMConfiguration(
                provider="openai", api_key=api_key, model=model, use_cache=False
            )

        if not self.__configuration.api_key:
            raise VisionError(
                "OpenAI adapter requires an API key; set OPENAI_API_KEY or pass "
                "api_key on the LLM configuration"
            )

        self.__client: Optional[httpx.AsyncClient] = None

    @property
    def model_name(self) -> str:
        """
        Name of the model being used.
        """

        return self.__configuration.model

    @property
    def __base_url(self) -> str:
        """
        Endpoint root, overridable for Azure and compatible gateways.
        """

        return str(self.__configuration.parameters.get("base_url", OPENAI_BASE_URL)).rstrip("/")

    def __ensure_client(self) -> httpx.AsyncClient:
        """
        Lazily construct the HTTP client so the adapter builds without a loop.
        """

        if self.__client is None:
            self.__client = httpx.AsyncClient(
                timeout=self.__configuration.timeout,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.__configuration.api_key}",
                },
            )

        return self.__client

    # -- request construction ---------------------------------------------

    @staticmethod
    def validate_request(
        *,
        tools: Optional[Dict[str, Any]],
        structured_output: Optional[StructuredOutput],
    ) -> None:
        """
        Refuse request shapes the provider cannot express.
        """

        if structured_output is not None and tools:
            raise VisionError("Structured-output mode cannot be combined with tool calling")

    @classmethod
    def translate_schema(cls, schema: Any) -> Any:
        """
        Rewrite a Gemini-dialect JSON Schema into the one OpenAI accepts.

        Gemini spells its types in uppercase (`"OBJECT"`, `"STRING"`). Everything
        else about the two dialects agrees, so this is a recursive lowercase of
        the `type` field and nothing more — deliberately not a validator, because
        silently repairing a malformed declaration would hide the bug in the
        declaration.
        """

        if isinstance(schema, list):
            return [cls.translate_schema(item) for item in schema]

        if not isinstance(schema, Mapping):
            return schema

        translated: Dict[str, Any] = {}

        for key, value in schema.items():
            if key == "type" and isinstance(value, str):
                translated[key] = value.lower()
            elif key in {"properties", "items", "$defs", "definitions"}:
                translated[key] = cls.translate_schema(value)
            elif key in {"anyOf", "oneOf", "allOf"}:
                translated[key] = cls.translate_schema(value)
            elif isinstance(value, Mapping) and key not in {"enum", "required"}:
                translated[key] = cls.translate_schema(value)
            else:
                translated[key] = value

        return translated

    @classmethod
    def translate_tools(cls, *, tools: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Turn a `function_declarations` payload into OpenAI tool envelopes.
        """

        if not tools:
            return None

        declarations = tools.get("function_declarations") or []

        if not declarations:
            return None

        translated: List[Dict[str, Any]] = []

        for declaration in declarations:
            function: Dict[str, Any] = {"name": declaration["name"]}

            if declaration.get("description"):
                function["description"] = declaration["description"]

            parameters = declaration.get("parameters")

            if parameters:
                function["parameters"] = cls.translate_schema(parameters)

            translated.append({"type": "function", "function": function})

        return translated

    @classmethod
    def structured_response_format(
        cls, *, structured_output: StructuredOutput
    ) -> Dict[str, Any]:
        """
        Build a strict `json_schema` response format from a Pydantic payload.

        Strict mode is the point of using structured output at all — it is what
        makes the response parseable without a repair pass — and OpenAI imposes
        two conditions Pydantic does not emit: every object must forbid extra
        properties, and every declared property must appear in `required`.
        Optional fields stay expressible because Pydantic already renders them as
        a nullable union.
        """

        payload = structured_output.payload
        schema = cls.__strictify(cls.translate_schema(payload.model_json_schema()))

        return {
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "schema": schema,
                "name": payload.__name__,
            },
        }

    @classmethod
    def __strictify(cls, schema: Any) -> Any:
        """
        Apply OpenAI's strict-mode obligations to every object in the schema.
        """

        if isinstance(schema, list):
            return [cls.__strictify(item) for item in schema]

        if not isinstance(schema, Mapping):
            return schema

        strict = {key: cls.__strictify(value) for key, value in schema.items()}

        if strict.get("type") == "object" or "properties" in strict:
            strict["additionalProperties"] = False
            properties = strict.get("properties") or {}
            strict["required"] = list(properties.keys())

        return strict

    @staticmethod
    def detect_mime(*, data: bytes) -> str:
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

    @classmethod
    def __image_part(cls, *, data: bytes, detail: str) -> Dict[str, Any]:
        """
        Wrap image bytes as an OpenAI `image_url` content part.
        """

        if not data:
            raise VisionError("Received empty image data for analysis")

        encoded = base64.b64encode(data).decode("ascii")

        return {
            "type": "image_url",
            "image_url": {
                "detail": detail,
                "url": f"data:{cls.detect_mime(data=data)};base64,{encoded}",
            },
        }

    @classmethod
    def build_messages(
        cls,
        *,
        prompt: Sequence[Union[str, bytes, Dict[str, str]]],
        system_instruction: Optional[str],
        conversation_history: Optional[Sequence[ConversationTurn]],
        detail: str,
    ) -> List[Dict[str, Any]]:
        """
        Assemble the chat messages for one request.

        A tool call replayed from history becomes assistant *text*. OpenAI
        rejects an assistant message carrying `tool_calls` unless each one is
        answered by a `tool` message with a matching id, and replayed history has
        the calls without the results. Rendering the call as text preserves what
        the turn said without fabricating a tool result that never existed — a
        real difference from the Gemini adapter, which round-trips the call.
        """

        messages: List[Dict[str, Any]] = []

        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        for turn in conversation_history or []:
            role = "assistant" if turn.role == "model" else "user"
            parts: List[Dict[str, Any]] = []
            rendered: List[str] = []

            for part in turn.parts:
                if part.function_call is not None:
                    rendered.append(
                        f"[tool call] {part.function_call.name}"
                        f"({json.dumps(part.function_call.args, sort_keys=True)})"
                    )
                elif part.image_data:
                    parts.append(cls.__image_part(data=part.image_data, detail=detail))
                elif part.text:
                    parts.append({"type": "text", "text": part.text})

            if rendered:
                # An assistant turn whose content is a call description. Kept as a
                # plain string rather than a part list: OpenAI accepts either, and
                # a string is what a reader of the transcript expects to see.
                messages.append({"role": role, "content": "\n".join(rendered)})
                continue

            if parts:
                messages.append({"role": role, "content": parts})

        content: List[Dict[str, Any]] = []

        for item in prompt:
            if isinstance(item, bytes):
                content.append(cls.__image_part(data=item, detail=detail))
            elif isinstance(item, str):
                content.append({"type": "text", "text": item})
            elif isinstance(item, Mapping):
                text = item.get("text", "")

                if text:
                    content.append({"type": "text", "text": text})
                elif item.get("data"):
                    raw = item["data"]
                    decoded = base64.b64decode(raw) if isinstance(raw, str) else raw
                    content.append(cls.__image_part(data=decoded, detail=detail))

        messages.append({"role": "user", "content": content})

        return messages

    def request_body(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Assemble the Chat Completions request body.

        Reasoning models take `reasoning_effort` and reject a non-default
        `temperature`; chat models are the other way round. `thinking_level`
        carries straight across, since both providers spell the same four levels.
        """

        body: Dict[str, Any] = {"model": self.model_name, "messages": messages}

        if self.__is_reasoning_model:
            body["reasoning_effort"] = self.__configuration.thinking_level
        else:
            body["temperature"] = self.__configuration.temperature

        if tools:
            body["tools"] = tools
            # Gemini is configured with `function_calling_config.mode = ANY`,
            # which obliges a call. `required` is the same instruction; without
            # it the explorer receives prose where it expects an action.
            body["tool_choice"] = "required"

        if response_format:
            body["response_format"] = response_format

        return body

    @property
    def __is_reasoning_model(self) -> bool:
        return self.model_name.startswith(REASONING_MODEL_PREFIXES)

    # -- response handling -------------------------------------------------

    @classmethod
    def parse_completion(cls, *, payload: Mapping[str, Any]) -> GenerateResult:
        """
        Turn a Chat Completions body into a `GenerateResult`.
        """

        choices = payload.get("choices") or []

        if not choices:
            raise VisionError(
                f"OpenAI returned no choices; finish reason unavailable. Payload keys: "
                f"{sorted(payload.keys())}"
            )

        message = choices[0].get("message") or {}
        tool_calls: List[OpenAIToolCall] = []

        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            name = function.get("name") or ""
            arguments = function.get("arguments") or "{}"

            try:
                parsed = json.loads(arguments)
            except (TypeError, ValueError):
                # One malformed call should not cost the whole turn. The parser
                # falls back to text when nothing survives, which is a better
                # outcome for the caller than an exception from in here.
                logger.warning(
                    "OpenAI tool call arguments were not valid JSON; dropping the call",
                    extra={
                        "tool.name": name,
                        "event": "openai.tool_call.unparsable",
                        "component": "adapters.llm.openai",
                    },
                )
                continue

            if not isinstance(parsed, dict):
                logger.warning(
                    "OpenAI tool call arguments were not an object; dropping the call",
                    extra={
                        "tool.name": name,
                        "event": "openai.tool_call.not_an_object",
                        "component": "adapters.llm.openai",
                    },
                )
                continue

            tool_calls.append(OpenAIToolCall(name=name, args=parsed))

        return GenerateResult(
            # `content` is `null` whenever the model answers with tool calls, and
            # `GenerateResult.content` is declared a `str`.
            content=message.get("content") or "",
            tool_calls=list(tool_calls),
            metrics=cls.__collect_usage_metrics(payload=payload),
        )

    @classmethod
    def __collect_usage_metrics(cls, *, payload: Mapping[str, Any]) -> Dict[str, float]:
        """
        Capture token usage under the names the Gemini adapter already emits.

        Callers read `GenerateResult.metrics` without knowing which provider
        produced it, so the keys are the contract. `thoughts_tokens` carries
        OpenAI's reasoning tokens, which is the same quantity under another name.
        """

        usage = payload.get("usage")

        if not usage:
            return {}

        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}

        metrics = {
            "total_tokens": float(usage.get("total_tokens", 0) or 0),
            "prompt_tokens": float(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": float(usage.get("completion_tokens", 0) or 0),
            "cached_tokens": float(prompt_details.get("cached_tokens", 0) or 0),
            "thoughts_tokens": float(completion_details.get("reasoning_tokens", 0) or 0),
            # No OpenAI equivalent. Reported as zero rather than omitted so the
            # metric set is the same shape whichever adapter produced it.
            "tool_use_prompt_tokens": 0.0,
        }

        logger.info(
            "OpenAI usage recorded",
            extra={
                "component": "adapters.llm.openai",
                "event": "openai.usage.recorded",
                "tokens.total": metrics["total_tokens"],
                "tokens.prompt": metrics["prompt_tokens"],
                "tokens.cached": metrics["cached_tokens"],
                "tokens.thoughts": metrics["thoughts_tokens"],
                "tokens.completion": metrics["completion_tokens"],
            },
        )

        return metrics

    # -- retry -------------------------------------------------------------

    @staticmethod
    def is_retryable(*, status_code: Optional[int]) -> bool:
        """
        Whether a provider status is worth another attempt.

        A 400 is a malformed request and a 401 is a bad key; retrying either
        spends the budget to arrive at the same answer. Rate limits and server
        errors are the ones that change on their own.
        """

        if status_code is None:
            return True

        return status_code in RETRYABLE_STATUS_CODES

    def retry_delay(self, *, attempt: int, retry_after_seconds: Optional[float]) -> float:
        """
        Exponential backoff with jitter, deferring to `Retry-After` when sent.
        """

        jitter = random.random() * RETRY_JITTER_SECONDS  # nosec B311 - not cryptographic

        if retry_after_seconds is not None:
            return min(MAX_RETRY_DELAY_SECONDS, retry_after_seconds + jitter)

        base = self.__configuration.retry_delay * (2**attempt)

        return min(MAX_RETRY_DELAY_SECONDS, base + jitter)

    @staticmethod
    def __retry_after(*, response: Optional[httpx.Response]) -> Optional[float]:
        """
        Read `Retry-After` when the provider sends one.
        """

        if response is None:
            return None

        header = response.headers.get("retry-after")

        if not header:
            return None

        try:
            return float(header)
        except ValueError:
            return None

    # -- generation --------------------------------------------------------

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
        Main handler for LLM interaction.

        `use_cache` is accepted and not acted on: OpenAI's prompt caching is
        automatic above a token threshold and has no create/attach step to
        perform. Cached tokens still arrive in the metrics, so the caller can see
        that caching happened even though it could not ask for it.
        """

        self.validate_request(tools=tools, structured_output=structured_output)

        detail = MEDIA_RESOLUTION_TO_DETAIL.get(
            str(self.__configuration.media_resolution).lower(), "high"
        )
        messages = self.build_messages(
            prompt=prompt,
            detail=detail,
            system_instruction=system_instruction,
            conversation_history=conversation_history,
        )
        body = self.request_body(
            messages=messages,
            tools=self.translate_tools(tools=tools),
            response_format=(
                self.structured_response_format(structured_output=structured_output)
                if structured_output is not None
                else None
            ),
        )

        client = self.__ensure_client()
        max_retries = self.__configuration.max_retries
        url = f"{self.__base_url}/chat/completions"

        for attempt in range(max_retries + 1):
            status_code: Optional[int] = None
            response: Optional[httpx.Response] = None

            try:
                response = await client.post(url, json=body)
                status_code = response.status_code
                response.raise_for_status()

                return self.parse_completion(payload=response.json())

            except asyncio.CancelledError:
                raise

            except VisionError:
                # Already classified and not a transport problem — a response the
                # adapter cannot make sense of will not make more sense on a
                # second attempt.
                raise

            except Exception as exception:
                retryable = self.is_retryable(status_code=status_code)
                detail_text = ""

                if response is not None and status_code is not None and not retryable:
                    # Surface the provider's own message on a terminal error; a
                    # bare "400 Bad Request" from a schema mismatch is otherwise
                    # very expensive to diagnose.
                    detail_text = f" body={response.text[:600]}"

                logger.warning(
                    "OpenAI generation failed",
                    extra={
                        "component": "adapters.llm.openai",
                        "event": "openai.request.failed",
                        "retry.attempt": attempt,
                        "retry.max_attempts": max_retries,
                        "exception.retryable": retryable,
                        "exception.type": type(exception).__name__,
                        "exception.status_code": status_code,
                    },
                )

                if not retryable or attempt == max_retries:
                    raise VisionError(f"LLM fail: {exception}{detail_text}") from exception

                await asyncio.sleep(
                    delay=self.retry_delay(
                        attempt=attempt,
                        retry_after_seconds=self.__retry_after(response=response),
                    )
                )

        raise VisionError("Unreachable")

    async def cleanup(self) -> None:
        """
        Release resources.
        """

        if self.__client is not None:
            await self.__client.aclose()
            self.__client = None
