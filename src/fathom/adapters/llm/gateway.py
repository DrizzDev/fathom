from __future__ import annotations

import asyncio
import json
import random
import time
from logging import getLogger
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from fathom.core.exceptions import VisionError
from fathom.interfaces.llm import LLMPort
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.conversation.turns import FunctionCallPart, TurnPart
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import GenerateResult

logger = getLogger(__name__)


class GatewayLLM(LLMPort):
    """
    Adapter for every provider reachable through the LiteLLM gateway.

    One adapter rather than one per vendor: LiteLLM already normalises
    authentication, endpoints and request shape across providers, so the only
    work left is translating between Fathom's neutral types and the
    OpenAI-style contract LiteLLM exposes. Gemini keeps its own native adapter
    because it is the tuned default and uses provider-specific controls
    (thinking level, media resolution) that do not survive the gateway.
    """

    __COMPONENT = "adapters.llm.gateway"

    #: Providers whose native controls are dropped when routed through the
    #: gateway. Logged once per run so a quality change is attributable
    #: rather than invisible.
    __LOSSY_FIELDS = ("thinking_level", "include_thoughts", "media_resolution")

    def __init__(self, *, configuration: LLMConfiguration) -> None:
        self.__configuration = configuration
        self.__client: Optional[Any] = None
        self.__initialize()

    @property
    def model_name(self) -> str:
        """
        The fully qualified model identifier sent to the gateway.
        """

        return self.__qualified_model()

    def derive(self, *, overrides: LLMConfiguration) -> LLMPort:
        """
        Build a sibling adapter bound to a different configuration.
        """

        return GatewayLLM(configuration=overrides)

    def __initialize(self) -> None:
        """
        Import the gateway lazily and warn about controls it cannot carry.

        The import is deferred so that installing Fathom without the
        ``gateway`` extra stays valid for anyone using Gemini alone.
        """

        try:
            import litellm
        except ImportError as failure:
            raise VisionError(
                "Provider "
                f"'{self.__configuration.provider}' requires the gateway extra. "
                "Install fathom[gateway]."
            ) from failure

        # Fathom raises its own errors and owns the retry loop below, so the
        # gateway must not also retry underneath us or a single logical
        # attempt becomes several silent ones.
        litellm.drop_params = True
        litellm.num_retries = 0
        self.__client = litellm

        dropped = [
            field
            for field in self.__LOSSY_FIELDS
            if getattr(self.__configuration, field, None) is not None
        ]
        if dropped:
            logger.info(
                "Gateway cannot carry provider-specific controls",
                extra={
                    "component": self.__COMPONENT,
                    "event": "gateway.controls.dropped",
                    "llm.model": self.__qualified_model(),
                    "llm.dropped": dropped,
                },
            )

    def __qualified_model(self) -> str:
        """
        Return the model prefixed with its provider.

        LiteLLM routes on a ``provider/model`` identifier. A model that already
        carries a prefix is passed through so an explicitly qualified value
        (``bedrock/anthropic.claude-...``) is never double-prefixed.
        """

        model = self.__configuration.model
        if "/" in model:
            return model
        return f"{self.__configuration.provider}/{model}"

    @staticmethod
    def __detect_mime(*, data: bytes) -> str:
        """
        Identify an image payload from its magic bytes.

        Only the formats the capture layer produces are recognised; anything
        else falls back to PNG rather than failing, because a wrong media type
        degrades the request while a raised error loses the whole step.
        """

        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "image/webp"
        return "image/png"

    @classmethod
    def __image_content(cls, *, data: bytes) -> Dict[str, Any]:
        """
        Wrap raw image bytes as an OpenAI-style inline image part.
        """

        import base64

        encoded = base64.b64encode(data).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{cls.__detect_mime(data=data)};base64,{encoded}"},
        }

    @classmethod
    def __to_content(
        cls,
        *,
        prompt: Sequence[Union[str, bytes, Dict[str, str]]],
    ) -> List[Dict[str, Any]]:
        """
        Translate Fathom's prompt parts into OpenAI-style content blocks.
        """

        content: List[Dict[str, Any]] = []
        for item in prompt:
            if isinstance(item, bytes):
                if not item:
                    raise VisionError("Received empty image data for analysis")
                content.append(cls.__image_content(data=item))
            elif isinstance(item, str):
                if item:
                    content.append({"type": "text", "text": item})
            else:
                text = item.get("text", "")
                if text:
                    content.append({"type": "text", "text": text})
        return content

    @classmethod
    def __to_message(cls, *, turn: ConversationTurn) -> Dict[str, Any]:
        """
        Translate one neutral conversation turn into a gateway message.

        A model turn carrying tool calls becomes an assistant message with
        ``tool_calls``; its arguments are re-encoded as JSON because the
        OpenAI contract carries them as a string while Fathom holds them as a
        mapping.
        """

        role = "assistant" if turn.role == "model" else "user"
        content: List[Dict[str, Any]] = []
        calls: List[Dict[str, Any]] = []

        for index, part in enumerate(turn.parts):
            content.extend(cls.__content_for(part=part))
            if part.function_call is not None:
                calls.append(
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(part.function_call.args),
                        },
                    }
                )

        message: Dict[str, Any] = {"role": role, "content": content}
        if calls:
            message["tool_calls"] = calls
        return message

    @classmethod
    def __content_for(cls, *, part: TurnPart) -> List[Dict[str, Any]]:
        """
        Translate a single turn part into zero or more content blocks.
        """

        if part.image_data:
            return [cls.__image_content(data=part.image_data)]
        if part.text:
            return [{"type": "text", "text": part.text}]
        return []

    @staticmethod
    def __to_tools(*, tools: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Translate Fathom's function declarations into the gateway tool schema.

        Declarations are already ``{name, description, parameters}``, so each
        one only needs wrapping in the OpenAI envelope.
        """

        if not tools:
            return None
        declarations = tools.get("function_declarations")
        if not declarations:
            return None
        return [{"type": "function", "function": declaration} for declaration in declarations]

    @staticmethod
    def __to_response_format(
        *,
        structured_output: Optional[StructuredOutput],
    ) -> Optional[Dict[str, Any]]:
        """
        Translate a structured-output contract into a gateway response format.
        """

        if structured_output is None:
            return None
        payload = structured_output.payload
        return {
            "type": "json_schema",
            "json_schema": {
                "name": payload.__name__,
                "schema": payload.model_json_schema(),
                "strict": False,
            },
        }

    @staticmethod
    def __to_tool_calls(*, message: Any) -> List[FunctionCallPart]:
        """
        Translate gateway tool calls into Fathom's neutral representation.

        ``ToolResponseParser`` reads ``.name`` and ``.args`` off each call and
        expects the arguments as a mapping, while the gateway returns them as
        a JSON string. A call whose arguments will not parse is dropped with a
        warning rather than raised, so one malformed call cannot discard the
        rest of a usable response.
        """

        raw = getattr(message, "tool_calls", None) or []
        calls: List[FunctionCallPart] = []

        for call in raw:
            function = getattr(call, "function", None)
            if function is None:
                continue
            name = getattr(function, "name", None)
            if not name:
                continue

            arguments = getattr(function, "arguments", None)
            if isinstance(arguments, Mapping):
                parsed: Dict[str, Any] = dict(arguments)
            else:
                try:
                    decoded = json.loads(arguments or "{}")
                except (TypeError, ValueError):
                    logger.warning(
                        "Gateway returned unparseable tool arguments",
                        extra={
                            "component": GatewayLLM.__COMPONENT,
                            "event": "gateway.tool.unparseable",
                            "tool.name": name,
                        },
                    )
                    continue
                parsed = decoded if isinstance(decoded, dict) else {}

            calls.append(FunctionCallPart(name=name, args=parsed))

        return calls

    @staticmethod
    def __collect_usage_metrics(*, response: Any) -> Dict[str, float]:
        """
        Read token counts off a gateway response.

        Cost is deliberately not reported. The gateway prices from its own
        table, which drifts from the provider invoice — most on image and
        cached tokens, which every step of a run carries.
        """

        usage = getattr(response, "usage", None)
        if usage is None:
            return {}

        metrics: Dict[str, float] = {}
        for source, target in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = getattr(usage, source, None)
            if isinstance(value, (int, float)):
                metrics[target] = float(value)
        return metrics

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
        Run one generation through the gateway with bounded retry.

        ``use_cache`` is accepted for interface parity and ignored: context
        caching is a Gemini-native feature with no gateway equivalent, so
        honouring it here would imply a saving that does not occur.
        """

        if self.__client is None:
            raise VisionError("Gateway client not ready")

        if structured_output is not None and tools:
            raise VisionError("Structured-output mode cannot be combined with tool calling")

        messages: List[Dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if conversation_history:
            messages.extend(self.__to_message(turn=turn) for turn in conversation_history)
        messages.append({"role": "user", "content": self.__to_content(prompt=prompt)})

        request: Dict[str, Any] = {
            "model": self.__qualified_model(),
            "messages": messages,
            "timeout": self.__configuration.timeout,
        }
        if self.__configuration.api_key:
            request["api_key"] = self.__configuration.api_key

        translated = self.__to_tools(tools=tools)
        if translated:
            request["tools"] = translated

        response_format = self.__to_response_format(structured_output=structured_output)
        if response_format:
            request["response_format"] = response_format

        max_retries = self.__configuration.max_retries

        for attempt in range(max_retries + 1):
            started = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    self.__client.acompletion(**request),
                    timeout=self.__configuration.timeout,
                )
            except Exception as failure:
                latency = time.perf_counter() - started
                logger.warning(
                    "Gateway request failed",
                    extra={
                        "component": self.__COMPONENT,
                        "event": "gateway.request.failed",
                        "llm.model": request["model"],
                        "llm.attempt": attempt,
                        "llm.max_retries": max_retries,
                        "latency": latency,
                        "error.type": type(failure).__name__,
                    },
                )
                if attempt >= max_retries:
                    raise VisionError(
                        f"Gateway request failed after {max_retries + 1} attempts: {failure}"
                    ) from failure

                delay = min(2.0**attempt, 8.0) + random.uniform(0.0, 0.5)
                await asyncio.sleep(delay)
                continue

            latency = time.perf_counter() - started
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise VisionError("Gateway returned no choices")

            message = choices[0].message
            content = getattr(message, "content", None) or ""
            calls = self.__to_tool_calls(message=message)

            logger.info(
                "Gateway request completed",
                extra={
                    "component": self.__COMPONENT,
                    "event": "gateway.request.completed",
                    "llm.model": request["model"],
                    "llm.attempt": attempt,
                    "llm.tool_calls": len(calls),
                    "latency": latency,
                },
            )

            return GenerateResult(
                content=content,
                tool_calls=calls,
                metrics=self.__collect_usage_metrics(response=response),
            )

        raise VisionError("Gateway retry loop exhausted without a result")

    async def prewarm(
        self,
        *,
        system_instruction: Optional[str],
        tools: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        No-op.

        Prewarming exists to populate Gemini's context cache. The gateway has
        no equivalent, and issuing a throwaway request to imitate one would
        cost a call and buy nothing.
        """

        return None

    async def cleanup(self) -> None:
        """
        No-op. The gateway holds no client resources of ours to release.
        """

        return None
