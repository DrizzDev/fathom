from __future__ import annotations

import asyncio
import base64
import json
import time
from logging import getLogger
from typing import Any, Dict, List, Optional, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field

from fathom.adapters.tools import OpenAIToolDialect
from fathom.constants.llm import ModelProvider
from fathom.core.exceptions import ProviderExhaustedError, VisionError
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.interfaces.tools import CanonicalToolPayload
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.conversation import ConversationTurn, TurnPart
from fathom.schemas.llm import InferenceSettings, ModelIdentity, StructuredOutput
from fathom.schemas.results import GenerateResult

logger = getLogger(__name__)


class OpenAIToolCall(BaseModel):
    """
    Provider-neutral tool-call shape consumed by Fathom's existing parser.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Canonical tool name returned by OpenAI.")
    args: Dict[str, Any] = Field(description="Decoded JSON arguments supplied to the tool.")


class OpenAILLM(LLMPort):
    """
    OpenAI adapter for Fathom's provider-neutral language-model port.

    The SDK import is lazy so Gemini-only installations can still import Fathom. Selecting the
    OpenAI provider produces a clear setup error when the optional dependency is absent.
    """

    def __init__(self, *, configuration: LLMConfiguration) -> None:
        """
        Bind configuration and construct the async OpenAI client.
        """

        self.__configuration = configuration

        try:
            from openai import AsyncOpenAI
        except ImportError as exception:
            raise VisionError(
                "OpenAI support requires the 'openai' package; install Fathom with OpenAI support"
            ) from exception

        if not self.__configuration.api_key:
            raise VisionError("OpenAI initialization failed: missing API key")

        self.__client = AsyncOpenAI(
            api_key=self.__configuration.api_key,
            timeout=self.__configuration.timeout,
            max_retries=0,
        )
        self.__dialect = OpenAIToolDialect()

    @property
    def model_name(self) -> str:
        """
        Name of the OpenAI model being used.
        """

        return self.__configuration.model

    @property
    def identity(self) -> ModelIdentity:
        """
        Declare the configured OpenAI model and comparable inference settings.
        """

        return ModelIdentity(
            provider=ModelProvider.OPENAI,
            model=self.__configuration.model,
            settings=InferenceSettings(
                cache=None,
                # The effective value, not the requested one: None here means the
                # provider's default applied, which is a different run from one that
                # pinned a level, and the manifest has to be able to say which.
                thinking=(
                    self.__configuration.reasoning_effort.value
                    if self.__configuration.reasoning_effort is not None
                    else None
                ),
                temperature=(
                    None if self.__is_reasoning_model else self.__configuration.temperature
                ),
            ),
        )

    def derive(self, *, overrides: LLMConfiguration) -> LLMPort:
        """
        Create a sibling adapter with explicit role overrides and shared API credentials.
        """

        return OpenAILLM(
            configuration=self.__configuration.model_copy(
                update=overrides.model_dump(exclude_unset=True)
            )
        )

    async def generate(
        self,
        *,
        use_cache: bool,
        prompt: Sequence[PromptPart],
        tools: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
        structured_output: Optional[StructuredOutput] = None,
    ) -> GenerateResult:
        """
        Generate one response and normalize OpenAI output into Fathom's result contract.
        """

        _ = use_cache  # OpenAI prompt caching is provider-managed and has no local prewarm call.

        if structured_output is not None and tools:
            raise VisionError("Structured-output mode cannot be combined with tool calling")

        messages = self.__messages(
            prompt=prompt,
            system_instruction=system_instruction,
            conversation_history=conversation_history,
        )
        request: Dict[str, Any] = {
            "model": self.__configuration.model,
            "messages": messages,
        }

        # GPT-5 reasoning models reject the legacy sampling parameter. Keep
        # temperature for older/non-reasoning models, while allowing GPT-5.2
        # (and its dated/role variants) to use the provider default reasoning
        # configuration.
        if not self.__is_reasoning_model:
            request["temperature"] = self.__configuration.temperature

        # Sent only when the experiment pinned one. Omitting the parameter and sending
        # a level are different requests, and some models accept function tools only
        # under an explicit level - so "unset" must stay distinguishable from "none"
        # rather than being collapsed into a default here.
        if self.__configuration.reasoning_effort is not None:
            request["reasoning_effort"] = self.__configuration.reasoning_effort.value

        if tools:
            request.update(self.__dialect.encode(payload=self.__canonical_tools(tools=tools)))
            request["tool_choice"] = "required"

        if structured_output is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": structured_output.payload.__name__.lower(),
                    "strict": True,
                    "schema": self.__openai_schema(
                        schema=structured_output.payload.model_json_schema()
                    ),
                },
            }

        last_exception: Optional[Exception] = None
        for attempt in range(self.__configuration.max_retries + 1):
            started = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    self.__client.chat.completions.create(**request),
                    timeout=self.__configuration.timeout,
                )
                latency = time.perf_counter() - started
                choice = response.choices[0].message
                tool_calls = [
                    self.__normalize_tool_call(call=call) for call in (choice.tool_calls or [])
                ]
                usage = getattr(response, "usage", None)

                return GenerateResult(
                    content=choice.content or "",
                    tool_calls=tool_calls,
                    metrics=self.__usage_metrics(usage=usage, latency=latency),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exception:
                last_exception = exception
                if attempt >= self.__configuration.max_retries:
                    break
                await asyncio.sleep(self.__configuration.retry_delay * (2**attempt))

        if self.__is_rate_limited(exception=last_exception):
            raise ProviderExhaustedError(
                f"OpenAI rate limit after "
                f"{self.__configuration.max_retries + 1} attempts: {last_exception}"
            ) from last_exception

        raise VisionError(f"OpenAI generation failed: {last_exception}") from last_exception

    @staticmethod
    def __is_rate_limited(*, exception: Optional[Exception]) -> bool:
        """
        Whether the provider refused the call for rate or quota reasons.

        Decided from the SDK's exception type and its structured status code, never
        from the message. A run that recovered from a throttle and later failed on its
        own merits mentions 429 in exactly the same words as one the provider shut
        out, so reading the text would make the two indistinguishable.
        """

        if exception is None:
            return False

        try:
            from openai import RateLimitError
        except ImportError:
            return False

        return isinstance(exception, RateLimitError) or (
            getattr(exception, "status_code", None) == 429
        )

    @property
    def __is_reasoning_model(self) -> bool:
        """
        Return whether this model uses OpenAI's reasoning-model request semantics.
        """

        model = self.__configuration.model.lower()
        return model.startswith("gpt-5") or model.startswith(("o1", "o3", "o4"))

    @classmethod
    def __openai_schema(cls, *, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Pydantic JSON Schema into OpenAI strict Structured Outputs schema.

        OpenAI's strict subset does not accept several validation-only Pydantic
        keywords, and every object property must be required. Optional values
        are represented as required nullable properties so the response can
        still be validated by the original Pydantic model.
        """

        unsupported = {
            "default",
            "exclusiveMaximum",
            "exclusiveMinimum",
            "format",
            "maxItems",
            "maxLength",
            "maximum",
            "minItems",
            "minLength",
            "minimum",
            "pattern",
            "title",
            "uniqueItems",
        }

        def transform(value: Any) -> Any:
            if isinstance(value, list):
                return [transform(item) for item in value]
            if not isinstance(value, dict):
                return value

            result = {
                key: transform(item)
                for key, item in value.items()
                if key not in unsupported and key != "const"
            }

            # OpenAI requires a $ref node to contain no sibling keywords,
            # while Pydantic emits descriptions beside references.
            if "$ref" in result:
                return {"$ref": result["$ref"]}

            if "const" in value:
                result["enum"] = [value["const"]]

            properties = result.get("properties")
            if isinstance(properties, dict):
                original_required = set(result.get("required", []))
                required = []
                for name, property_schema in properties.items():
                    required.append(name)
                    if name not in original_required:
                        original_property = value["properties"][name]
                        if original_property.get("default") is None:
                            properties[name] = cls.__nullable_schema(schema=property_schema)
                result["required"] = required
                result["additionalProperties"] = False

            return result

        return cast("Dict[str, Any]", transform(schema))

    @staticmethod
    def __nullable_schema(*, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add nullability without disturbing an existing nullable union.
        """

        if schema.get("type") == "null":
            return schema
        if any(
            item.get("type") == "null" for item in schema.get("anyOf", []) if isinstance(item, dict)
        ):
            return schema
        return {"anyOf": [schema, {"type": "null"}]}

    async def cleanup(self) -> None:
        """
        Close the OpenAI async client transport.
        """

        await self.__client.close()

    @staticmethod
    def __canonical_tools(*, tools: Dict[str, Any]) -> CanonicalToolPayload:
        """
        Accept canonical tools and translate the old Gemini envelope during migration.
        """

        if "tools" in tools:
            return {"tools": list(tools.get("tools", []))}

        return {"tools": list(tools.get("function_declarations", []))}

    @classmethod
    def __messages(
        cls,
        *,
        prompt: Sequence[PromptPart],
        system_instruction: Optional[str],
        conversation_history: Optional[Sequence[ConversationTurn]],
    ) -> List[Dict[str, Any]]:
        """
        Convert Fathom turns and multimodal prompt parts into Chat Completions messages.
        """

        messages: List[Dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        for turn in conversation_history or ():
            messages.append(
                {
                    "role": "assistant" if turn.role == "model" else "user",
                    "content": cls.__parts(parts=turn.parts),
                }
            )

        messages.append({"role": "user", "content": cls.__parts(parts=prompt)})
        return messages

    @classmethod
    def __parts(cls, *, parts: Sequence[Any]) -> Any:
        """
        Encode text and raw image bytes as OpenAI message content parts.
        """

        encoded: List[Dict[str, Any]] = []
        for part in parts:
            if isinstance(part, bytes):
                encoded.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(part).decode('ascii')}"
                        },
                    }
                )
            elif isinstance(part, str):
                encoded.append({"type": "text", "text": part})
            elif isinstance(part, TurnPart):
                if part.text is not None:
                    encoded.append({"type": "text", "text": part.text})
                elif part.image_data is not None:
                    mime = part.mime_type or "image/png"
                    encoded.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{base64.b64encode(part.image_data).decode('ascii')}"
                            },
                        }
                    )
                elif part.function_call is not None:
                    encoded.append(
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "tool": part.function_call.name,
                                    "arguments": part.function_call.args,
                                }
                            ),
                        }
                    )
            elif isinstance(part, dict):
                encoded.append({"type": "text", "text": str(part.get("text", part))})

        return encoded

    @staticmethod
    def __normalize_tool_call(*, call: Any) -> OpenAIToolCall:
        """
        Normalize an OpenAI function call to the attributes expected by ToolResponseParser.
        """

        try:
            arguments = json.loads(call.function.arguments or "{}")
        except (TypeError, ValueError) as exception:
            raise VisionError(f"OpenAI returned invalid tool arguments: {exception}") from exception

        if not isinstance(arguments, dict):
            raise VisionError("OpenAI returned tool arguments that are not a JSON object")

        return OpenAIToolCall(name=call.function.name, args=arguments)

    @staticmethod
    def __usage_metrics(*, usage: Any, latency: float) -> Dict[str, float]:
        """
        Normalize OpenAI usage counters into Fathom metrics.
        """

        prompt_tokens = float(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = float(getattr(usage, "completion_tokens", 0) or 0)
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": float(cached),
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_seconds": latency,
        }
