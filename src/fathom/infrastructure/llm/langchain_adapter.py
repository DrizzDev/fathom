from __future__ import annotations

import asyncio
import base64
import os
import random
import threading
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

from fathom.exceptions import VisionError
from fathom.interfaces import IVisionProvider
from fathom.schemas.configuration import GeminiConfig
from fathom.schemas.results import AnalysisResult
from fathom.services.cache import CacheService
from fathom.services.parsing import ToolResponseParser

logger = getLogger(__name__)


class _FunctionCallShim:
    """
    Shim that wraps a LangChain ``tool_call`` dict so it exposes the same
    ``.name`` / ``.args`` attribute interface that :class:`ToolResponseParser`
    expects from the raw ``google-genai`` SDK objects.
    """

    def __init__(self, tool_call: Dict[str, Any]) -> None:
        self.name: str = tool_call["name"]
        self.args: Dict[str, Any] = tool_call.get("args") or {}


class LangChainLLMClient(IVisionProvider):
    """
    Drop-in replacement for :class:`GeminiLLMClient` that delegates to
    LangChain's ``ChatGoogleGenerativeAI`` or ``ChatVertexAI``.

    Implements :class:`IVisionProvider` so the rest of the Fathom stack
    (``GeminiVisionTool``, ``StepPlanner``, etc.) works without changes.

    Includes full prompt caching parity with the original client via
    :class:`CacheService` backed by the raw ``google-genai`` SDK.
    """

    def __init__(self, configuration: GeminiConfig) -> None:
        self.__configuration = configuration
        self.__parser = ToolResponseParser()
        self.__model: Any = None  # LangChain ChatModel instance
        self.__genai_client: Any = None  # Raw genai.Client for caching
        self.__cache: Optional[CacheService] = None
        self.__credentials: Any = None
        self.__resolved_project: Optional[str] = None

        # Background model readiness — set once __create_model finishes
        self.__model_ready = threading.Event()
        self.__model_init_error: Optional[Exception] = None

        self.__initialize()

    # ── initialisation ─────────────────────────────────────────────────

    def __initialize(self) -> None:
        """
        Build the raw genai.Client (fast, ~150 ms) synchronously, then
        kick off the heavy LangChain model construction in a background
        daemon thread so the caller isn't blocked.
        """

        # ── 1. Fast path: raw genai.Client for caching (~150 ms) ──────
        self.__init_genai_client()

        # ── 2. Slow path: LangChain model — run in background ─────────
        thread = threading.Thread(
            target=self.__create_model_background,
            daemon=True,
            name="fathom-llm-init",
        )
        thread.start()

    def __create_model_background(self) -> None:
        """Create the LangChain ChatModel in a background thread.

        Sets ``__model_ready`` on success or stores the exception in
        ``__model_init_error`` so the first ``analyze()`` call can
        re-raise it on the caller's async context.
        """
        try:
            if self.__configuration.api_key:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self.__model = ChatGoogleGenerativeAI(
                    model=self.__configuration.model,
                    google_api_key=self.__configuration.api_key,
                    temperature=self.__configuration.temperature,
                    max_output_tokens=self.__configuration.max_output_tokens,
                    timeout=self.__configuration.timeout,
                )
            else:
                from langchain_google_vertexai import ChatVertexAI

                self.__model = ChatVertexAI(
                    model_name=self.__configuration.model,
                    project=self.__resolved_project or self.__configuration.project_id,
                    location=self.__configuration.location or "global",
                    temperature=self.__configuration.temperature,
                    max_output_tokens=self.__configuration.max_output_tokens,
                    credentials=self.__credentials,
                )
        except Exception as exc:
            self.__model_init_error = exc
            logger.error("Background LangChain model init failed: %s", exc)
        finally:
            self.__model_ready.set()

    async def _ensure_model_ready(self) -> None:
        """Block (off the event loop) until the background model init finishes."""
        if not self.__model_ready.is_set():
            await asyncio.to_thread(self.__model_ready.wait)
        if self.__model_init_error:
            raise VisionError(
                f"LangChain model init failed: {self.__model_init_error}"
            ) from self.__model_init_error

    def __init_genai_client(self) -> None:
        """
        Create a raw ``google.genai.Client`` for the caching API.

        Mirrors the credential resolution logic in ``GeminiLLMClient``.
        """

        from google import genai
        from google.oauth2 import service_account

        project = self.__configuration.project_id
        location = self.__configuration.location or "global"

        if self.__configuration.credentials_path:
            path = Path(self.__configuration.credentials_path)
            if path.exists():
                self.__credentials = service_account.Credentials.from_service_account_file(
                    str(path),
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                if not project:
                    project = getattr(self.__credentials, "project_id", None)
            else:
                logger.warning(f"Credential file not found at: {path}")

        if not project:
            project = os.environ.get("GEMINI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")

        http_options: Any = {"timeout": self.__configuration.timeout * 1000}  # ms

        self.__resolved_project = project

        if self.__configuration.api_key:
            self.__genai_client = genai.Client(
                http_options=http_options,
                api_key=self.__configuration.api_key,
            )
        else:
            self.__genai_client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                http_options=http_options,
                credentials=self.__credentials,
            )

        self.__cache = CacheService(self.__genai_client, self.__configuration.model)

    # ── public properties ──────────────────────────────────────────────

    @property
    def configuration(self) -> GeminiConfig:
        return self.__configuration

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics (parity with GeminiLLMClient)."""
        return self.__cache.stats.to_dict() if self.__cache else {}

    @property
    def credentials(self) -> Any:
        """Return underlying credentials if available."""
        return self.__credentials

    # ── IVisionProvider.analyze ────────────────────────────────────────

    async def analyze(
        self,
        system_instruction: str,
        user_content: List[Any],
        tools: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Translate the Fathom calling convention into LangChain primitives,
        invoke the model, then re-use the existing :class:`ToolResponseParser`
        to produce an :class:`AnalysisResult`.

        Caching behaviour matches :class:`GeminiLLMClient` exactly:
        - Stable content (system instruction + tools) is cached server-side.
        - On cache hit, the raw ``genai.Client`` is used directly (bypassing
          LangChain) to avoid cache-name mangling by the LangChain wrappers.
        - On cache miss, the LangChain model is used with ``SystemMessage``
          and ``bind_tools``.
        """

        # ── 0. Cache lookup (doesn't need LangChain model) ─────────────
        cache_name: Optional[str] = None
        if self.__cache:
            cache_name = await self.__cache.get_cached_content(
                system_instruction=system_instruction,
                tools=tools.get("function_declarations") if tools else None,
            )

        # ── 1. Dispatch: cached → raw genai SDK (no model needed) ─────
        if cache_name:
            return await self.__invoke_with_genai_cached(
                cache_name=cache_name,
                user_content=user_content,
            )

        # ── 2. Cache miss → need the LangChain model; wait if still loading
        await self._ensure_model_ready()

        return await self.__invoke_with_langchain(
            system_instruction=system_instruction,
            user_content=user_content,
            tools=tools,
        )

    # ── IVisionProvider.cleanup ────────────────────────────────────────

    async def cleanup(self) -> None:
        """Delete the server-side cache on shutdown."""
        if self.__cache:
            await self.__cache.delete_cache()

    # ── private invoke paths ──────────────────────────────────────────

    async def __invoke_with_genai_cached(
        self,
        cache_name: str,
        user_content: List[Any],
    ) -> AnalysisResult:
        """
        Cache-hit path: call the raw ``genai.Client`` directly.

        This mirrors :class:`GeminiLLMClient.analyze` when ``cached_content``
        is active.  We bypass LangChain here because both
        ``ChatGoogleGenerativeAI`` and ``ChatVertexAI`` mangle the cache
        resource name when it is passed as a constructor argument.
        """

        from google.genai import types

        # Build user-content parts (same as GeminiLLMClient)
        parts: List[Any] = []
        for item in user_content:
            if isinstance(item, bytes):
                if not item:
                    raise VisionError("Received empty image data for analysis")
                mime = self.__detect_mime(data=item)
                parts.append(types.Part.from_bytes(data=item, mime_type=mime))
            elif isinstance(item, str):
                parts.append({"text": item})
            else:
                parts.append(item)

        config = types.GenerateContentConfig(
            candidate_count=1,
            temperature=self.__configuration.temperature,
            max_output_tokens=self.__configuration.max_output_tokens,
            automatic_function_calling={"disable": True},
            cached_content=cache_name,
        )

        max_retries = self.__configuration.max_retries
        for attempt in range(max_retries + 1):
            try:
                response = await self.__genai_client.aio.models.generate_content(
                    config=config,
                    model=self.__configuration.model,
                    contents=[types.Content(role="user", parts=parts)],
                )
                result = self.__parser.parse(response)

                # Extract token usage
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    result.metrics["prompt_tokens"] = getattr(usage, "prompt_token_count", 0) or 0
                    result.metrics["completion_tokens"] = (
                        getattr(usage, "candidates_token_count", 0) or 0
                    )
                    result.metrics["cached_tokens"] = (
                        getattr(usage, "cached_content_token_count", 0) or 0
                    )

                return result
            except Exception as exc:
                error_msg = str(exc)
                is_quota = "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg

                if attempt == max_retries:
                    raise VisionError(f"LLM fail: {exc}") from exc

                if is_quota:
                    logger.warning(
                        f"Quota exceeded (429). Pausing 30s before retry "
                        f"{attempt + 1}/{max_retries}…"
                    )
                    jitter = random.random() * 5.0  # nosec
                    delay = 30.0 + jitter
                else:
                    jitter = random.random() * 0.5  # nosec
                    delay = (self.__configuration.retry_delay * (2**attempt)) + jitter

                await asyncio.sleep(delay)

        raise VisionError("Unreachable")  # pragma: no cover

    async def __invoke_with_langchain(
        self,
        system_instruction: str,
        user_content: List[Any],
        tools: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Cache-miss path: use the LangChain model with full system instruction
        and tool bindings.
        """

        from langchain_core.messages import HumanMessage, SystemMessage

        messages: List[Any] = [SystemMessage(content=system_instruction)]

        human_parts: List[Any] = []
        for item in user_content:
            if isinstance(item, bytes):
                if not item:
                    raise VisionError("Received empty image data for analysis")
                mime = self.__detect_mime(data=item)
                b64 = base64.b64encode(item).decode("utf-8")
                human_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                )
            elif isinstance(item, str):
                human_parts.append({"type": "text", "text": item})
            else:
                human_parts.append(item)

        messages.append(HumanMessage(content=human_parts))

        model = self.__model
        if tools:
            lc_tools = self.__convert_tools(tools)
            if lc_tools:
                model = model.bind_tools(lc_tools, tool_choice="any")

        max_retries = self.__configuration.max_retries
        for attempt in range(max_retries + 1):
            try:
                response = await model.ainvoke(messages)
                return self.__parse_langchain_response(response)
            except Exception as exc:
                error_msg = str(exc)
                is_quota = "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg

                if attempt == max_retries:
                    raise VisionError(f"LLM fail: {exc}") from exc

                if is_quota:
                    logger.warning(
                        f"Quota exceeded (429). Pausing 30s before retry "
                        f"{attempt + 1}/{max_retries}…"
                    )
                    jitter = random.random() * 5.0  # nosec
                    delay = 30.0 + jitter
                else:
                    jitter = random.random() * 0.5  # nosec
                    delay = (self.__configuration.retry_delay * (2**attempt)) + jitter

                await asyncio.sleep(delay)

        raise VisionError("Unreachable")  # pragma: no cover

    # ── response parsing helpers ───────────────────────────────────────

    def __parse_langchain_response(self, response: Any) -> AnalysisResult:
        """
        Adapt a LangChain ``AIMessage`` into the shim format that
        :class:`ToolResponseParser.parse` expects.
        """

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            text = getattr(response, "content", "") or ""
            logger.warning(f"No tool calls in LangChain response. Text: {text}")
            return self.__parser.parse(self.__build_shim_response(text=str(text)))

        shim = self.__build_shim_response_from_tool_calls(tool_calls)
        result = self.__parser.parse(shim)

        # Extract token usage from response metadata
        usage = getattr(response, "usage_metadata", None) or {}
        if usage:
            if isinstance(usage, dict):
                result.metrics["prompt_tokens"] = usage.get("input_tokens", 0)
                result.metrics["completion_tokens"] = usage.get("output_tokens", 0)
                result.metrics["cached_tokens"] = usage.get("input_token_details", {}).get(
                    "cache_read", 0
                )
            else:
                result.metrics["prompt_tokens"] = getattr(usage, "input_tokens", 0) or 0
                result.metrics["completion_tokens"] = getattr(usage, "output_tokens", 0) or 0

        return result

    @staticmethod
    def __build_shim_response_from_tool_calls(tool_calls: List[Dict[str, Any]]) -> Any:
        """
        Build a shim object that mimics the Gemini SDK response structure so
        :class:`ToolResponseParser.parse` can process it unchanged.
        """

        class _Part:
            def __init__(self, fc: Any) -> None:
                self.function_call = fc
                self.text = None

        class _Content:
            def __init__(self, parts: List[Any]) -> None:
                self.parts = parts

        class _Candidate:
            def __init__(self, content: Any) -> None:
                self.content = content
                self.finish_reason = "STOP"

        class _Response:
            def __init__(self, candidates: List[Any]) -> None:
                self.candidates = candidates

        parts = [_Part(_FunctionCallShim(tc)) for tc in tool_calls]
        return _Response([_Candidate(_Content(parts))])

    @staticmethod
    def __build_shim_response(text: str) -> Any:
        """
        Build a shim response containing only text (no tool calls).
        """

        class _Part:
            def __init__(self, t: str) -> None:
                self.function_call = None
                self.text = t

        class _Content:
            def __init__(self, parts: List[Any]) -> None:
                self.parts = parts

        class _Candidate:
            def __init__(self, content: Any) -> None:
                self.content = content
                self.finish_reason = "STOP"

        class _Response:
            def __init__(self, candidates: List[Any]) -> None:
                self.candidates = candidates

        return _Response([_Candidate(_Content([_Part(text)]))])

    @staticmethod
    def __convert_tools(tools: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert Gemini-native ``function_declarations`` to the dict schema
        format accepted by LangChain's ``bind_tools``.
        """

        type_map: Dict[str, str] = {
            "STRING": "string",
            "INTEGER": "integer",
            "NUMBER": "number",
            "BOOLEAN": "boolean",
            "ARRAY": "array",
            "OBJECT": "object",
        }

        def _convert_schema(gemini_schema: Dict[str, Any]) -> Dict[str, Any]:
            schema: Dict[str, Any] = {}
            raw_type = gemini_schema.get("type", "STRING")
            schema["type"] = type_map.get(raw_type, raw_type.lower())

            if "description" in gemini_schema:
                schema["description"] = gemini_schema["description"]

            if "enum" in gemini_schema:
                schema["enum"] = gemini_schema["enum"]

            if "properties" in gemini_schema:
                schema["properties"] = {
                    k: _convert_schema(v) for k, v in gemini_schema["properties"].items()
                }

            if "required" in gemini_schema:
                schema["required"] = gemini_schema["required"]

            if "items" in gemini_schema:
                schema["items"] = _convert_schema(gemini_schema["items"])

            return schema

        declarations = tools.get("function_declarations", [])
        lc_tools: List[Dict[str, Any]] = []

        for decl in declarations:
            params = decl.get("parameters", {})
            lc_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": decl["name"],
                        "description": decl.get("description", ""),
                        "parameters": _convert_schema(params),
                    },
                }
            )

        return lc_tools

    @staticmethod
    def __detect_mime(data: bytes) -> str:
        """Detect image MIME type from file signature bytes."""
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
            return "image/gif"
        if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"
