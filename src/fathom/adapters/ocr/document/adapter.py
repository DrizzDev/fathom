from __future__ import annotations

import asyncio
import importlib
import json
import logging
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

from google.api_core import exceptions as google_exceptions
from google.api_core.client_options import ClientOptions
from google.oauth2 import service_account
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fathom.adapters.ocr.document.mapper import DocumentAiMapper
from fathom.interfaces.ocr import OcrPort
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.ocr import DocumentAiConfiguration, OcrResult
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)

try:
    documentai_v1: Any = importlib.import_module("google.cloud.documentai_v1")
except ImportError:  # pragma: no cover - exercised when optional dependency is absent

    class _MissingDocumentAi:
        """
        Minimal stand-in so the module remains importable without Document AI installed.
        """

        class RawDocument:
            def __init__(self, *, content: bytes, mime_type: str) -> None:
                self.content = content
                self.mime_type = mime_type

        class ProcessRequest:
            def __init__(
                self, *, name: str, raw_document: "_MissingDocumentAi.RawDocument"
            ) -> None:
                self.name = name
                self.raw_document = raw_document

        class DocumentProcessorServiceClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                _ = args, kwargs
                raise ImportError("google-cloud-documentai is required to construct DocumentAiOcr")

    documentai_v1 = _MissingDocumentAi()

try:
    _json_format = importlib.import_module("google.protobuf.json_format")
    MessageToJson: Optional[Callable[..., str]] = cast(
        "Callable[..., str]",
        _json_format.MessageToJson,
    )
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    MessageToJson = None


_TRANSIENT_GOOGLE_EXCEPTIONS = (
    google_exceptions.ServiceUnavailable,
    google_exceptions.DeadlineExceeded,
    google_exceptions.InternalServerError,
    google_exceptions.GatewayTimeout,
    google_exceptions.TooManyRequests,
)


class DocumentAiOcr(OcrPort):
    """
    OCR adapter that delegates the RPC to Google Document AI.
    """

    def __init__(
        self,
        *,
        configuration: DocumentAiConfiguration,
        mapper: Optional[DocumentAiMapper] = None,
        client: Optional[Any] = None,
        workflow_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the adapter with configuration, mapper, optional client, and run context.
        """

        self.__workflow_id = workflow_id
        self.__configuration = configuration
        self.__mapper = mapper if mapper is not None else DocumentAiMapper()
        self.__client = client if client is not None else self.__build_client()

    async def extract(
        self,
        *,
        capture: ScreenCapture,
        budget: PerceptionBudget,
    ) -> OcrResult:
        """
        Run one Document AI pass; degrade to an empty result on any failure.
        """

        started = time.monotonic()
        log_context = self.__log_context(activity=capture.activity)

        logger.info(
            "OCR request started",
            extra={
                **log_context,
                "budget.ocr.ms": budget.ocr,
                "event": "ocr.request.started",
                "image.bytes": len(capture.image),
            },
        )

        try:
            return await self.__extract_unsafe(
                budget=budget,
                capture=capture,
                started=started,
                context=log_context,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "OCR request timed out — degrading to empty result",
                extra={
                    **log_context,
                    "event": "ocr.request.timeout",
                    "budget.ocr.ms": budget.ocr,
                },
            )
        except Exception:
            logger.exception(
                "OCR request failed — degrading to empty result",
                extra={
                    **log_context,
                    "event": "ocr.request.failed",
                    "budget.ocr.ms": budget.ocr,
                },
            )

        return OcrResult(
            tokens=(),
            duration=int((time.monotonic() - started) * 1000),
        )

    async def __extract_unsafe(
        self,
        *,
        started: float,
        capture: ScreenCapture,
        budget: PerceptionBudget,
        context: Dict[str, Any],
    ) -> OcrResult:
        """
        Submit the Document AI request without the outer suppression net.
        """

        timeout = budget.ocr / 1000.0

        document = await asyncio.wait_for(
            asyncio.to_thread(self.__process_document, capture.image),
            timeout=timeout,
        )

        duration = int((time.monotonic() - started) * 1000)
        tokens = self.__mapper.map_document(
            document=document,
            width=capture.width,
            height=capture.height,
        )
        raw_response = self.__serialize_document(document=document)

        logger.info(
            "OCR request completed",
            extra={
                **context,
                "duration.ms": duration,
                "token.count": len(tokens),
                "event": "ocr.request.completed",
            },
        )
        return OcrResult(tokens=tokens, duration=duration, raw_response=raw_response)

    @retry(  # type: ignore[untyped-decorator]
        reraise=True,
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        wait=wait_exponential(multiplier=0.05, min=0.05, max=0.4),
        retry=retry_if_exception_type(_TRANSIENT_GOOGLE_EXCEPTIONS),
    )
    def __process_document(self, image: bytes) -> Any:
        """
        Run the synchronous Document AI process call on the worker thread.

        Transient google-api errors (5xx, DEADLINE_EXCEEDED, 429) are
        retried up to three times with a short exponential backoff
        (50ms..400ms) so a momentary upstream hiccup does not blow the
        whole OCR budget. The outer ``asyncio.wait_for`` still bounds
        total wall time — a saturated upstream that keeps failing past
        the budget surfaces as an empty :class:`OcrResult` (suppressed
        and logged at the adapter boundary; see :meth:`extract`).
        """

        request = documentai_v1.ProcessRequest(
            name=self.__configuration.processor_path,
            raw_document=documentai_v1.RawDocument(content=image, mime_type="image/png"),
        )
        response = self.__client.process_document(request=request)
        return response.document

    @staticmethod
    def __serialize_document(*, document: Any) -> Optional[str]:
        """
        Serialize the provider document for raw OCR debugging artifacts.
        """

        if MessageToJson is not None:
            try:
                message = getattr(document, "_pb", document)
                return MessageToJson(
                    message,
                    preserving_proto_field_name=True,
                    indent=2,
                )
            except Exception:
                logger.debug(
                    "Document AI protobuf JSON serialization failed; falling back to plain JSON",
                    exc_info=True,
                    extra={
                        "component": "adapter.ocr.document",
                        "event": "ocr.raw.serialize.protobuf_failed",
                    },
                )

        try:
            return json.dumps(
                DocumentAiOcr.__plain_value(value=document),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        except Exception:
            logger.debug(
                "Document AI raw response serialization failed",
                exc_info=True,
                extra={
                    "component": "adapter.ocr.document",
                    "event": "ocr.raw.serialize.failed",
                },
            )
            return None

    @staticmethod
    def __plain_value(*, value: Any) -> Any:
        """
        Convert simple test doubles into JSON-serializable structures.
        """

        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, (list, tuple)):
            return [DocumentAiOcr.__plain_value(value=item) for item in value]

        if isinstance(value, dict):
            return {
                str(key): DocumentAiOcr.__plain_value(value=item) for key, item in value.items()
            }

        if hasattr(value, "__dict__"):
            return {
                key: DocumentAiOcr.__plain_value(value=item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }

        return str(value)

    def __build_client(self) -> Any:
        """
        Construct the default Document AI client targeted at the configured location.

        When the configuration carries explicit service-account material
        (inline dict or path), it is materialized into a
        :class:`google.oauth2.service_account.Credentials` and handed to
        the gRPC client. This mirrors the Gemini adapter so Document AI
        authenticates against the same identity instead of falling
        through to ambient ADC / GCE metadata.
        """

        endpoint = f"{self.__configuration.location}-documentai.googleapis.com"
        credentials = self.__resolve_credentials()
        return documentai_v1.DocumentProcessorServiceClient(
            credentials=credentials,
            client_options=ClientOptions(api_endpoint=endpoint),
        )

    def __resolve_credentials(self) -> Optional[service_account.Credentials]:
        """
        Materialize service-account credentials from the configuration.

        Accepts either an inline JSON dict (preferred for server
        deployments where the secret lives in the env) or an absolute
        file path. Returns ``None`` when no material is configured —
        the gRPC transport then raises explicitly rather than silently
        probing GCE metadata.
        """

        material = self.__configuration.credentials
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]

        if isinstance(material, dict):
            return cast(
                "service_account.Credentials",
                service_account.Credentials.from_service_account_info(
                    info=material,
                    scopes=scopes,
                ),
            )

        if isinstance(material, str):
            path = Path(material)
            if not path.is_file():
                logger.warning(
                    "Document AI credentials path does not exist",
                    extra={
                        "component": "adapter.ocr.document",
                        "event": "ocr.credentials.path_missing",
                        "path": str(path),
                    },
                )
                return None
            return cast(
                "service_account.Credentials",
                service_account.Credentials.from_service_account_file(
                    filename=str(path),
                    scopes=scopes,
                ),
            )

        return None

    def __log_context(self, *, activity: str) -> Dict[str, Any]:
        """
        Return shared structured-logging context for this adapter invocation.
        """

        return {
            "component": "adapter.ocr.document",
            "workflow.id": self.__workflow_id,
            "activity": activity,
            "processor": self.__configuration.processor,
        }
