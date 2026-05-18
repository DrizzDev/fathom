from __future__ import annotations

import asyncio
import logging
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional

from google.api_core import exceptions as google_exceptions
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1
from google.oauth2 import service_account
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fathom.adapters.ocr.document.mapper import DocumentAiMapper
from fathom.core.exceptions import OcrError
from fathom.interfaces.ocr import OcrPort
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.ocr import DocumentAiConfiguration, OcrResult
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)


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

        self.__configuration = configuration
        self.__mapper = mapper if mapper is not None else DocumentAiMapper()
        self.__client = client if client is not None else self.__build_client()
        self.__workflow_id = workflow_id

    async def extract(
        self,
        *,
        capture: ScreenCapture,
        budget: PerceptionBudget,
    ) -> OcrResult:
        """
        Submit one Document AI process request and return the mapped tokens.
        """

        timeout = budget.ocr / 1000.0
        started = time.monotonic()
        log_context = self.__log_context(activity=capture.activity)

        logger.info(
            "OCR request started",
            extra={
                **log_context,
                "event": "ocr.request.started",
                "budget.ocr.ms": budget.ocr,
                "image.bytes": len(capture.image),
            },
        )

        try:
            document = await asyncio.wait_for(
                asyncio.to_thread(self.__process_document, capture.image),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exception:
            logger.warning(
                "OCR request timed out",
                extra={
                    **log_context,
                    "event": "ocr.request.timeout",
                    "budget.ocr.ms": budget.ocr,
                },
            )
            raise OcrError(
                f"Document AI exceeded the {budget.ocr} ms OCR budget.",
                retryable=True,
            ) from exception
        except OcrError:
            raise
        except Exception as exception:
            logger.warning(
                "OCR request failed",
                extra={
                    **log_context,
                    "event": "ocr.request.failed",
                    "error.message": str(exception),
                },
            )
            raise OcrError(
                f"Document AI request failed: {exception}",
                retryable=False,
            ) from exception

        duration = int((time.monotonic() - started) * 1000)
        tokens = self.__mapper.map_document(
            document=document,
            width=capture.width,
            height=capture.height,
        )

        logger.info(
            "OCR request completed",
            extra={
                **log_context,
                "event": "ocr.request.completed",
                "duration.ms": duration,
                "token.count": len(tokens),
            },
        )
        return OcrResult(tokens=tokens, duration=duration)

    @retry(  # type: ignore[untyped-decorator]
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.05, min=0.05, max=0.4),
        retry=retry_if_exception_type(_TRANSIENT_GOOGLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def __process_document(self, image: bytes) -> Any:
        """
        Run the synchronous Document AI process call on the worker thread.

        Transient google-api errors (5xx, DEADLINE_EXCEEDED, 429) are
        retried up to three times with a short exponential backoff
        (50ms..400ms) so a momentary upstream hiccup does not blow the
        whole OCR budget. The outer ``asyncio.wait_for`` still bounds
        total wall time — a saturated upstream that keeps failing past
        the budget will surface as a single :class:`OcrError`.
        """

        request = documentai_v1.ProcessRequest(
            name=self.__configuration.processor_path,
            raw_document=documentai_v1.RawDocument(content=image, mime_type="image/png"),
        )
        response = self.__client.process_document(request=request)
        return response.document

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
            return service_account.Credentials.from_service_account_info(
                info=material,
                scopes=scopes,
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
            return service_account.Credentials.from_service_account_file(
                filename=str(path),
                scopes=scopes,
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
