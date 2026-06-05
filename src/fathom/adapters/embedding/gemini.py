from __future__ import annotations

import asyncio
import time
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fathom.constants.embedding import EmbeddingProvider
from fathom.constants.runtime import MILLISECONDS_PER_SECOND
from fathom.core.exceptions import EmbeddingError
from fathom.interfaces.embedding import EmbeddingPort
from fathom.schemas.embedding import (
    EmbeddingConfiguration,
    EmbeddingResult,
    EmbeddingVector,
)

logger = getLogger(__name__)


class GeminiEmbeddingAdapter(EmbeddingPort):
    """
    Embedding port backed by Gemini's text-embedding endpoint with bounded retry and per-attempt timeout.
    """

    def __init__(
        self,
        *,
        client: Any,
        workflow_id: Optional[str] = None,
        configuration: Optional[EmbeddingConfiguration] = None,
    ) -> None:
        """
        Bind the adapter to a Gemini client, embedding configuration, and run context.
        """

        self.__client = client
        self.__workflow_id = workflow_id
        self.__configuration = (
            configuration if configuration is not None else EmbeddingConfiguration()
        )

    @property
    def configuration(self) -> EmbeddingConfiguration:
        """
        Return the immutable embedding configuration this adapter was bound to.
        """

        return self.__configuration

    async def embed(self, *, texts: Tuple[str, ...]) -> EmbeddingResult:
        """
        Embed every input text via Gemini with retry / timeout enforced.
        """

        if not texts:
            raise EmbeddingError("No texts supplied to embedding adapter")

        sanitized = self.__sanitize(texts=texts)

        attempts = self.__configuration.retry.attempts
        timeout_seconds = self.__configuration.timeout / MILLISECONDS_PER_SECOND

        retrying = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(attempts),
            retry=retry_if_exception_type(asyncio.TimeoutError),
            wait=wait_exponential(
                min=0,
                multiplier=timeout_seconds,
                exp_base=self.__configuration.retry.backoff,
            ),
        )

        started = time.monotonic()
        log_context = self.__log_context(text_count=len(sanitized))

        try:
            async for attempt in retrying:
                with attempt:
                    response = await self.__attempt(
                        texts=sanitized,
                        log_context=log_context,
                        timeout_seconds=timeout_seconds,
                        attempt_index=attempt.retry_state.attempt_number - 1,
                    )
        except asyncio.TimeoutError as exception:
            logger.warning(
                "Embedding call exhausted attempt budget",
                extra={**log_context, "event": "embedding.exhausted"},
            )
            raise EmbeddingError("Embedding call exhausted attempt budget") from exception
        except EmbeddingError:
            raise
        except Exception as exception:
            logger.exception(
                "Embedding call failed with unexpected error",
                extra={**log_context, "event": "embedding.failed"},
            )
            raise EmbeddingError(f"Embedding call failed: {exception}") from exception

        duration = int((time.monotonic() - started) * MILLISECONDS_PER_SECOND)

        return EmbeddingResult(
            vectors=response,
            duration=duration,
            model=self.__configuration.model,
            provider=EmbeddingProvider.GEMINI,
        )

    @staticmethod
    def __sanitize(*, texts: Tuple[str, ...]) -> Tuple[str, ...]:
        """
        Strip and validate each input text; reject empty payloads early.
        """

        cleaned: list[str] = []

        for index, text in enumerate(texts):
            value = (text or "").strip()
            if not value:
                raise EmbeddingError(f"Empty text at index {index} cannot be embedded")

            cleaned.append(value)

        return tuple(cleaned)

    async def __attempt(
        self,
        *,
        attempt_index: int,
        texts: Tuple[str, ...],
        timeout_seconds: float,
        log_context: Dict[str, Any],
    ) -> Tuple[EmbeddingVector, ...]:
        """
        Run one bounded embedding call and project the response into typed :class:`EmbeddingVector` instances.
        """

        logger.info(
            "Embedding call started",
            extra={
                **log_context,
                "event": "embedding.started",
                "attempt.index": attempt_index,
                "timeout.seconds": timeout_seconds,
            },
        )
        try:
            response = await asyncio.wait_for(
                self.__invoke(texts=texts),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Embedding attempt timed out",
                extra={
                    **log_context,
                    "event": "embedding.timeout",
                    "attempt.index": attempt_index,
                    "timeout.seconds": timeout_seconds,
                },
            )
            raise

        return self.__project(response=response)

    async def __invoke(self, *, texts: Tuple[str, ...]) -> Any:
        """
        Run the Gemini embedding call on a worker thread; the SDK exposes
        a synchronous API only so we offload it to avoid blocking the loop.
        """

        return await asyncio.to_thread(
            self.__client.models.embed_content,
            model=self.__configuration.model,
            contents=list(texts),
        )

    @staticmethod
    def __project(*, response: Any) -> Tuple[EmbeddingVector, ...]:
        """
        Validate provider response shape and convert each entry into an :class:`EmbeddingVector`.
        """

        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise EmbeddingError("Gemini embedding response missing 'embeddings' field")

        projected: list[EmbeddingVector] = []

        for index, entry in enumerate(embeddings):
            values = getattr(entry, "values", None)
            if not values:
                raise EmbeddingError(f"Gemini embedding entry {index} missing 'values' field")
            projected.append(EmbeddingVector(values=tuple(float(value) for value in values)))

        return tuple(projected)

    def __log_context(self, *, text_count: int) -> Dict[str, Any]:
        """
        Return shared structured-logging context for this adapter.
        """

        return {
            "text.count": text_count,
            "workflow.id": self.__workflow_id,
            "model": self.__configuration.model,
            "component": "adapter.embedding.gemini",
        }
