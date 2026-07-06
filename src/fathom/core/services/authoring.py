from __future__ import annotations

import hashlib
import time
from logging import getLogger
from typing import Dict, Tuple

from pydantic import ValidationError

from fathom.authoring.application.request import AuthoringRequest, AuthoringRequestBuilder
from fathom.authoring.application.reviewer import AuthoringReviewer
from fathom.constants.authoring import AuthoringArtifactKind, AuthoringStatus
from fathom.core.exceptions import ConfigurationError, FathomError, LanguageComplianceError
from fathom.interfaces.authoring import AuthoringPort
from fathom.interfaces.llm import LLMPort
from fathom.schemas.authoring import AuthoringArtifact, AuthoringResponse, AuthoringTask
from fathom.schemas.flow import Flow, Issue, Report
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import GenerateResult

logger = getLogger(__name__)


class AuthoringService(AuthoringPort):
    """
    LLM-backed authoring port that produces validated script text from authoring tasks.
    """

    def __init__(
        self,
        *,
        llm: LLMPort,
        attempts: int,
        use_cache: bool,
        reviewer: AuthoringReviewer,
        requests: AuthoringRequestBuilder,
    ) -> None:
        """
        Bind model access, request assembly, deterministic review, and retry settings.
        """

        self.__llm = llm
        self.__requests = requests
        self.__reviewer = reviewer
        self.__use_cache = use_cache

        if attempts < 1:
            raise ConfigurationError("Authoring attempts must be at least 1.")

        self.__attempts = attempts
        self.__output = StructuredOutput(payload=Flow)

    async def author(self, *, task: AuthoringTask) -> AuthoringResponse:
        """
        Author one task using task-specific prompts and deterministic gates.
        """

        started = time.perf_counter()

        feedback: Tuple[Issue, ...] = ()

        logger.info(
            "authoring service started",
            extra={
                "event": "authoring.service.started",
                "execution.id": task.execution_id,
                "authoring.kind": task.kind.value,
                "authoring.dialect": task.dialect.value,
                "authoring.model": self.__llm.model_name,
            },
        )

        for attempt in range(1, self.__attempts + 1):
            attempt_task = self.__task_with_feedback(task=task, feedback=feedback)
            request = self.__requests.build(task=attempt_task)

            try:
                flow = await self.__flow(task=task, request=request, attempt=attempt)
            except FathomError as exception:
                logger.warning(
                    "authoring service failed before review",
                    extra={
                        "event": "authoring.service.failed",
                        "execution.id": task.execution_id,
                        "authoring.attempt": attempt,
                        "authoring.kind": task.kind.value,
                        "exception.type": type(exception).__name__,
                        "exception.message": str(exception),
                    },
                )
                return AuthoringResponse(
                    status=AuthoringStatus.FAILED,
                    reason="Authoring failed before deterministic review.",
                )

            review = self.__reviewer.review(task=task, flow=flow)

            if review.accepted:
                logger.info(
                    "authoring service generated script",
                    extra={
                        "event": "authoring.service.generated",
                        "execution.id": task.execution_id,
                        "authoring.attempt": attempt,
                        "authoring.kind": task.kind.value,
                        "authoring.line_count": len(review.text.splitlines()),
                        "duration.ms": round((time.perf_counter() - started) * 1000, 3),
                    },
                )
                return AuthoringResponse(
                    status=AuthoringStatus.GENERATED,
                    artifact=AuthoringArtifact(
                        content=review.text,
                        dialect=task.dialect,
                        kind=AuthoringArtifactKind.TEXT,
                        advisories=review.advisories,
                        lineage=review.lineage,
                    ),
                )

            feedback = review.issues
            logger.info(
                "authoring service attempt rejected",
                extra={
                    "event": "authoring.service.attempt.rejected",
                    "authoring.attempt": attempt,
                    "execution.id": task.execution_id,
                    "authoring.kind": task.kind.value,
                    "authoring.issue_codes": [issue.code.value for issue in review.issues],
                },
            )

            if not review.issues:
                return AuthoringResponse(
                    status=AuthoringStatus.FAILED,
                    reason="Authoring produced no renderable script.",
                )

        return AuthoringResponse(
            status=AuthoringStatus.FAILED,
            reason="Authoring failed deterministic review after repair attempts.",
        )

    def __task_with_feedback(
        self, *, task: AuthoringTask, feedback: Tuple[Issue, ...]
    ) -> AuthoringTask:
        """
        Return the task with review feedback attached for repair attempts.
        """

        if not feedback:
            return task

        return task.model_copy(update={"review": Report(issues=feedback)})

    async def __flow(
        self,
        *,
        attempt: int,
        task: AuthoringTask,
        request: AuthoringRequest,
    ) -> Flow:
        """
        Request a structured Flow from the LLM and validate the returned payload.
        """

        self.__log_request_payload(task=task, request=request, attempt=attempt)

        result = await self.__llm.generate(
            prompt=request.parts,
            use_cache=self.__use_cache,
            structured_output=self.__output,
            system_instruction=request.instruction,
        )

        self.__log_llm_result(task=task, result=result, attempt=attempt)

        return self.__parse_flow(result=result)

    def __log_request_payload(
        self, *, task: AuthoringTask, request: AuthoringRequest, attempt: int
    ) -> None:
        """
        Record the exact authoring request text and binary artifact fingerprints.
        """

        logger.info(
            "authoring llm request payload",
            extra={
                "event": "authoring.llm.request.payload",
                "execution.id": task.execution_id,
                "authoring.attempt": attempt,
                "authoring.kind": task.kind.value,
                "authoring.step": task.step_number,
                "authoring.dialect": task.dialect.value,
                "authoring.model": self.__llm.model_name,
                "authoring.cache_requested": self.__use_cache,
                "authoring.request.instruction": request.instruction,
                "authoring.request.parts": [
                    self.__part_payload(index=index, part=part)
                    for index, part in enumerate(request.parts)
                ],
            },
        )

    @classmethod
    def __part_payload(cls, *, index: int, part: object) -> Dict[str, object]:
        """
        Return a structured log view of one prompt part.
        """

        if isinstance(part, str):
            return {"index": index, "kind": "text", "content": part}

        if isinstance(part, bytes):
            return {
                "index": index,
                "kind": "bytes",
                "length": len(part),
                "sha256": hashlib.sha256(part, usedforsecurity=False).hexdigest(),
            }

        if isinstance(part, dict):
            return {
                "index": index,
                "kind": "mapping",
                "content": cls.__mapping_payload(part=part),
            }

        return {
            "index": index,
            "kind": type(part).__name__,
            "content": str(part),
        }

    @staticmethod
    def __mapping_payload(*, part: Dict[object, object]) -> Dict[str, object]:
        """
        Return a string-keyed mapping suitable for structured logs.
        """

        return {str(key): value for key, value in part.items()}

    @staticmethod
    def __parse_flow(*, result: GenerateResult) -> Flow:
        """
        Validate the model response as a Flow.
        """

        try:
            return Flow.model_validate_json(result.content)
        except ValidationError as exception:
            raise LanguageComplianceError(
                f"Authoring LLM returned a non-conforming Flow: {exception}"
            ) from exception

    def __log_llm_result(
        self, *, task: AuthoringTask, result: GenerateResult, attempt: int
    ) -> None:
        """
        Record the complete authoring LLM response and token metrics.
        """

        logger.info(
            "authoring llm request completed",
            extra={
                "event": "authoring.llm.request.completed",
                "execution.id": task.execution_id,
                "authoring.attempt": attempt,
                "authoring.kind": task.kind.value,
                "authoring.step": task.step_number,
                "authoring.dialect": task.dialect.value,
                "authoring.model": self.__llm.model_name,
                "authoring.prompt_tokens": result.metrics.get("prompt_tokens"),
                "authoring.cached_tokens": result.metrics.get("cached_tokens"),
                "authoring.output_tokens": result.metrics.get("completion_tokens"),
                "authoring.response.content": result.content,
                "authoring.response.metrics": result.metrics,
                "authoring.response.tool_calls": [
                    self.__tool_call_payload(tool_call=tool_call) for tool_call in result.tool_calls
                ],
            },
        )

    @staticmethod
    def __tool_call_payload(*, tool_call: object) -> Dict[str, object]:
        """
        Return a structured log view of one model tool call.
        """

        if isinstance(tool_call, dict):
            return {str(key): value for key, value in tool_call.items()}

        return {
            "raw": str(tool_call),
            "name": getattr(tool_call, "name", None),
            "args": getattr(tool_call, "args", None),
        }
