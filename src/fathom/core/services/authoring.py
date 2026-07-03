from __future__ import annotations

import time
from logging import getLogger
from typing import Optional, Tuple

from pydantic import ValidationError

from fathom.authoring.agent.packet import AuthoringPacketBuilder
from fathom.authoring.agent.prompts import AuthoringPromptFactory
from fathom.authoring.agent.prompts.base import AuthoringPrompt
from fathom.authoring.agent.reference import AuthoringReferenceProvider
from fathom.constants.authoring import AuthoringArtifactKind, AuthoringKind, AuthoringStatus
from fathom.constants.flow import IssueCode
from fathom.core.dialect.policy import Policy
from fathom.core.exceptions import ConfigurationError, LanguageComplianceError
from fathom.interfaces.authoring import AuthoringPort
from fathom.interfaces.dialect import Dialect
from fathom.interfaces.llm import LLMPort
from fathom.schemas.authoring import AuthoringArtifact, AuthoringResponse, AuthoringTask
from fathom.schemas.authoring.packet import AuthoringPacket
from fathom.schemas.flow import Evidence, Flow, Issue, Report
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
        policy: Policy,
        dialect: Dialect,
        use_cache: bool,
        attempts: int,
        packet_builder: Optional[AuthoringPacketBuilder] = None,
        prompt_factory: Optional[AuthoringPromptFactory] = None,
        reference_provider: Optional[AuthoringReferenceProvider] = None,
    ) -> None:
        """
        Bind model, dialect, policy, packet builder, prompt factory, and dialect reference provider.
        """

        self.__llm = llm
        self.__policy = policy
        self.__dialect = dialect
        self.__use_cache = use_cache
        if attempts < 1:
            raise ConfigurationError("Authoring attempts must be at least 1.")

        self.__attempts = attempts
        self.__packet_builder = packet_builder or AuthoringPacketBuilder()
        self.__prompt_factory = prompt_factory or AuthoringPromptFactory()
        self.__reference_provider = reference_provider or AuthoringReferenceProvider()
        self.__output = StructuredOutput(payload=Flow)

    async def author(self, *, task: AuthoringTask) -> AuthoringResponse:
        """
        Author one task using task-specific prompts and deterministic gates.
        """

        started = time.perf_counter()

        feedback: Tuple[Issue, ...] = ()
        prompt = self.__prompt_factory.prompt(kind=task.kind)
        reference = self.__reference_provider.reference(dialect=task.dialect)

        logger.info(
            "authoring service started",
            extra={
                "event": "authoring.service.started",
                "workflow.id": task.workflow_id,
                "authoring.kind": task.kind.value,
                "authoring.dialect": task.dialect.value,
                "authoring.model": self.__llm.model_name,
            },
        )

        for attempt in range(1, self.__attempts + 1):
            attempt_task = self.__task_with_feedback(task=task, feedback=feedback)
            packet = self.__packet_builder.build(task=attempt_task, dialect=reference)

            try:
                flow = await self.__flow(prompt=prompt, packet=packet, attempt=attempt)
            except Exception as exception:  # noqa: BLE001 - provider failures must fall back cleanly.
                logger.warning(
                    "authoring service failed before review",
                    extra={
                        "event": "authoring.service.failed",
                        "workflow.id": task.workflow_id,
                        "authoring.kind": task.kind.value,
                        "authoring.attempt": attempt,
                        "exception.type": type(exception).__name__,
                        "exception.message": str(exception),
                    },
                )
                return AuthoringResponse(
                    status=AuthoringStatus.FAILED,
                    reason="Authoring failed before deterministic review.",
                )

            issues, text = self.__review(task=task, flow=flow)

            if not issues and text:
                logger.info(
                    "authoring service generated script",
                    extra={
                        "event": "authoring.service.generated",
                        "workflow.id": task.workflow_id,
                        "authoring.kind": task.kind.value,
                        "authoring.attempt": attempt,
                        "authoring.line_count": len(text.splitlines()),
                        "duration.ms": round((time.perf_counter() - started) * 1000, 3),
                    },
                )
                return AuthoringResponse(
                    status=AuthoringStatus.GENERATED,
                    artifact=AuthoringArtifact(
                        dialect=task.dialect,
                        kind=AuthoringArtifactKind.TEXT,
                        content=text,
                    ),
                )

            feedback = issues
            logger.info(
                "authoring service attempt rejected",
                extra={
                    "event": "authoring.service.attempt.rejected",
                    "authoring.attempt": attempt,
                    "workflow.id": task.workflow_id,
                    "authoring.kind": task.kind.value,
                    "authoring.issue_codes": [issue.code.value for issue in issues],
                },
            )

            if not issues:
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
        prompt: AuthoringPrompt,
        packet: AuthoringPacket,
    ) -> Flow:
        """
        Request a structured Flow from the LLM and validate the returned payload.
        """

        system_instruction = prompt.system_instruction()
        user_prompt = prompt.user_prompt(packet=packet)

        logger.info(
            "authoring llm request started",
            extra={
                "event": "authoring.llm.request.started",
                "authoring.attempt": attempt,
                "authoring.model": self.__llm.model_name,
                "authoring.cache_requested": self.__use_cache,
            },
        )

        result = await self.__llm.generate(
            prompt=[user_prompt],
            use_cache=self.__use_cache,
            structured_output=self.__output,
            system_instruction=system_instruction,
        )

        self.__log_llm_result(result=result, attempt=attempt)

        return self.__parse_flow(result=result)

    def __review(self, *, task: AuthoringTask, flow: Flow) -> Tuple[Tuple[Issue, ...], str]:
        """
        Render, check, and policy-review an authored flow.
        """

        issues: Tuple[Issue, ...] = ()
        evidence = self.__evidence(task=task)

        if task.kind is AuthoringKind.RUN and evidence is not None:
            issues = self.__policy.evaluate(flow=flow, evidence=evidence).issues

        try:
            text = self.__dialect.renderer.render(flow=flow)
        except LanguageComplianceError as exception:
            logger.warning(
                "authoring render failed",
                extra={
                    "event": "authoring.render.failed",
                    "workflow.id": task.workflow_id,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            return issues + (self.__render_issue(exception=exception),), ""

        rendered = text.strip()
        syntax = self.__dialect.checker.check(text=rendered)

        return issues + syntax.issues, rendered

    @staticmethod
    def __render_issue(*, exception: LanguageComplianceError) -> Issue:
        """
        Convert an unrenderable flow into deterministic repair feedback.
        """

        return Issue(
            code=IssueCode.UNRENDERABLE_VALUE,
            message=f"Flow could not be rendered: {exception}",
        )

    @staticmethod
    def __evidence(*, task: AuthoringTask) -> Optional[Evidence]:
        """
        Return task evidence when the task view contains normalized execution evidence.
        """

        if task.evidence.run is not None:
            return task.evidence.run.evidence

        if task.evidence.step is not None:
            return task.evidence.step.evidence

        if task.evidence.repair is not None:
            return task.evidence.repair.evidence

        return None

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

    def __log_llm_result(self, *, result: GenerateResult, attempt: int) -> None:
        """
        Record LLM token metrics for the authoring attempt.
        """

        logger.info(
            "authoring llm request completed",
            extra={
                "event": "authoring.llm.request.completed",
                "authoring.attempt": attempt,
                "authoring.model": self.__llm.model_name,
                "authoring.prompt_tokens": result.metrics.get("prompt_tokens"),
                "authoring.cached_tokens": result.metrics.get("cached_tokens"),
                "authoring.output_tokens": result.metrics.get("completion_tokens"),
            },
        )
