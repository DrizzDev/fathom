from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import datetime, timedelta, timezone
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

from fathom.constants.agent import DirectiveKind
from fathom.constants.messages import HITL_DEFAULT_PROMPT
from fathom.constants.state import CompletionReason
from fathom.conversation.identity import InteractionIdentity
from fathom.core.exceptions import WorkflowCancelledError
from fathom.core.services.hitl import HITLService
from fathom.interfaces.abort import AbortDetectorPort
from fathom.schemas.abort import AbortDecision
from fathom.schemas.recording import Answer, Question
from fathom.schemas.results import ExecutionResult
from fathom.schemas.steps import Step
from fathom.strategies.graph.context import GraphContext

logger = getLogger(__name__)


class Hitl:
    """
    Drives Human-In-The-Loop pause / resume and ASK_USER turns.
    """

    def __init__(
        self,
        *,
        context: GraphContext,
        aborter: AbortDetectorPort,
    ) -> None:
        """
        Bind the bridge to the shared graph context and the operator-abort detector.
        """

        self.__context = context
        self.__aborter = aborter

    def available(self) -> bool:
        """
        Return the intervention authority the bridge would ``ask()`` against; an absent service is unavailable.
        """

        hitl = self.__context.hitl
        return hitl.available if isinstance(hitl, HITLService) else False

    async def prompt(self, *, step: int) -> None:
        """
        Wait for human resume if paused and consume any injected context as guidance.
        """

        hitl = self.__context.hitl
        if not isinstance(hitl, HITLService):
            return

        if not await hitl.is_pause_requested():
            return

        logger.info(
            "HITL pause acknowledged; waiting for resume",
            extra={
                **self.__log_context(),
                "event": "hitl.pause.requested",
                "step.index": step,
            },
        )
        await self.__wait_for_resume_with_cancellation(hitl=hitl)
        await self.__drain_context(hitl=hitl)

    async def ask(self, *, step: Step, start_time: float) -> ExecutionResult:
        """
        Drive a single ASK_USER turn and return a synthetic execution result.
        """

        question = step.action.text or HITL_DEFAULT_PROMPT
        current_step = self.__context.agent_state.step_count

        logger.info(
            "Intercepting ASK_USER action",
            extra={
                **self.__log_context(),
                "event": "hitl.ask.intercept",
                "step.index": current_step + 1,
            },
        )

        question_id = await self.__record_question(
            step=step,
            prompt=question,
            started_at=start_time,
            current_step=current_step,
        )
        response = await self.__ask_with_cancellation(
            prompt=question,
            step=current_step + 1,
        )
        await self.__record_answer(
            step=step,
            response=response,
            question=question_id,
            started_at=start_time,
        )

        decision = await self.__aborter.aborted(response=response)

        if decision.aborted:
            await self.__trigger_workflow_cancellation(
                response=response,
                decision=decision,
                step=current_step + 1,
            )

        await self.__context.context_manager.inject_user_guidance(
            guidance=response,
            step=current_step,
        )
        self.__context.agent_state.record_hitl_intervention()

        kind, target_descriptor = self.__classify_response(response=response)
        directive = self.__context.agent_state.set_operator_directive(
            kind=kind,
            source_text=response,
            target_descriptor=target_descriptor,
        )

        logger.info(
            "HITL intervention recorded",
            extra={
                **self.__log_context(),
                "event": "hitl.ask.recorded",
                "directive.ttl": directive.ttl_turns,
                "directive.kind": directive.kind.value,
                "directive.target": directive.target_descriptor,
            },
        )
        return ExecutionResult(
            success=True,
            duration=int((time.time() - start_time) * 1000),
        )

    async def __record_question(
        self,
        *,
        step: Step,
        prompt: str,
        current_step: int,
        started_at: float,
    ) -> Optional[str]:
        """
        Persist the HITL question as a user-visible conversation message.
        """

        recorder = getattr(self.__context, "recorder", None)
        if recorder is None:
            return None

        identity = InteractionIdentity(execution=self.__context.execution_id)
        task = identity.step_task(
            step_number=step.step_number,
            action_descriptor=step.action.to_description(),
        )
        question = identity.derived_message(
            name="hitl.question",
            qualifier=f"{step.step_number}:{step.action.to_description()}",
        )
        try:
            await recorder.record_hitl_question(
                question=Question(
                    task=task,
                    id=question,
                    tenant=self.__context.tenant,
                    thread=self.__context.thread,
                    actor=self.__context.responder,
                    workspace=self.__context.workspace,
                    workflow=self.__context.workflow_id,
                    execution=self.__context.execution_id,
                    created=datetime.fromtimestamp(started_at, tz=timezone.utc),
                    body={
                        "question": prompt,
                        "step": current_step + 1,
                    },
                    metadata={},
                )
            )
            return question
        except Exception as exception:
            await self.__context.telemetry.warning(
                "Conversation HITL question recording failed",
                step=current_step + 1,
                error=str(exception),
            )
            return None

    async def __record_answer(
        self,
        *,
        step: Step,
        response: str,
        started_at: float,
        question: Optional[str],
    ) -> None:
        """
        Persist the HITL response as a user-visible conversation message.
        """

        recorder = getattr(self.__context, "recorder", None)
        if recorder is None or question is None:
            return

        identity = InteractionIdentity(execution=self.__context.execution_id)
        task = identity.step_task(
            step_number=step.step_number,
            action_descriptor=step.action.to_description(),
        )
        try:
            await recorder.record_hitl_answer(
                answer=Answer(
                    task=task,
                    question=question,
                    tenant=self.__context.tenant,
                    thread=self.__context.thread,
                    actor=self.__context.requester,
                    workspace=self.__context.workspace,
                    workflow=self.__context.workflow_id,
                    execution=self.__context.execution_id,
                    body={"answer": response},
                    id=identity.derived_message(
                        name="hitl.answer",
                        qualifier=f"{step.step_number}:{response}",
                    ),
                    created=datetime.fromtimestamp(
                        started_at,
                        tz=timezone.utc,
                    )
                    + timedelta(milliseconds=1),
                    metadata={},
                )
            )
        except Exception as exception:
            await self.__context.telemetry.warning(
                "Conversation HITL answer recording failed",
                error=str(exception),
                step=step.step_number + 1,
            )

    async def __trigger_workflow_cancellation(
        self,
        *,
        step: int,
        response: str,
        decision: AbortDecision,
    ) -> None:
        """
        Cancel the run when the operator's response carries an abort directive.
        """

        logger.info(
            "HITL abort directive detected; cancelling workflow",
            extra={
                **self.__log_context(),
                "event": "hitl.abort.cancellation_requested",
                "step.index": step,
                "response.preview": response[:120],
                "abort.fallback": decision.fallback,
                "abort.confidence": round(decision.confidence, 4),
            },
        )

        self.__context.cancel()

        logger.info(
            "Workflow cancellation requested from HITL abort directive",
            extra={
                **self.__log_context(),
                "event": "workflow.cancel.requested",
                "step.index": step,
                "source": "hitl_abort",
            },
        )

        raise WorkflowCancelledError(
            workflow_id=self.__context.workflow_id,
            reason=CompletionReason.OPERATOR_ABORTED.value,
        )

    async def __wait_for_resume_with_cancellation(self, *, hitl: HITLService) -> None:
        """
        Wait for pause/resume input while honoring graph-level cancellation.
        """

        resume_task = asyncio.create_task(hitl.wait_for_resume())

        try:
            while not resume_task.done():
                if self.__context.is_cancelled:
                    resume_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await resume_task

                    raise WorkflowCancelledError(workflow_id=self.__context.workflow_id)

                await asyncio.sleep(0.1)

            await resume_task
        finally:
            if not resume_task.done():
                resume_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await resume_task

    async def __ask_with_cancellation(self, *, prompt: str, step: int) -> str:
        """
        Wait for an ASK_USER response while honoring graph-level cancellation.
        """

        ask_task = asyncio.create_task(self.__context.hitl.ask(prompt=prompt, step=step))
        try:
            while not ask_task.done():
                if self.__context.is_cancelled:
                    ask_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ask_task
                    raise WorkflowCancelledError(workflow_id=self.__context.workflow_id)

                await asyncio.sleep(0.1)

            return await ask_task
        finally:
            if not ask_task.done():
                ask_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ask_task

    async def __drain_context(self, *, hitl: HITLService) -> None:
        """
        Drain injected human context after resume and route it through realignment.
        """

        consumed = 0
        while await hitl.has_injected_context():
            context = await hitl.peek_next_context()
            if not context:
                break

            decision = await self.__aborter.aborted(response=context)
            if decision.aborted:
                await hitl.consume_context()
                await self.__trigger_workflow_cancellation(
                    response=context,
                    decision=decision,
                    step=self.__context.agent_state.step_count,
                )

            consumed += 1
            logger.info(
                "HITL injected context applied",
                extra={
                    **self.__log_context(),
                    "event": "hitl.context.injected",
                    "context.index": consumed,
                    "context.preview": context[:80],
                },
            )
            await self.__context.context_manager.inject_user_guidance(
                guidance=context,
                step=self.__context.agent_state.step_count,
            )
            self.__context.agent_state.record_hitl_intervention()
            await hitl.consume_context()

        if consumed > 0:
            logger.info(
                "HITL drained injected contexts",
                extra={
                    **self.__log_context(),
                    "event": "hitl.context.drained",
                    "context.count": consumed,
                },
            )

    def __log_context(self) -> Dict[str, Any]:
        """
        Return shared structured-logging context for HITL entries.
        """

        return {
            "component": "graph.intent.hitl",
            "workflow.id": self.__context.workflow_id,
        }

    @staticmethod
    def __classify_response(*, response: str) -> Tuple[DirectiveKind, Optional[str]]:
        """
        Map a non-abort HITL response into a directive kind plus a target descriptor.
        """

        normalized = response.strip().lower()

        for prefix in ("tap on ", "click on ", "press on ", "tap the ", "click the "):
            if normalized.startswith(prefix):
                descriptor = response.strip()[len(prefix) :].strip()
                descriptor = descriptor.split(",")[0].strip() or descriptor
                return DirectiveKind.RETRY_ACTION, descriptor

        for prefix in ("go to ", "navigate to ", "open "):
            if normalized.startswith(prefix):
                descriptor = response.strip()[len(prefix) :].strip()
                return DirectiveKind.NAVIGATE, descriptor

        return DirectiveKind.FREE_FORM, None
