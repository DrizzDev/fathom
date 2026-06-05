from __future__ import annotations

import asyncio
import contextlib
import time
from logging import getLogger
from typing import Any, Dict

from fathom.constants.agent import DirectiveKind
from fathom.constants.messages import HITL_DEFAULT_PROMPT
from fathom.core.exceptions import WorkflowCancelledError
from fathom.core.services.hitl import HITLService
from fathom.schemas.results import ExecutionResult
from fathom.schemas.steps import Step
from fathom.strategies.graph.context import GraphContext

logger = getLogger(__name__)


class Hitl:
    """
    Drives Human-In-The-Loop pause / resume and ASK_USER turns.
    """

    def __init__(self, *, context: GraphContext) -> None:
        """
        Initialize the bridge with the shared graph context.
        """

        self.__context = context

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

        response = await self.__ask_with_cancellation(
            prompt=question,
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
    def __classify_response(*, response: str) -> tuple[DirectiveKind, str | None]:
        """
        Map a raw HITL response into a directive kind plus a target
        descriptor the planner guards can match against.
        """

        normalised = response.strip().lower()

        completion_phrases = (
            "mark the execution as completed",
            "mark as completed",
            "mark as complete",
            "mark execution complete",
            "complete the execution",
            "finish the execution",
            "stop the execution",
            "close the execution",
            "end the execution",
        )
        if any(phrase in normalised for phrase in completion_phrases):
            return DirectiveKind.COMPLETE, response.strip()

        for prefix in ("tap on ", "click on ", "press on ", "tap the ", "click the "):
            if normalised.startswith(prefix):
                descriptor = response.strip()[len(prefix) :].strip()
                descriptor = descriptor.split(",")[0].strip() or descriptor
                return DirectiveKind.RETRY_ACTION, descriptor

        navigation_prefixes = ("go to ", "navigate to ", "open ")
        for prefix in navigation_prefixes:
            if normalised.startswith(prefix):
                descriptor = response.strip()[len(prefix) :].strip()
                return DirectiveKind.NAVIGATE, descriptor

        return DirectiveKind.FREE_FORM, None
