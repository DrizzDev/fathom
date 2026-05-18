from __future__ import annotations

import time
from logging import getLogger
from typing import Any, Dict

from fathom.constants.messages import HITL_DEFAULT_PROMPT
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
        await hitl.wait_for_resume()
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

        response = await self.__context.hitl.ask(
            prompt=question,
            step=current_step + 1,
        )
        await self.__context.context_manager.inject_user_guidance(
            guidance=response,
            step=current_step,
        )
        self.__context.agent_state.record_hitl_intervention()

        logger.info(
            "HITL intervention recorded",
            extra={
                **self.__log_context(),
                "event": "hitl.ask.recorded",
            },
        )
        return ExecutionResult(
            success=True,
            duration=int((time.time() - start_time) * 1000),
        )

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
