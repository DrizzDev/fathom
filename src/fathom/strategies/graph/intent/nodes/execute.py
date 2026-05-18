from __future__ import annotations

import logging
import time
from typing import Any, Dict, cast

from fathom.constants import ActionType
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.schemas.execution import ExecutionContext
from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider
from fathom.strategies.graph.state import IntentGraphState

logger = logging.getLogger(__name__)


class ExecuteNode:
    """
    EXECUTE graph node; runs the supervised action against the device.
    """

    def __init__(self, *, provider: IntentNodeProvider) -> None:
        """
        Bind the node to the shared intent provider.
        """

        self.__provider = provider

    async def __call__(self, state: IntentGraphState) -> IntentGraphState:
        """
        Run the EXECUTE node handler.
        """

        return await self.run(state=state)

    async def run(self, *, state: IntentGraphState) -> IntentGraphState:
        """
        Invoke the supervisor-approved action against the device.
        """

        logger.info(
            "Starting execution node",
            extra={
                "component": "graph.intent.execute",
                "event": "execute.log",
                "workflow.id": self.__provider.context.workflow_id,
            },
        )

        if await self.__provider.is_cancelled():
            logger.warning(
                "Execution cancelled",
                extra={
                    "component": "graph.intent.execute",
                    "event": "execute.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            self.__provider.context.agent_state.mark_complete(
                reason=CompletionReason.CANCELLED.value
            )
            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        if state.get(IntentStateKey.EXECUTION_BLOCKED):
            logger.info(
                "Skipping execution: supervisor blocked the action",
                extra={
                    "component": "graph.intent.execute",
                    "event": "execute.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            return cast("IntentGraphState", {})

        context = state.get(IntentStateKey.EXECUTION_CONTEXT)
        if not isinstance(context, ExecutionContext):
            logger.error(
                "Missing ExecutionContext; supervise must run first",
                extra={
                    "component": "graph.intent.execute",
                    "event": "execute.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            return cast("IntentGraphState", {})

        step = context.step
        logger.info(
            "Executing action: type=%s, target=%s, confidence=%.2f",
            step.action.action_type.value,
            step.action.target,
            step.action.confidence,
        )

        start_time = time.time()

        if step.action.memory_updates:
            logger.info(
                f"Processing memory updates: {step.action.memory_updates}",
                extra={
                    "component": "graph.intent.execute",
                    "event": "execute.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            for key, value in step.action.memory_updates.items():
                await self.__provider.context.memory.set(key=key, value=str(value))

        if step.action.action_type == ActionType.ASK_USER:
            execution_result = await self.__provider.hitl.ask(step=step, start_time=start_time)
        else:
            logger.info(
                f"Calling action executor for {step.action.action_type.value}",
                extra={
                    "component": "graph.intent.execute",
                    "event": "execute.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            execution_result = await self.__provider.context.action_executor.act(
                step=step,
                pre_capture=context.capture,
                package_name=context.package,
                session_id=self.__provider.context.workflow_id,
            )

        logger.info(
            "Action executed: success=%s, duration=%dms, error=%s",
            execution_result.success,
            execution_result.duration,
            execution_result.error,
        )

        updated_context = context.model_copy(
            update={
                "execution_result": execution_result,
                "duration": int((time.time() - start_time) * 1000),
            }
        )
        result_dict: Dict[Any, Any] = {IntentStateKey.EXECUTION_CONTEXT: updated_context}
        self.__provider.persistence.persist(result=result_dict)

        return result_dict  # type: ignore[return-value]
