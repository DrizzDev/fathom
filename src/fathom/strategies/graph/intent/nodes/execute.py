from __future__ import annotations

import logging
import time
from typing import Any, Dict, cast

from fathom.constants import ActionType
from fathom.constants.messages import HITL_UNAVAILABLE_REPLAN_DIAGNOSTIC
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.exceptions import HITLNotAvailableError
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.results import ExecutionResult
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
                "event": "execute.log",
                "component": "graph.intent.execute",
                "workflow.id": self.__provider.context.workflow_id,
            },
        )

        if await self.__provider.is_cancelled():
            logger.warning(
                "Execution cancelled",
                extra={
                    "event": "execute.log",
                    "component": "graph.intent.execute",
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

        context = state.get(IntentStateKey.EXECUTION_CONTEXT)
        if not isinstance(context, ExecutionContext):
            message = "Execution failed: missing ExecutionContext; SUPERVISE did not commit."
            logger.error(
                message,
                extra={
                    "event": "execute.log",
                    "component": "graph.intent.execute",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            self.__provider.context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)
            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                    CommonStateKey.FAILURE_DIAGNOSTIC: message,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        step = context.step
        logger.info(
            "Executing action: target=%s, confidence=%.2f, type=%s",
            step.action.target,
            step.action.confidence,
            step.action.action_type.value,
        )

        start_time = time.time()

        if step.action.memory_updates:
            logger.info(
                f"Processing memory updates: {step.action.memory_updates}",
                extra={
                    "event": "execute.log",
                    "component": "graph.intent.execute",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            for key, value in step.action.memory_updates.items():
                await self.__provider.context.memory.set(key=key, value=str(value))

        if step.action.action_type == ActionType.ASK_USER:
            try:
                execution_result = await self.__provider.hitl.ask(
                    step=step,
                    start_time=start_time,
                )
            except HITLNotAvailableError:
                return self.__route_back_for_replan()
        else:
            observation = state.get(CommonStateKey.SCREEN_OBSERVATION)
            resolved_observation = (
                observation if isinstance(observation, ScreenObservation) else None
            )
            logger.info(
                f"Calling action executor for {step.action.action_type.value}",
                extra={
                    "event": "execute.log",
                    "component": "graph.intent.execute",
                    "action.target": step.action.target,
                    "action.label_id": step.action.label_id,
                    "action.type": step.action.action_type.value,
                    "workflow.id": self.__provider.context.workflow_id,
                    "action.bounds": (
                        step.action.bounds.model_dump() if step.action.bounds else None
                    ),
                },
            )
            execution_result = await self.__provider.context.action_executor.act(
                step=step,
                pre_capture=context.capture,
                package_name=context.package,
                session_id=self.__provider.context.workflow_id,
                observation=resolved_observation,
                is_cancelled=self.__provider.is_cancelled,
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

        diagnostic = self.__swipe_abort_diagnostic(execution_result=execution_result)
        if diagnostic is not None:
            result_dict[IntentStateKey.INJECTED_CONTEXT] = diagnostic

        self.__provider.persistence.persist(result=result_dict)

        return cast("IntentGraphState", result_dict)

    def __route_back_for_replan(self) -> IntentGraphState:
        """Clear the stale ASK_USER step and signal SHOULD_RETRY so the planner re-decides."""

        result = cast(
            "IntentGraphState",
            {
                IntentStateKey.PLAN: None,
                IntentStateKey.PLANNED_STEP: None,
                IntentStateKey.SHOULD_RETRY: True,
                IntentStateKey.EXECUTION_CONTEXT: None,
                CommonStateKey.FAILURE_DIAGNOSTIC: HITL_UNAVAILABLE_REPLAN_DIAGNOSTIC,
            },
        )
        self.__provider.persistence.persist(result=result)
        return result

    @staticmethod
    def __swipe_abort_diagnostic(*, execution_result: ExecutionResult) -> str | None:
        """
        Build one analyzer-facing hint when the swipe coordinator aborted with a typed reason.
        """

        execution = execution_result.swipe_execution
        if execution is None or execution.aborted_for is None or execution.effective:
            return None

        return (
            f"Last swipe aborted ({execution.aborted_for.value}); "
            f"rejected={len(execution.rejections)} attempts={len(execution.attempts)}. "
            "Reconsider gesture origin or dismiss any blocking surface before retrying."
        )
