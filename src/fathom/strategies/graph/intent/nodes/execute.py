from __future__ import annotations

import time
from datetime import datetime, timezone
from logging import getLogger
from typing import Any, Dict, cast

from fathom.constants import ActionType
from fathom.constants.collaboration import TaskKind
from fathom.constants.messages import HITL_UNAVAILABLE_REPLAN_DIAGNOSTIC
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.conversation.identity import InteractionIdentity
from fathom.core.exceptions import HITLNotAvailableError
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.recording import Step as RecordedStep
from fathom.schemas.results import ExecutionResult
from fathom.schemas.steps import Step
from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider
from fathom.strategies.graph.state import IntentGraphState

logger = getLogger(__name__)


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
                    CommonStateKey.IS_COMPLETE: True,
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.FAILURE_DIAGNOSTIC: message,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        start_time = time.time()

        step = self.__step_with_started_at(step=context.step, started_at=start_time)
        context = context.model_copy(update={"step": step})
        logger.info(
            "Executing action: target=%s, confidence=%.2f, type=%s",
            step.action.target,
            step.action.confidence,
            step.action.action_type.value,
        )

        await self.__record_step_started(step=step, created=start_time)

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
                observation=resolved_observation,
                is_cancelled=self.__provider.is_cancelled,
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

        diagnostic = self.__swipe_abort_diagnostic(execution_result=execution_result)
        if diagnostic is not None:
            result_dict[IntentStateKey.INJECTED_CONTEXT] = diagnostic

        self.__provider.persistence.persist(result=result_dict)

        return cast("IntentGraphState", result_dict)

    @staticmethod
    def __step_with_started_at(*, step: Step, started_at: float) -> Step:
        """
        Return the step with deterministic recorder timing metadata attached.
        """

        metadata = {
            **step.metadata,
            "started_at": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
        }
        return step.model_copy(update={"metadata": metadata})

    async def __record_step_started(self, *, step: Step, created: float) -> None:
        """
        Record the start of one graph action task when recording is enabled.
        """

        context = self.__provider.context
        recorder = getattr(context, "recorder", None)

        tenant = getattr(context, "tenant", None)
        thread = getattr(context, "thread", None)
        responder = getattr(context, "responder", None)

        workspace = getattr(context, "workspace", None)
        workflow_id = getattr(context, "workflow_id", None)
        execution_id = getattr(context, "execution_id", None)

        if recorder is None:
            return

        if (
            not isinstance(tenant, str)
            or not isinstance(thread, str)
            or not isinstance(responder, str)
            or not isinstance(workflow_id, str)
            or not isinstance(execution_id, str)
        ):
            return

        if workspace is not None and not isinstance(workspace, str):
            return

        identity = InteractionIdentity(execution=execution_id)
        try:
            await recorder.record_step_started(
                step=RecordedStep(
                    id=identity.step_task(
                        step_number=step.step_number,
                        action_descriptor=step.action.to_description(),
                    ),
                    tenant=tenant,
                    thread=thread,
                    actor=responder,
                    kind=TaskKind.AGENT,
                    workspace=workspace,
                    root=identity.task(),
                    workflow=workflow_id,
                    parent=identity.task(),
                    reference=step.screen_hash,
                    execution=identity.execution,
                    objective=step.action.to_description(),
                    origin=identity.message(name="request"),
                    created=datetime.fromtimestamp(created, tz=timezone.utc),
                    metadata={
                        "step": step.step_number,
                        "target": step.action.target,
                        "action": step.action.action_type.value,
                    },
                )
            )
        except Exception as exception:
            await context.telemetry.warning(
                "Conversation step start recording failed",
                error=str(exception),
                step=step.step_number + 1,
            )

    def __route_back_for_replan(self) -> IntentGraphState:
        """
        Clear the stale ASK_USER step and signal SHOULD_RETRY so the planner re-decides.
        """

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
