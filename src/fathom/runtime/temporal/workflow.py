import logging
from collections import deque
from datetime import timedelta
from typing import Any, Deque, Dict, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

from fathom.infrastructure.temporal.state import SignalStateRegistry, WorkflowSignalState
from fathom.schemas.configuration import WorkflowHostPolicyConfiguration
from fathom.schemas.run import ExplorationRunRequest, IntentRunRequest

logger = logging.getLogger(__name__)


class FathomBaseWorkflow:
    """
    Base state and signals for Fathom workflows.
    """

    def __init__(self) -> None:
        self.__paused = False
        self.__cancelled = False
        self.__injected_contexts: Deque[str] = deque()

    def __log(self, message: str, level: int = logging.INFO) -> None:
        """
        Safe logging that works both inside and outside Temporal workflow context.
        """

        try:
            workflow.info()
            workflow.logger.log(level, message)
        except RuntimeError:
            logger.log(level, message)

    def __signal_state(self) -> WorkflowSignalState:
        """
        Resolve the shared signal state mirror for this workflow.
        """

        return SignalStateRegistry.shared().get(workflow_id=workflow.info().workflow_id)

    def timeout(self, *, max_steps: int, policy: WorkflowHostPolicyConfiguration) -> timedelta:
        """
        Compute the activity timeout budget from the requested step count.
        """

        timeout = max(
            policy.timeout_floor,
            (max_steps * policy.timeout_per_step) + policy.timeout_overhead,
        )
        return timedelta(minutes=timeout)

    def heartbeat(self, *, policy: WorkflowHostPolicyConfiguration) -> timedelta:
        """
        Compute the activity heartbeat timeout from the workflow policy.
        """

        return timedelta(seconds=policy.heartbeat_seconds)

    @workflow.signal  # type: ignore[untyped-decorator]
    async def pause(self) -> None:
        """
        Signal to pause execution.
        """

        self.__log(message="Received pause signal")

        self.__paused = True
        self.__signal_state().mark_paused()

    @workflow.signal  # type: ignore[untyped-decorator]
    async def resume(self) -> None:
        """
        Signal to resume execution.
        """

        self.__log(message="Received resume signal")

        self.__paused = False
        self.__signal_state().mark_resumed()

    @workflow.signal  # type: ignore[untyped-decorator]
    async def inject(self, context: str) -> None:
        """
        Signal to inject user context.
        """

        self.__log(message=f"Received inject signal with context: {context}")

        self.__injected_contexts.append(context)
        self.__signal_state().enqueue_context(context=context)

    @workflow.signal  # type: ignore[untyped-decorator]
    async def consume_context(self) -> None:
        """
        Consume the next context from the queue.
        """

        if self.__injected_contexts:
            consumed = self.__injected_contexts.popleft()
            self.__log(message=f"Consumed context: {consumed}")

    @workflow.signal  # type: ignore[untyped-decorator]
    async def cancel(self) -> None:
        """
        Signal to cancel execution.
        """

        self.__log(message="Received cancel signal")

        self.__paused = False
        self.__cancelled = True
        self.__signal_state().mark_cancelled()

    @workflow.query  # type: ignore[untyped-decorator]
    def get_state(self) -> Dict[str, Any]:
        """
        Query current workflow state.
        """

        self.__log(
            message=f"get_state queried: paused={self.__paused} "
            f"cancelled={self.__cancelled} pending_contexts={len(self.__injected_contexts)}",
            level=logging.DEBUG,
        )

        return {
            "paused": self.__paused,
            "cancelled": self.__cancelled,
            "has_context": len(self.__injected_contexts) > 0,
            "pending_contexts": len(self.__injected_contexts),
        }

    @workflow.query  # type: ignore[untyped-decorator]
    def peek_next_context(self) -> Optional[str]:
        """
        Peek at the next context without consuming it.
        """

        if self.__injected_contexts:
            return self.__injected_contexts[0]

        return None

    @workflow.query  # type: ignore[untyped-decorator]
    def get_injected_context(self) -> Optional[str]:
        """
        DEPRECATED: Use peek_next_context + consume_context signal.
        Kept for backward compatibility during transition.
        """

        self.__log(
            message="DEPRECATED get_injected_context query called — migrate to in-process state",
            level=logging.WARNING,
        )

        if not self.__injected_contexts:
            return None

        return self.__injected_contexts[0]


@workflow.defn(name="FathomWorkflow")
class FathomWorkflow(FathomBaseWorkflow):
    """
    Temporal workflow for executing Fathom intent tasks.
    """

    @workflow.run  # type: ignore[untyped-decorator]
    async def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Fathom intent with HITL support.
        """

        try:
            validated_request = IntentRunRequest.model_validate(request)
            runtime = validated_request.runtime.model_copy(
                update={"workflow_id": workflow.info().workflow_id}
            )
            validated_request = validated_request.model_copy(update={"runtime": runtime})

            workflow.logger.info(
                f"Starting Fathom intent workflow for session {validated_request.runtime.session_id} "
                f"with intent: {validated_request.objective.intent}"
            )

            activity_policy = validated_request.interaction.execution_configuration.workflow.intent

            timeout = self.timeout(
                policy=activity_policy,
                max_steps=validated_request.objective.max_steps,
            )
            heartbeat = self.heartbeat(policy=activity_policy)
            arguments = [workflow.info().workflow_id, validated_request.model_dump(mode="json")]

            result = await workflow.execute_activity(
                activity="EXECUTE_INTENT",
                args=arguments,
                heartbeat_timeout=heartbeat,
                start_to_close_timeout=timeout,
                retry_policy=RetryPolicy(maximum_attempts=1),
            )

            workflow.logger.info(
                f"Workflow completed successfully: {result.get('steps')} steps in "
                f"{result.get('duration')}ms"
            )
            return dict(result)

        except Exception as exception:
            workflow.logger.exception(
                "Fathom intent workflow failed",
                exc_info=exception,
                extra={
                    "workflow_id": workflow.info().workflow_id,
                    "error_type": type(exception).__name__,
                    "error": str(exception),
                },
            )
            return {
                "steps": 0,
                "duration": 0,
                "metrics": None,
                "success": False,
                "error": str(exception),
                "error_type": type(exception).__name__,
            }


@workflow.defn(name="FathomExplorationWorkflow")
class FathomExplorationWorkflow(FathomBaseWorkflow):
    """
    Temporal workflow for executing Fathom autonomous exploration.
    """

    @workflow.run  # type: ignore[untyped-decorator]
    async def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Fathom exploration with HITL support.
        """

        try:
            validated_request = ExplorationRunRequest.model_validate(request)
            runtime = validated_request.runtime.model_copy(
                update={"workflow_id": workflow.info().workflow_id}
            )
            validated_request = validated_request.model_copy(update={"runtime": runtime})

            workflow.logger.info(
                f"Starting Fathom exploration for session {validated_request.runtime.session_id}"
            )

            activity_policy = (
                validated_request.interaction.execution_configuration.workflow.exploration
            )
            timeout = self.timeout(
                policy=activity_policy,
                max_steps=validated_request.objective.max_steps,
            )
            heartbeat = self.heartbeat(policy=activity_policy)
            arguments = [workflow.info().workflow_id, validated_request.model_dump(mode="json")]

            result = await workflow.execute_activity(
                activity="EXECUTE_EXPLORATION",
                args=arguments,
                heartbeat_timeout=heartbeat,
                start_to_close_timeout=timeout,
                retry_policy=RetryPolicy(maximum_attempts=1),
            )

            workflow.logger.info(f"Exploration completed: {result.get('steps')} steps")
            return dict(result)

        except Exception as exception:
            workflow.logger.exception(
                "Fathom exploration workflow failed",
                exc_info=exception,
                extra={
                    "workflow_id": workflow.info().workflow_id,
                    "error_type": type(exception).__name__,
                    "error": str(exception),
                },
            )
            return {
                "steps": 0,
                "duration": 0,
                "metrics": None,
                "success": False,
                "error": str(exception),
                "error_type": type(exception).__name__,
            }
