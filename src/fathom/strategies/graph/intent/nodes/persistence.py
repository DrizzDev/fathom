from __future__ import annotations

from logging import getLogger
from typing import Any, Awaitable, Callable, Dict, Optional, Union, cast

from fathom.constants.events import FathomEvent
from fathom.constants.execution import LAUNCHER_PACKAGES
from fathom.constants.state import IntentStateKey
from fathom.core.agent.state import AgentState
from fathom.schemas.steps import StepResult
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.state import IntentGraphState

logger = getLogger(__name__)


class GraphStatePersistence:
    """
    Round-trips AgentState through the LangGraph state dict for checkpoint resume.
    """

    def __init__(self, *, context: GraphContext) -> None:
        """
        Initialize the persistence helper with the shared graph context.
        """

        self.__context = context

    def restore(self, *, state: IntentGraphState) -> None:
        """
        Restore the live AgentState from a graph-state checkpoint when present.
        """

        checkpoint = state.get(IntentStateKey.AGENT_STATE_CHECKPOINT)
        current_index_value = state.get(IntentStateKey.CURRENT_SUB_GOAL_INDEX, 0)
        current_index = (
            int(current_index_value) if isinstance(current_index_value, (int, str)) else 0
        )

        if checkpoint and isinstance(checkpoint, dict):
            logger.debug(
                "Restoring agent state from graph checkpoint",
                extra={
                    **self.__log_context(),
                    "event": "graph.state.restore",
                    "step.count": checkpoint.get("step_count"),
                    "sub_goal.index": checkpoint.get("current_sub_goal_index"),
                },
            )
            restored = AgentState.from_checkpoint(
                checkpoint,
                capabilities=self.__context.capabilities,
            )
            self.__context.set_agent_state(restored)
            return

        if current_index > 0:
            logger.debug(
                "Restoring sub-goal index from graph state",
                extra={
                    **self.__log_context(),
                    "sub_goal.index": current_index,
                    "event": "graph.state.subgoal_index_restore",
                },
            )
            if self.__context.agent_state.sub_goal_list and current_index < len(
                self.__context.agent_state.sub_goal_list
            ):
                self.__context.agent_state.set_current_sub_goal_index(current_index)

    def persist(self, *, result: Union[IntentGraphState, Dict[str, Any]]) -> None:
        """
        Serialize live AgentState back into the graph state for checkpointing.
        """

        checkpoint = self.__context.agent_state.to_checkpoint()
        current_index = self.__context.agent_state.current_sub_goal_index

        logger.debug(
            "Persisting agent state to graph",
            extra={
                **self.__log_context(),
                "event": "graph.state.persist",
                "sub_goal.index": current_index,
                "step.count": checkpoint.get("step_count"),
            },
        )

        result_dict = cast("Dict[str, Any]", result)
        result_dict[IntentStateKey.AGENT_STATE_CHECKPOINT.value] = checkpoint
        result_dict[IntentStateKey.CURRENT_SUB_GOAL_INDEX.value] = current_index

    @staticmethod
    def should_skip_launcher(*, execution_activity: str, observed_activity: str) -> bool:
        """
        Skip persistence only when the step both starts and ends on the launcher.
        """

        observed_package = observed_activity.split("/")[0]
        execution_package = execution_activity.split("/")[0]

        if execution_package not in LAUNCHER_PACKAGES:
            return False

        return observed_package in LAUNCHER_PACKAGES or observed_package == "unknown"

    def enqueue_history(
        self,
        *,
        step_result: StepResult,
        current_activity: Optional[str],
        execution_activity: Optional[str] = None,
    ) -> None:
        """
        Queue ordered history persistence for the completed step.
        """

        publish = self.__build_publisher(step_number=step_result.step.step_number)
        self.__context.history.enqueue_save_step(
            result=step_result,
            on_complete=publish,
            intent=self.__context.intent,
            package_name=current_activity,
            execution_activity=execution_activity,
        )

    def __build_publisher(self, *, step_number: int) -> Callable[[str], Awaitable[None]]:
        """
        Return an async callback that publishes the generated script telemetry.
        """

        telemetry = self.__context.telemetry

        async def __publish(script_data: str) -> None:
            await telemetry.info(
                script_data,
                step=step_number + 1,
                type=FathomEvent.SCRIPT_GENERATED,
            )

        return __publish

    def __log_context(self) -> Dict[str, Any]:
        """
        Return the shared structured-logging context for persistence entries.
        """

        return {
            "component": "graph.intent.persistence",
            "workflow.id": self.__context.workflow_id,
        }
