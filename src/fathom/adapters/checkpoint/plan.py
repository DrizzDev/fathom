from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, cast

from langchain_core.runnables import RunnableConfig
from langgraph.constants import START

from fathom.constants.state import IntentStateKey
from fathom.interfaces.plan import PlanStore
from fathom.schemas.plan import Plan
from fathom.schemas.subgoal import GoalState


class _GraphSnapshot(Protocol):
    """
    Minimal view of a compiled-graph state snapshot: its merged channel values.
    """

    values: Dict[str, Any]


class _GraphHandle(Protocol):
    """
    Compiled-graph state read/seed surface the plan store depends on.
    """

    async def aget_state(self, config: RunnableConfig) -> _GraphSnapshot:
        """
        Return the current merged state snapshot for the configured thread.
        """

        ...

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: Dict[str, Any],
        as_node: Optional[str] = None
    ) -> RunnableConfig:
        """
        Commit a state patch attributed to ``as_node`` for the configured thread.
        """

        ...


class LangGraphPlanStore(PlanStore):
    """
    Reads and seeds the accepted plan through a run's LangGraph checkpoint.
    """

    async def read(self, *, run: object, workflow: str) -> Optional[Plan]:
        """
        Return the accepted plan committed for this run, or None on a fresh run.
        """

        graph = cast("_GraphHandle", run)
        snapshot = await graph.aget_state(self.__config(workflow=workflow))
        checkpoint = snapshot.values.get(IntentStateKey.AGENT_STATE_CHECKPOINT.value)

        if not isinstance(checkpoint, dict):
            return None

        goals = checkpoint.get("sub_goals")
        intent = checkpoint.get("intent")
        if not isinstance(goals, list) or not goals or not isinstance(intent, str):
            return None

        cursor_raw = checkpoint.get("current_sub_goal_index", 0)
        cursor = int(cursor_raw) if isinstance(cursor_raw, (int, str)) else 0

        return Plan(
            intent=intent,
            cursor=cursor,
            goals=tuple(GoalState.model_validate(goal).goal for goal in goals),
        )

    async def seed(self, *, run: object, workflow: str, plan: Plan) -> None:
        """
        Commit the accepted plan to the checkpoint before graph entry, attributed to the start node.
        """

        graph = cast("_GraphHandle", run)
        checkpoint: Dict[str, Any] = {
            "intent": plan.intent,
            "current_sub_goal_index": plan.cursor,
            "sub_goals": [
                GoalState(goal=goal).model_dump(mode="json") for goal in plan.goals
            ],
        }

        await graph.aupdate_state(
            self.__config(workflow=workflow),
            {
                IntentStateKey.AGENT_STATE_CHECKPOINT.value: checkpoint,
                IntentStateKey.CURRENT_SUB_GOAL_INDEX.value: plan.cursor,
            },
            as_node=START,
        )

    @staticmethod
    def __config(*, workflow: str) -> RunnableConfig:
        """
        Build the thread-scoped checkpoint config for the given workflow.
        """

        return {"configurable": {"thread_id": workflow}}
