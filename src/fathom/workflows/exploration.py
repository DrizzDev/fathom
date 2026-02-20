from __future__ import annotations

import asyncio
import contextlib
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from fathom.graph.exploration_nodes import ExplorationNodeContext

from fathom.exceptions import FathomError
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.interfaces import IMemoryProvider
from fathom.schemas.configuration import WorkflowConfig
from fathom.schemas.results import ExplorationResult
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool
from fathom.workflows.base import BaseWorkflow

logger = getLogger(__name__)


class ExplorationWorkflow(BaseWorkflow[ExplorationResult]):
    """
    Workflow for systematic application mapping.

    Executes a LangGraph StateGraph that drives DFS exploration
    with full instrumentation (audit, history, tracing, metrics).
    """

    def __init__(
        self,
        workflow_id: str,
        device: DeviceTool,
        capture: CaptureTool,
        vision: Optional[VisionTool] = None,
        *,
        seed: Optional[int] = None,
        memory: Optional[IMemoryProvider] = None,
        configuration: Optional[WorkflowConfig] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        target_package: Optional[str] = None,
    ) -> None:
        """
        Initialize exploration workflow.

        Args:
            target_package: When provided, exploration is scoped to this
                application package.  After every action the agent verifies
                the foreground package and recovers if the device has drifted
                outside the target app.
        """

        super().__init__(workflow_id=workflow_id, configuration=configuration)

        self.__device = device
        self.__vision = vision
        self.__capture = capture
        self.__memory = memory

        self.__seed = seed
        self.__knowledge_graph = knowledge_graph
        self.__target_package = target_package
        self.__completion_reason = ""

    @property
    def name(self) -> str:
        """
        Returns the workflow type name.
        """

        return "exploration"

    async def execute(self) -> ExplorationResult:
        """
        Runs the mapping process via the LangGraph StateGraph.
        """

        logger.info("Executing exploration")

        from fathom.graph.exploration_graph import build_exploration_graph

        config = self.configuration

        if not self.__vision or not self.__knowledge_graph or not self.__memory:
            raise FathomError(
                "Exploration requires vision, knowledge_graph, and memory to be configured."
            )

        compiled_graph, node_ctx = build_exploration_graph(
            device=self.__device,
            capture=self.__capture,
            vision=self.__vision,
            knowledge_graph=self.__knowledge_graph,
            memory=self.__memory,
            max_steps=config.max_steps if config else 100,
            workflow_id=self.workflow_id,
            cancel_event=self.cancel_event,
            pause_event=self.pause_event,
            target_package=self.__target_package,
        )

        initial_state = {
            "max_steps": config.max_steps if config else 100,
            "step_number": 0,
            "step_results": [],
            "is_complete": False,
            "bfs_phase": "scan",
            "content_exhausted": False,
        }

        graph_task = asyncio.create_task(
            compiled_graph.ainvoke(
                initial_state,
                config={
                    "recursion_limit": 710,
                    "configurable": {"thread_id": self.workflow_id},
                },
            )
        )

        cancel_waiter = asyncio.create_task(self.cancel_event.wait())

        done, pending = await asyncio.wait(
            {graph_task, cancel_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if cancel_waiter in done and graph_task not in done:
            logger.info("LangGraph exploration cancelled by user")
            graph_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await graph_task

            self.__completion_reason = "Workflow cancelled by user"
            return self.__build_result(
                node_ctx=node_ctx,
                final_state=None,
                cancelled=True,
            )

        cancel_waiter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cancel_waiter

        final_state = graph_task.result()
        return self.__build_result(
            node_ctx=node_ctx,
            final_state=final_state,
            cancelled=False,
        )

    def __build_result(
        self,
        node_ctx: ExplorationNodeContext,
        final_state: Optional[Dict[str, Any]],
        *,
        cancelled: bool,
    ) -> ExplorationResult:
        """Map graph terminal state to ExplorationResult."""

        kg = node_ctx.knowledge_graph

        metrics_report = node_ctx.metrics.to_report_dict()

        if cancelled or final_state is None:
            stats = kg.get_stats()
            unique = stats.get("unique_screens", 0)
            unexplored = stats.get("unexplored", 0)
            coverage = ((unique - unexplored) / unique * 100) if unique > 0 else 0.0

            return ExplorationResult(
                unique_screens=unique,
                screen_graph=kg.export_json(),
                coverage_percentage=coverage,
                total_actions=stats.get("total_transitions", 0),
                total_transitions=stats.get("total_transitions", 0),
                discovered_activities=stats.get("activities", []),
                knowledge_graph=kg.export_json(),
                metrics=metrics_report,
            )

        step_results = final_state.get("step_results", [])
        for sr in step_results:
            self.record_step(result=sr)

        stats = kg.get_stats()
        unique = stats.get("unique_screens", 0)
        unexplored = stats.get("unexplored", 0)
        coverage = ((unique - unexplored) / unique * 100) if unique > 0 else 0.0

        completion_reason = final_state.get("completion_reason", "")
        is_complete = final_state.get("is_complete", False)

        return ExplorationResult(
            success=bool(is_complete and unique > 0),
            completion_reason=str(completion_reason) if completion_reason else "",
            unique_screens=unique,
            screen_graph=kg.export_json(),
            coverage_percentage=coverage,
            total_actions=stats.get("total_transitions", 0),
            total_transitions=stats.get("total_transitions", 0),
            discovered_activities=stats.get("activities", []),
            knowledge_graph=kg.export_json(),
            metrics=metrics_report,
        )

    def get_progress(self) -> Dict[str, Any]:
        """
        Real-time progress data.
        """

        return {
            "elapsed": self.elapsed,
            "steps": self.steps_executed,
        }
