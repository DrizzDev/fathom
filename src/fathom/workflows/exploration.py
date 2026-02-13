from __future__ import annotations

import asyncio
import contextlib
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from fathom.graph.exploration_nodes import ExplorationNodeContext

from fathom.agent.strategies.exploration import ExplorationStrategy
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

    When ``use_langgraph=True``, the execution loop is replaced by a
    LangGraph StateGraph that drives the same underlying BFS exploration
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
        use_langgraph: bool = False,
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
        self.__use_langgraph = use_langgraph
        self.__target_package = target_package
        self.__strategy: Optional[ExplorationStrategy] = None
        self.__completion_reason = ""

    @property
    def name(self) -> str:
        """
        Returns the workflow type name.
        """

        return "exploration"

    async def execute(self) -> ExplorationResult:
        """
        Runs the mapping process.

        Delegates to the LangGraph StateGraph when ``use_langgraph`` is
        ``True``; otherwise falls back to the original strategy loop.
        """

        logger.info("Executing exploration (langgraph=%s)", self.__use_langgraph)

        if self.__use_langgraph:
            return await self.__execute_langgraph()

        return await self.__execute_classic()

    # ── Classic path (original strategy loop) ──────────────────────

    async def __execute_classic(self) -> ExplorationResult:
        """Original while-loop strategy execution."""

        self.__strategy = ExplorationStrategy(
            seed=self.__seed,
            device=self.__device,
            vision=self.__vision,
            capture=self.__capture,
            max_steps=self.configuration.max_steps,
            timeout=self.configuration.total_timeout,
            knowledge_graph=self.__knowledge_graph,
            target_package=self.__target_package,
        )

        while await self.__should_continue():
            if self.is_cancelled():
                break

            result = await self.__strategy.execute_step()

            if result.step_result:
                self.record_step(result=result.step_result)

        return self.__summarize_classic()

    # ── LangGraph path ─────────────────────────────────────────────

    async def __execute_langgraph(self) -> ExplorationResult:
        """Execute using a LangGraph StateGraph.

        The graph invocation is wrapped in an asyncio task so that
        ``cancel()`` (via SIGINT) can cancel the task immediately.
        """

        from fathom.graph.exploration_graph import build_exploration_graph

        config = self.configuration

        if not self.__vision or not self.__knowledge_graph or not self.__memory:
            logger.warning(
                "LangGraph exploration requires vision, knowledge_graph, and memory. "
                "Falling back to classic path."
            )
            return await self.__execute_classic()

        compiled_graph, node_ctx = build_exploration_graph(
            device=self.__device,
            capture=self.__capture,
            vision=self.__vision,
            knowledge_graph=self.__knowledge_graph,
            memory=self.__memory,
            max_steps=config.max_steps if config else 100,
            workflow_id=self.workflow_id,
            cancel_event=self.cancel_event,
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

        # Run ainvoke in a task so we can cancel it
        graph_task = asyncio.create_task(
            compiled_graph.ainvoke(
                initial_state,
                config={
                    "recursion_limit": 710,
                    "configurable": {"thread_id": self.workflow_id},
                },
            )
        )

        # Monitor: wait for either graph completion or cancellation
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
            return self.__build_langgraph_result(
                node_ctx=node_ctx,
                final_state=None,
                cancelled=True,
            )

        # Clean up cancel waiter
        cancel_waiter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cancel_waiter

        final_state = graph_task.result()
        return self.__build_langgraph_result(
            node_ctx=node_ctx,
            final_state=final_state,
            cancelled=False,
        )

    def __build_langgraph_result(
        self,
        node_ctx: ExplorationNodeContext,
        final_state: Optional[Dict[str, Any]],
        *,
        cancelled: bool,
    ) -> ExplorationResult:
        """Map graph terminal state to ExplorationResult."""

        kg = node_ctx.knowledge_graph

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
            )

        step_results = final_state.get("step_results", [])
        for sr in step_results:
            self.record_step(result=sr)

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
        )

    async def __should_continue(self) -> bool:
        """
        Lifecycle check.
        """

        if self.has_exceeded_timeout():
            return False

        if self.has_exceeded_steps():
            return False

        if self.__strategy is None:
            return False

        return await self.__strategy.should_continue()

    def __summarize_classic(self) -> ExplorationResult:
        """
        Aggregates discovery metrics for the classic path.
        Uses KnowledgeGraph export when available for richer cross-run data.
        """

        if self.__strategy is None:
            return ExplorationResult(
                total_actions=0,
                unique_screens=0,
                total_transitions=0,
                coverage_percentage=0.0,
            )

        # Prefer KnowledgeGraph for stats + export (includes cross-run data)
        kg = self.__strategy.knowledge_graph
        if kg:
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
            )

        # Fallback: in-memory ExplorationGraph
        graph = self.__strategy.graph
        stats = graph.get_stats()

        unique = stats.get("unique_screens", 0)
        unexplored = stats.get("unexplored", 0)
        coverage = ((unique - unexplored) / unique * 100) if unique > 0 else 0.0

        graph_data: Dict[str, Any] = {
            "nodes": {
                key: {
                    "activity": node.activity,
                    "visits": node.visits,
                    "actions": list(node.actions),
                }
                for key, node in graph.nodes.items()
            },
            "edges": [{"from": edge[0], "action": edge[1], "to": edge[2]} for edge in graph.edges],
        }

        return ExplorationResult(
            unique_screens=unique,
            screen_graph=graph_data,
            coverage_percentage=coverage,
            total_actions=stats.get("total_actions", 0),
            total_transitions=stats.get("total_transitions", 0),
            discovered_activities=list({node.activity for node in graph.nodes.values()}),
        )

    def get_progress(self) -> Dict[str, Any]:
        """
        Real-time progress data.
        """

        progress = {
            "elapsed": self.elapsed,
            "steps": self.steps_executed,
        }
        if self.__strategy:
            progress.update(self.__strategy.get_progress())

        return progress
