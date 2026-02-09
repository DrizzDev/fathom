from __future__ import annotations

from typing import Any, Dict, Optional

from fathom.agent.strategies.exploration import ExplorationStrategy
from fathom.schemas.configuration import WorkflowConfig
from fathom.schemas.results import ExplorationResult
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool
from fathom.workflows.base import BaseWorkflow


class ExplorationWorkflow(BaseWorkflow[ExplorationResult]):
    """
    Workflow for systematic application mapping.
    """

    def __init__(
        self,
        workflow_id: str,
        device: DeviceTool,
        capture: CaptureTool,
        vision: Optional[VisionTool] = None,
        *,
        seed: Optional[int] = None,
        configuration: Optional[WorkflowConfig] = None,
    ) -> None:
        """
        Initialize exploration workflow.
        """

        super().__init__(workflow_id=workflow_id, configuration=configuration)

        self.__device = device
        self.__vision = vision
        self.__capture = capture

        self.__seed = seed
        self.__strategy: Optional[ExplorationStrategy] = None

    @property
    def name(self) -> str:
        """
        Returns the workflow type name.
        """

        return "exploration"

    async def execute(self) -> ExplorationResult:
        """
        Runs the mapping process.
        """

        self.__strategy = ExplorationStrategy(
            seed=self.__seed,
            device=self.__device,
            vision=self.__vision,
            capture=self.__capture,
            max_steps=self.configuration.max_steps,
            timeout=self.configuration.total_timeout,
        )

        while await self.__should_continue():
            if self.is_cancelled():
                break

            result = await self.__strategy.execute_step()

            if result.step_result:
                self.record_step(result=result.step_result)

        return self.__summarize()

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

    def __summarize(self) -> ExplorationResult:
        """
        Aggregates discovery metrics.
        """

        if self.__strategy is None:
            return ExplorationResult(
                total_actions=0,
                unique_screens=0,
                total_transitions=0,
                coverage_percentage=0.0,
            )

        graph = self.__strategy.graph
        stats = graph.get_stats()

        # Calculate coverage based on discovered vs explored nodes
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
