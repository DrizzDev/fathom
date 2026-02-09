from __future__ import annotations

from typing import Any, Dict, Optional

from fathom.agent.strategies.exploration import ExplorationStrategy
from fathom.schemas.results import ExplorationResult
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool
from fathom.workflows.base import BaseWorkflow, WorkflowConfig


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
        config: Optional[WorkflowConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(workflow_id, config)
        self.__device = device
        self.__capture = capture
        self.__vision = vision
        self.__seed = seed
        self.__strategy: Optional[ExplorationStrategy] = None

    @property
    def name(self) -> str:
        return "exploration"

    async def execute(self) -> ExplorationResult:
        """Runs the mapping process."""
        self.__strategy = ExplorationStrategy(
            device=self.__device,
            capture=self.__capture,
            vision=self.__vision,
            max_steps=self.config.max_steps,
            timeout=self.config.total_timeout,
            seed=self.__seed,
        )

        while await self.__should_continue():
            if self.is_cancelled():
                break

            result = await self.__strategy.execute_step()

            if result.step_result:
                self.record_step(result.step_result)

        return self.__summarize()

    async def __should_continue(self) -> bool:
        """Lifecycle check."""
        if self.has_exceeded_timeout():
            return False
        if self.has_exceeded_steps():
            return False
        if self.__strategy is None:
            return False
        return await self.__strategy.should_continue()

    def __summarize(self) -> ExplorationResult:
        """Aggregates discovery metrics."""
        if self.__strategy is None:
            return ExplorationResult(
                unique_screens=0,
                total_transitions=0,
                total_actions=0,
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
            "edges": [{"from": e[0], "action": e[1], "to": e[2]} for e in graph.edges],
        }

        return ExplorationResult(
            unique_screens=unique,
            total_transitions=stats.get("total_transitions", 0),
            total_actions=stats.get("total_actions", 0),
            coverage_percentage=coverage,
            discovered_activities=list({n.activity for n in graph.nodes.values()}),
            screen_graph=graph_data,
        )

    def get_progress(self) -> Dict[str, Any]:
        """Real-time progress data."""
        progress = {
            "steps": self.steps_executed,
            "elapsed": self.elapsed,
        }
        if self.__strategy:
            progress.update(self.__strategy.get_progress())
        return progress
