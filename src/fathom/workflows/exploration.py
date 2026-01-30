from __future__ import annotations

from typing import Any, Dict, Optional

from fathom.agent.strategies.exploration import ExplorationStrategy
from fathom.schemas.results import ExplorationResult
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool
from fathom.workflows.base import BaseWorkflow, WorkflowConfig


class ExplorationWorkflow(BaseWorkflow[ExplorationResult]):
    """Workflow for exploring and mapping an application.

    Systematically explores an app to discover:
    - All reachable screens
    - Transitions between screens
    - Available actions on each screen
    - Coverage metrics

    Example:
        ```python
        workflow = ExplorationWorkflow(
            workflow_id="explore-001",
            device=device_tool,
            capture=capture_tool,
            config=WorkflowConfig(max_steps=100),
        )
        result = await workflow.run()
        print(f"Discovered {result.unique_screens} screens")
        ```
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
        """Initialize exploration workflow.

        Args:
            workflow_id: Unique workflow identifier.
            device: Device tool for actions.
            capture: Capture tool for screenshots.
            vision: Optional vision tool for guided exploration.
            config: Optional workflow configuration.
            seed: Random seed for reproducibility.
        """
        super().__init__(workflow_id, config)
        self.__device = device
        self.__capture = capture
        self.__vision = vision
        self.__seed = seed

        self.__strategy: Optional[ExplorationStrategy] = None

    @property
    def name(self) -> str:
        """
        Workflow type name.
        """
        return "exploration"

    async def execute(self) -> ExplorationResult:
        """Execute the exploration workflow.

        Runs the exploration strategy loop until:
        - Max steps exceeded
        - Timeout exceeded
        - All screens explored
        - Workflow cancelled

        Returns:
            ExplorationResult with discovery details.
        """
        self.__strategy = ExplorationStrategy(
            self.__device,
            self.__capture,
            self.__vision,
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

            if result.should_checkpoint:
                await self.__save_checkpoint()

        return self.__build_result()

    async def __should_continue(self) -> bool:
        """
        Check if exploration should continue.
        """
        if self.has_exceeded_timeout():
            return False
        if self.has_exceeded_steps():
            return False
        if self.__strategy is None:
            return False
        return await self.__strategy.should_continue()

    async def __save_checkpoint(self) -> None:
        """
        Save checkpoint (placeholder for persistence).
        """
        pass

    def __build_result(self) -> ExplorationResult:
        """
        Build exploration result from strategy state.
        """
        if self.__strategy is None:
            return ExplorationResult(
                unique_screens=0,
                total_transitions=0,
                total_actions=0,
                coverage_percentage=0.0,
            )

        graph = self.__strategy.graph
        stats = graph.get_coverage_stats()

        activities = list({node.activity for node in graph.nodes.values()})

        unexplored_raw = stats.get("unexplored_screens", 0)
        total_screens_raw = stats.get("unique_screens", 1)
        unexplored = int(unexplored_raw) if isinstance(unexplored_raw, (int, float)) else 0
        total_screens = int(total_screens_raw) if isinstance(total_screens_raw, (int, float)) else 1

        explored = total_screens - unexplored
        coverage = (explored / total_screens * 100) if total_screens > 0 else 0.0

        graph_data: Dict[str, Any] = {
            "nodes": {
                h: {
                    "activity": n.activity,
                    "visit_count": n.visit_count,
                    "actions_tried": list(n.actions_tried),
                }
                for h, n in graph.nodes.items()
            },
            "edges": [{"from": e[0], "action": e[1], "to": e[2]} for e in graph.edges],
        }

        unique_raw = stats.get("unique_screens", 0)
        transitions_raw = stats.get("total_transitions", 0)
        actions_raw = stats.get("total_actions_tried", 0)

        return ExplorationResult(
            unique_screens=int(unique_raw) if isinstance(unique_raw, (int, float)) else 0,
            total_transitions=int(transitions_raw)
            if isinstance(transitions_raw, (int, float))
            else 0,
            total_actions=int(actions_raw) if isinstance(actions_raw, (int, float)) else 0,
            coverage_percentage=coverage,
            discovered_activities=activities,
            screen_graph=graph_data,
        )

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current exploration progress.
        """
        progress: Dict[str, Any] = {
            "steps_executed": self.steps_executed,
            "max_steps": self.config.max_steps,
            "elapsed_seconds": self.elapsed,
        }

        if self.__strategy:
            strategy_progress = self.__strategy.get_progress()
            for k, v in strategy_progress.items():
                progress[k] = v

        return progress
