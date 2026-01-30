"""Intent workflow for goal-directed automation."""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

from fathom.agent.planner import StepPlanner
from fathom.agent.reasoner import Reasoner
from fathom.agent.state import AgentState
from fathom.agent.strategies.intent import IntentStrategy
from fathom.schemas.results import IntentResult
from fathom.schemas.screens import ScreenCapture
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool
from fathom.workflows.base import BaseWorkflow, WorkflowConfig


class IntentWorkflow(BaseWorkflow[IntentResult]):
    """Workflow for executing a specific intent.

    Orchestrates the full execution of a goal-directed automation:
    1. Initialize agent state and strategy
    2. Execute steps until completion, failure, or timeout
    3. Return structured result

    Example:
        ```python
        workflow = IntentWorkflow(
            workflow_id="login-001",
            intent="Login with phone number",
            vision=vision_tool,
            device=device_tool,
            capture=capture_tool,
        )
        result = await workflow.run()
        print(f"Success: {result.success}")
        ```
    """

    def __init__(
        self,
        workflow_id: str,
        intent: str,
        vision: VisionTool,
        device: DeviceTool,
        capture: CaptureTool,
        *,
        config: Optional[WorkflowConfig] = None,
    ) -> None:
        """Initialize intent workflow.

        Args:
            workflow_id: Unique workflow identifier.
            intent: Goal to achieve.
            vision: Vision tool for screen analysis.
            device: Device tool for action execution.
            capture: Capture tool for screenshots.
            config: Optional workflow configuration.
        """
        super().__init__(workflow_id, config)
        self.__intent = intent
        self.__vision = vision
        self.__device = device
        self.__capture = capture

        self.__planner = StepPlanner(vision)
        self.__reasoner = Reasoner(intent)
        self.__state = AgentState(
            intent,
            max_steps=self.config.max_steps,
        )

        self.__strategy: Optional[IntentStrategy] = None
        self.__completion_reason = ""
        self.__final_screen: Optional[ScreenCapture] = None

    @property
    def name(self) -> str:
        """Workflow type name."""
        return "intent"

    @property
    def intent(self) -> str:
        """The goal being pursued."""
        return self.__intent

    async def execute(self) -> IntentResult:
        """Execute the intent workflow.

        Runs the intent strategy loop until:
        - Intent is completed
        - Max steps exceeded
        - Timeout exceeded
        - Agent is stuck
        - Workflow is cancelled

        Returns:
            IntentResult with execution details.
        """
        self.__strategy = IntentStrategy(
            self.__intent,
            self.__planner,
            self.__device,
            self.__capture,
            max_steps=self.config.max_steps,
            step_timeout=self.config.step_timeout,
        )

        while await self.__should_continue():
            if self.is_cancelled():
                self.__completion_reason = "Workflow cancelled"
                break

            result = await self.__strategy.execute_step()

            if result.step_result:
                self.record_step(result.step_result)

            if result.is_terminal:
                self.__completion_reason = result.message
                break

            if self.should_checkpoint():
                await self.__save_checkpoint()

        if not self.__completion_reason:
            if self.has_exceeded_timeout():
                self.__completion_reason = "Workflow timeout exceeded"
            elif self.has_exceeded_steps():
                self.__completion_reason = f"Max steps ({self.config.max_steps}) exceeded"
            else:
                self.__completion_reason = "Execution completed"

        with contextlib.suppress(Exception):
            self.__final_screen = await self.__capture.capture()

        return IntentResult(
            intent=self.__intent,
            success=self.__state.is_complete,
            steps_taken=self.steps_executed,
            completion_reason=self.__completion_reason,
            final_screen=self.__final_screen,
        )

    async def __should_continue(self) -> bool:
        """Check if execution should continue."""
        if self.has_exceeded_timeout():
            return False
        if self.has_exceeded_steps():
            return False
        if self.__strategy is None:
            return False
        return await self.__strategy.should_continue()

    async def __save_checkpoint(self) -> None:
        """Save checkpoint (placeholder for persistence)."""
        pass

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        progress = {
            "intent": self.__intent,
            "steps_executed": self.steps_executed,
            "max_steps": self.config.max_steps,
            "elapsed_seconds": self.elapsed,
            "is_complete": self.__state.is_complete,
            "is_stuck": self.__state.is_stuck,
        }

        if self.__strategy:
            progress["strategy_progress"] = self.__strategy.get_progress()

        return progress
