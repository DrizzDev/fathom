from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional

from fathom.agent.planner import StepPlanner
from fathom.agent.state import AgentState
from fathom.agent.strategies.intent import IntentStrategy
from fathom.interfaces import IMemoryProvider
from fathom.schemas.results import IntentResult
from fathom.schemas.screens import ScreenCapture
from fathom.services.decomposer import IntentDecomposer
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool
from fathom.workflows.base import BaseWorkflow, WorkflowConfig

logger = getLogger(__name__)


class IntentWorkflow(BaseWorkflow[IntentResult]):
    """
    Workflow for executing a specific intent.

    Orchestrates the full execution of a goal-directed automation:
    1. Decompose complex intent into atomic sub-intents
    2. Execute each sub-intent sequentially using IntentStrategy
    3. Return combined result
    """

    def __init__(
        self,
        workflow_id: str,
        intent: str,
        vision: VisionTool,
        device: DeviceTool,
        capture: CaptureTool,
        memory: IMemoryProvider,
        *,
        config: Optional[WorkflowConfig] = None,
    ) -> None:
        super().__init__(workflow_id=workflow_id, config=config)

        self.__vision = vision
        self.__device = device
        self.__memory = memory
        self.__capture = capture
        self.__original_intent = intent
        self.__decomposer = IntentDecomposer(model=vision.provider)

        self.__planner = StepPlanner(vision_tool=vision)
        self.__state = AgentState(
            intent=intent,
            max_steps=self.config.max_steps,
        )
        self.__strategy: Optional[IntentStrategy] = None

        self.__completion_reason = ""
        self.__sub_intents: List[str] = []
        self.__final_screen: Optional[ScreenCapture] = None

    @property
    def name(self) -> str:
        """
        Returns the name of the workflow.
        """

        return "intent"

    @property
    def intent(self) -> str:
        """
        Returns the original intent.
        """

        return self.__original_intent

    async def execute(self) -> IntentResult:
        """
        Execute the intent workflow with decomposition.
        """

        # 1. Decompose
        logger.info(f"Decomposing intent: {self.__original_intent}")
        self.__sub_intents = await self.__decomposer.decompose(intent=self.__original_intent)
        logger.info(f"Sub-intents: {self.__sub_intents}")

        # 2. Iterate through sub-intents
        for sub_intent in self.__sub_intents:
            if self.is_cancelled():
                self.__completion_reason = "Workflow cancelled"
                break

            logger.info(f"Executing Sub-Intent: {sub_intent}")

            # Create strategy for this specific sub-intent
            # We reuse the same planner and state to maintain history across sub-intents
            self.__strategy = IntentStrategy(
                intent=sub_intent,
                device=self.__device,
                memory=self.__memory,
                planner=self.__planner,
                capture=self.__capture,
                workflow_id=self.workflow_id,
                step_timeout=self.config.step_timeout,
                use_xml=self.config.use_xml_bounding_boxes,
                max_steps=self.config.max_steps // len(self.__sub_intents) + 5,
            )

            # Internal loop for this sub-intent
            while await self.__strategy.should_continue():
                if self.is_cancelled():
                    break

                result = await self.__strategy.execute_step()

                if result.step_result:
                    self.record_step(result=result.step_result)

                if result.is_terminal:
                    # If it's an error, we might want to retry or abort
                    if result.status == result.status.ERROR:
                        logger.warning(f"Sub-intent failed: {result.message}")
                    break

            if self.is_cancelled():
                break

        # Finalize
        # Success is determined by whether the LAST sub-intent completed or state is marked complete
        success = self.__strategy.state.is_complete if self.__strategy else False
        metrics = self.__strategy.metrics if self.__strategy else {}

        if not self.__completion_reason:
            if success:
                self.__completion_reason = "Goal successfully achieved"
            else:
                self.__completion_reason = "Execution failed to complete all sub-intents"

        return IntentResult(
            metrics=metrics,
            success=success,
            intent=self.__original_intent,
            steps_taken=self.steps_executed,
            final_screen=self.__final_screen,
            completion_reason=self.__completion_reason,
        )

    async def __should_continue(self) -> bool:
        """
        Required by base but unused in our overridden execute loop.
        """

        return not self.is_cancelled() and not self.__state.is_complete

    def get_progress(self) -> Dict[str, Any]:
        """
        Get progress for the intent workflow.
        """

        progress = {
            "elapsed_seconds": self.elapsed,
            "intent": self.__original_intent,
            "sub_intents": self.__sub_intents,
            "max_steps": self.config.max_steps,
            "steps_executed": self.steps_executed,
        }
        if self.__strategy:
            progress["current_sub_intent"] = self.__strategy.get_progress()

        return progress
