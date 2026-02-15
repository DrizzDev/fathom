"""
LEGACY CODE - DEPRECATED

This module contains the old IntentWorkflow implementation.
It is preserved for backward compatibility via the 'fathom-old' command.

NEW CODE: Use the hexagonal architecture instead:
- Strategy: src/fathom/strategies/intent.py
- Runner: src/fathom/runtime/runner.py

This code will be removed in a future major version.
"""

from __future__ import annotations

import warnings
from logging import getLogger
from typing import Any, Dict, Optional

from fathom.agent.planner import StepPlanner
from fathom.agent.strategies.intent import IntentStrategy
from fathom.interfaces import IMemoryProvider
from fathom.schemas.configuration import WorkflowConfig
from fathom.schemas.results import IntentResult
from fathom.schemas.screens import ScreenCapture
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool
from fathom.workflows.base import BaseWorkflow

logger = getLogger(__name__)


class IntentWorkflow(BaseWorkflow[IntentResult]):
    """
    DEPRECATED: Old IntentWorkflow implementation.
    
    Use the new hexagonal architecture instead:
    - from fathom.strategies.intent import IntentStrategy
    - from fathom.runtime.runner import FathomRunner
    
    This class is preserved for backward compatibility and will be removed
    in a future major version.
    
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
        configuration: Optional[WorkflowConfig] = None,
    ) -> None:
        """
        Initialize intent workflow.
        """

        super().__init__(workflow_id=workflow_id, configuration=configuration)

        self.__vision = vision
        self.__device = device
        self.__memory = memory
        self.__capture = capture
        self.__original_intent = intent

        self.__planner = StepPlanner(vision_tool=vision)
        # Create strategy immediately since we no longer decompose
        self.__strategy = IntentStrategy(
            intent=intent,
            device=device,
            memory=memory,
            planner=self.__planner,
            capture=capture,
            workflow_id=workflow_id,
            step_timeout=configuration.step_timeout if configuration else 30.0,
            use_xml=configuration.use_xml_bounding_boxes if configuration else False,
            max_steps=configuration.max_steps if configuration else 10,
        )

        self.__completion_reason = ""
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

        # Execute strategy loop
        logger.info(f"Executing intent: {self.__original_intent}")

        while await self.__strategy.should_continue():
            if self.is_cancelled():
                self.__completion_reason = "Workflow cancelled"
                break

            result = await self.__strategy.execute_step()

            if result.step_result:
                self.record_step(result=result.step_result)

            if result.is_terminal:
                if result.status == result.status.ERROR:
                    logger.warning(f"Intent failed: {result.message}")
                break

        # Finalize
        success = self.__strategy.state.is_complete

        if not self.__completion_reason:
            if success:
                self.__completion_reason = "Goal successfully achieved"
            else:
                self.__completion_reason = "Execution failed or timed out"

        return IntentResult(
            metrics=self.__strategy.metrics,
            success=success,
            intent=self.__original_intent,
            steps_taken=self.steps_executed,
            final_screen=self.__final_screen,
            completion_reason=self.__completion_reason,
            step_results=self.recorded_steps,
        )

    async def __should_continue(self) -> bool:
        """
        Required by base but unused in our overridden execute loop.
        """

        return not self.is_cancelled() and not (
            self.__strategy and self.__strategy.state.is_complete
        )

    def get_progress(self) -> Dict[str, Any]:
        """
        Get progress for the intent workflow.
        """

        progress = {
            "elapsed": self.elapsed,
            "intent": self.__original_intent,
            "steps_executed": self.steps_executed,
            "max_steps": self.configuration.max_steps,
        }
        if self.__strategy:
            progress["strategy"] = self.__strategy.get_progress()

        return progress
