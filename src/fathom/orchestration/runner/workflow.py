from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from fathom.interfaces import IMemoryProvider
from fathom.schemas.orchestration import ExecutionContext, RunnerConfig, RunnerResult
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool
from fathom.workflows.base import BaseWorkflow, WorkflowConfig
from fathom.workflows.intent import IntentWorkflow
from fathom.workflows.exploration import ExplorationWorkflow


class WorkflowRunner:
    """Runs workflows synchronously with lifecycle management.

    Provides:
    - Workflow instantiation from registry
    - Execution with context tracking
    - Checkpoint persistence hooks
    - Progress reporting hooks
    - Error handling and cleanup
    """

    def __init__(
        self,
        device: DeviceTool,
        capture: CaptureTool,
        memory: IMemoryProvider,
        vision: Optional[VisionTool] = None,
        *,
        runner_configuration: Optional[RunnerConfig] = None,
        workflow_configuration: Optional[WorkflowConfig] = None,
    ) -> None:
        """Initialize workflow runner."""
        self.__device = device
        self.__capture = capture
        self.__memory = memory
        self.__vision = vision
        self.__runner_configuration = runner_configuration or RunnerConfig()
        self.__workflow_configuration = workflow_configuration or WorkflowConfig()

        self.__active_workflow: Optional[BaseWorkflow[Any]] = None

    @property
    def has_active_workflow(self) -> bool:
        """
        Check if a workflow is currently running.
        """
        return self.__active_workflow is not None

    def run_intent(
        self,
        intent: str,
        workflow_id: str,
        *,
        configuration: Optional[WorkflowConfig] = None,
    ) -> RunnerResult:
        """Run an intent workflow."""
        if self.__vision is None:
            raise ValueError("Vision tool required for intent workflows")

        workflow = IntentWorkflow(
            workflow_id=workflow_id,
            intent=intent,
            vision=self.__vision,
            device=self.__device,
            capture=self.__capture,
            memory=self.__memory,
            config=configuration or self.__workflow_configuration,
        )

        return self.__run_workflow(workflow=workflow)

    def run_exploration(
        self,
        workflow_id: str,
        *,
        configuration: Optional[WorkflowConfig] = None,
    ) -> RunnerResult:
        """Run an exploration workflow."""
        workflow = ExplorationWorkflow(
            workflow_id=workflow_id,
            device=self.__device,
            capture=self.__capture,
            vision=self.__vision,
            config=configuration or self.__workflow_configuration,
        )

        return self.__run_workflow(workflow=workflow)

    def cancel_active(self) -> bool:
        """Cancel the currently active workflow."""
        if self.__active_workflow is None:
            return False

        self.__active_workflow.cancel()
        return True

    def __run_workflow(
        self,
        workflow: BaseWorkflow[Any],
    ) -> RunnerResult:
        """Run a workflow with full lifecycle."""
        context = ExecutionContext(
            workflow_id=workflow.workflow_id,
        )

        self.__active_workflow = workflow
        start_time = time.time()
        checkpoints_saved = 0

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(workflow.execute())
            finally:
                loop.close()

            if self.__runner_configuration.enable_checkpoints:
                self.__save_checkpoint(workflow=workflow, context=context)
                checkpoints_saved += 1

        finally:
            context.finish()
            self.__active_workflow = None

        return RunnerResult(
            workflow_result=result,
            execution_context=context,
            checkpoints_saved=checkpoints_saved,
            total_duration=time.time() - start_time,
        )

    def __save_checkpoint(
        self,
        workflow: BaseWorkflow[Any],
        context: ExecutionContext,
    ) -> None:
        """Save workflow checkpoint."""
        if not self.__runner_configuration.enable_checkpoints:
            return

        checkpoint = {
            "workflow": workflow.get_checkpoint(),
            "context": context.to_checkpoint(),
            "timestamp": time.time(),
        }

        if self.__runner_configuration.checkpoint_callback:
            self.__runner_configuration.checkpoint_callback(checkpoint)
