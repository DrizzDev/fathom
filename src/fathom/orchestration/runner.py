from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from fathom.orchestration.context import ExecutionContext
from fathom.schemas.results import WorkflowResult
from fathom.tools.capture import CaptureTool
from fathom.tools.device import DeviceTool
from fathom.tools.vision import VisionTool
from fathom.workflows.base import BaseWorkflow, WorkflowConfig
from fathom.workflows.exploration import ExplorationWorkflow
from fathom.workflows.intent import IntentWorkflow


@dataclass
class RunnerConfig:
    """
    Configuration for workflow runner.
    """

    enable_logging: bool = True
    enable_checkpoints: bool = True
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    checkpoint_callback: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class RunnerResult:
    """
    Result from workflow runner.
    """

    workflow_result: WorkflowResult
    execution_context: ExecutionContext

    checkpoints_saved: int = 0
    total_duration: float = 0.0


class WorkflowRunner:
    """
    Runs workflows synchronously with lifecycle management.

    Provides:
    - Workflow instantiation from registry
    - Execution with context tracking
    - Checkpoint persistence hooks
    - Progress reporting hooks
    - Error handling and cleanup

    Example:
        ```python
        runner = WorkflowRunner(
            device=device_tool,
            capture=capture_tool,
            vision=vision_tool,
        )

        result = runner.run_intent(
            intent="Login to app",
            workflow_id="login-001",
        )
        print(f"Success: {result.workflow_result.success}")
        ```
    """

    def __init__(
        self,
        device: DeviceTool,
        capture: CaptureTool,
        vision: Optional[VisionTool] = None,
        *,
        runner_config: Optional[RunnerConfig] = None,
        workflow_config: Optional[WorkflowConfig] = None,
    ) -> None:
        """
        Initialize workflow runner.

        Args:
            device: Device tool for actions.
            capture: Capture tool for screenshots.
            vision: Optional vision tool for analysis.
            runner_config: Runner configuration.
            workflow_config: Default workflow configuration.
        """

        self.__device = device
        self.__capture = capture
        self.__vision = vision
        self.__runner_config = runner_config or RunnerConfig()
        self.__workflow_config = workflow_config or WorkflowConfig()

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
        config: Optional[WorkflowConfig] = None,
    ) -> RunnerResult:
        """
        Run an intent workflow.

        Args:
            intent: Goal to achieve.
            workflow_id: Unique workflow identifier.
            config: Optional workflow configuration override.

        Returns:
            RunnerResult with execution details.
        """

        if self.__vision is None:
            raise ValueError("Vision tool required for intent workflows")

        workflow = IntentWorkflow(
            intent=intent,
            vision=self.__vision,
            device=self.__device,
            capture=self.__capture,
            workflow_id=workflow_id,
            config=config or self.__workflow_config,
        )

        return self.__run_workflow(workflow)

    def run_exploration(
        self,
        workflow_id: str,
        *,
        seed: Optional[int] = None,
        config: Optional[WorkflowConfig] = None,
    ) -> RunnerResult:
        """
        Run an exploration workflow.

        Args:
            workflow_id: Unique workflow identifier.
            config: Optional workflow configuration override.
            seed: Random seed for reproducibility.

        Returns:
            RunnerResult with execution details.
        """

        workflow = ExplorationWorkflow(
            seed=seed,
            device=self.__device,
            vision=self.__vision,
            capture=self.__capture,
            workflow_id=workflow_id,
            config=config or self.__workflow_config,
        )

        return self.__run_workflow(workflow)

    def cancel_active(self) -> bool:
        """
        Cancel the currently active workflow.

        Returns:
            True if a workflow was cancelled.
        """

        if self.__active_workflow is None:
            return False

        self.__active_workflow.cancel()
        return True

    def __run_workflow(
        self,
        workflow: BaseWorkflow[Any],
    ) -> RunnerResult:
        """
        Run a workflow with full lifecycle.

        Args:
            workflow: Workflow to run.

        Returns:
            RunnerResult with execution details.
        """

        context = ExecutionContext(workflow_id=workflow.workflow_id)

        self.__active_workflow = workflow
        start_time = time.time()
        checkpoints_saved = 0

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(workflow.run())
            finally:
                loop.close()

            if self.__runner_config.enable_checkpoints:
                self.__save_checkpoint(workflow, context)
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
        """
        Save workflow checkpoint.

        Args:
            workflow: Workflow to checkpoint.
            context: Execution context.
        """

        if not self.__runner_config.enable_checkpoints:
            return

        checkpoint = {
            "timestamp": time.time(),
            "context": context.to_checkpoint(),
            "workflow": workflow.get_checkpoint(),
        }

        if self.__runner_config.checkpoint_callback:
            self.__runner_config.checkpoint_callback(checkpoint)

    def __report_progress(
        self,
        context: ExecutionContext,
        workflow: BaseWorkflow[Any],
    ) -> None:
        """
        Report workflow progress.

        Args:
            workflow: Running workflow.
            context: Execution context.
        """

        if not self.__runner_config.progress_callback:
            return

        progress = {
            "workflow_type": workflow.name,
            "status": workflow.status.value,
            "workflow_id": workflow.workflow_id,
            "elapsed_seconds": workflow.elapsed,
            "progress": workflow.get_progress(),
            "steps_executed": workflow.steps_executed,
        }

        self.__runner_config.progress_callback(progress)
