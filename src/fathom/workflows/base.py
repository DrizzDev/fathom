import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from fathom.constants import WorkflowStatus
from fathom.schemas.configuration import WorkflowConfig
from fathom.schemas.results import WorkflowResult
from fathom.schemas.steps import StepResult

T = TypeVar("T")


class BaseWorkflow(ABC, Generic[T]):
    """
    Abstract base for all workflows.

    Workflows encapsulate:
    - Execution logic for a specific automation type
    - State management and checkpointing
    - Progress tracking and reporting
    - Error handling and recovery

    Designed for sync execution now, extensible to Temporal later.
    """

    def __init__(
        self,
        workflow_id: str,
        configuration: Optional[WorkflowConfig] = None,
    ) -> None:
        """
        Initialize workflow.

        Args:
            workflow_id: Unique identifier for this workflow run.
            configuration: Workflow configuration.
        """

        self.__workflow_id = workflow_id
        self.__configuration = configuration or WorkflowConfig()

        self.__status = WorkflowStatus.PENDING
        self.__end_time: Optional[float] = None
        self.__start_time: Optional[float] = None

        self.__cancelled = False
        self.__error: Optional[str] = None
        self.__step_results: List[StepResult] = []

    @property
    def workflow_id(self) -> str:
        """
        Workflow identifier.
        """

        return self.__workflow_id

    @property
    def configuration(self) -> WorkflowConfig:
        """
        Workflow configuration.
        """

        return self.__configuration

    @property
    def status(self) -> WorkflowStatus:
        """
        Current workflow status.
        """

        return self.__status

    @property
    def steps_executed(self) -> int:
        """
        Number of steps executed.
        """

        return len(self.__step_results)

    @property
    def is_running(self) -> bool:
        """
        Whether workflow is currently running.
        """

        return self.__status == WorkflowStatus.RUNNING

    @property
    def elapsed(self) -> float:
        """
        Elapsed time in seconds.
        """

        if self.__start_time is None:
            return 0.0

        end = self.__end_time or time.time()
        return end - self.__start_time

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Workflow type name.
        """

        raise NotImplementedError

    @abstractmethod
    async def execute(self) -> T:
        """
        Execute the workflow.

        Returns:
            Workflow-specific result type.
        """

        raise NotImplementedError

    @abstractmethod
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress information.
        """

        raise NotImplementedError

    async def run(self) -> WorkflowResult:
        """
        Run the workflow with lifecycle management.

        Handles:
        - Status transitions
        - Timing
        - Error capture
        - Result construction

        Returns:
            WorkflowResult with execution details.
        """

        self.__start_time = time.time()
        self.__status = WorkflowStatus.RUNNING

        try:
            await self.execute()

            if self.__cancelled:
                self.__status = WorkflowStatus.CANCELLED
            else:
                self.__status = WorkflowStatus.COMPLETED

        except TimeoutError as exception:
            self.__status = WorkflowStatus.TIMEOUT
            self.__error = str(exception)

        except Exception as exception:
            self.__status = WorkflowStatus.FAILED
            self.__error = str(exception)

        finally:
            self.__end_time = time.time()

        return WorkflowResult(
            error=self.__error,
            status=self.__status,
            duration=self.elapsed,
            metadata=self.get_progress(),
            workflow_id=self.__workflow_id,
            steps_executed=self.steps_executed,
            step_results=self.__step_results.copy(),
        )

    def record_step(self, result: StepResult) -> None:
        """
        Record a completed step.

        Args:
            result: Step execution result.
        """

        self.__step_results.append(result)

    def cancel(self) -> None:
        """
        Request workflow cancellation.
        """

        self.__cancelled = True

    def is_cancelled(self) -> bool:
        """
        Check if cancellation was requested.
        """

        return self.__cancelled

    def should_checkpoint(self) -> bool:
        """
        Check if a checkpoint should be taken.
        """

        return (self.steps_executed % self.__configuration.checkpoint_interval) == 0

    def has_exceeded_timeout(self) -> bool:
        """
        Check if total timeout has been exceeded.
        """

        return self.elapsed >= self.__configuration.total_timeout

    def has_exceeded_steps(self) -> bool:
        """
        Check if max steps have been reached.
        """

        return self.steps_executed >= self.__configuration.max_steps

    def get_checkpoint(self) -> Dict[str, Any]:
        """
        Get checkpoint data for persistence.

        Returns:
            Serializable checkpoint dictionary.
        """

        return {
            "workflow_type": self.name,
            "status": self.__status.value,
            "progress": self.get_progress(),
            "elapsed_seconds": self.elapsed,
            "workflow_id": self.__workflow_id,
            "steps_executed": self.steps_executed,
            "configuration": self.__configuration.model_dump(),
        }
