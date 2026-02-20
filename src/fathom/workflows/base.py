import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from fathom.schemas.configuration import WorkflowConfig
from fathom.schemas.steps import StepResult

T = TypeVar("T")


class BaseWorkflow(ABC, Generic[T]):
    """
    Abstract base for all workflows.

    Workflows encapsulate:
    - Execution logic for a specific automation type
    - Progress tracking and reporting
    - Error handling and recovery
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

        self.__end_time: Optional[float] = None
        self.__start_time: Optional[float] = None

        self.__cancelled = False
        self.__cancel_event = asyncio.Event()
        self.__pause_event = asyncio.Event()
        self.__pause_event.set()
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
    def steps_executed(self) -> int:
        """
        Number of steps executed.
        """

        return len(self.__step_results)

    @property
    def recorded_steps(self) -> List[StepResult]:
        """
        Returns a copy of recorded step results.
        """

        return self.__step_results.copy()

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

        Sets both the boolean flag (for backward-compat sync checks) and
        the ``asyncio.Event`` so that async code waiting on it wakes up
        immediately.
        """

        self.__cancelled = True
        self.__cancel_event.set()

    def pause(self) -> None:
        """Pause workflow execution until resumed."""

        self.__pause_event.clear()

    def resume(self) -> None:
        """Resume workflow execution after a pause."""

        self.__pause_event.set()

    def is_paused(self) -> bool:
        """Check if workflow is currently paused."""

        return not self.__pause_event.is_set()

    @property
    def pause_event(self) -> asyncio.Event:
        """Async-friendly pause event.

        Nodes can ``await pause_event.wait()`` to block until resumed.
        """

        return self.__pause_event

    def is_cancelled(self) -> bool:
        """
        Check if cancellation was requested.
        """

        return self.__cancelled

    @property
    def cancel_event(self) -> asyncio.Event:
        """
        Async-friendly cancellation event.

        Nodes / tasks can ``await cancel_event.wait()`` or poll
        ``cancel_event.is_set()`` for fast, non-blocking cancellation.
        """

        return self.__cancel_event

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
