from __future__ import annotations

import httpx


class FathomError(Exception):
    """
    Base exception for all Fathom errors.

    Attributes:
        message: Human-readable error description.
        retryable: Whether this error may succeed on retry.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, retryable={self.retryable})"


class StrategyError(FathomError):
    """
    Exception raised by strategy execution.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message, retryable=retryable)


class ExecutionError(FathomError):
    """
    Exception raised during execution.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message, retryable=retryable)


class ConfigurationError(FathomError):
    """
    Error in configuration or setup.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class PortError(FathomError):
    """
    Exception raised by port operations.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message, retryable=retryable)


class ToolError(FathomError):
    """
    Base for tool-related errors.
    """

    def __init__(
        self, message: str, *, tool_name: str = "unknown", retryable: bool = False
    ) -> None:
        super().__init__(f"[{tool_name}] {message}", retryable=retryable)
        self.tool_name = tool_name


class DeviceError(FathomError):
    """
    Error during device interaction.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message, retryable=retryable)

    @classmethod
    def is_transient(cls, exception: BaseException) -> bool:
        """
        Check if an error is transient and should be retried.
        """

        if isinstance(exception, httpx.HTTPStatusError):
            # Fail fast on client errors (4xx) like 404. Retry on server errors (5xx)
            return not (400 <= exception.response.status_code < 500)

        # Retry on transport/network errors
        return bool(isinstance(exception, httpx.RequestError))


class VisionError(ToolError):
    """
    Error during vision analysis.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message, tool_name="vision", retryable=retryable)


class ToolTimeoutError(ToolError):
    """
    Tool call timed out.
    """

    def __init__(self, tool_name: str, timeout: float) -> None:
        message = f"Timed out after {timeout}s"
        super().__init__(message, tool_name=tool_name, retryable=True)
        self.timeout = timeout


class ToolConnectionError(ToolError):
    """
    Failed to connect to tool backend.
    """

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(message, tool_name=tool_name, retryable=True)


class ToolExecutionError(ToolError):
    """
    Tool execution failed.
    """

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(message, tool_name=tool_name, retryable=False)


class MissingDependencyError(ConfigurationError):
    """
    Required optional dependency is not installed.
    """

    def __init__(self, dependency: str, feature: str) -> None:
        message = f"Missing dependency '{dependency}' required for {feature}"
        super().__init__(message)

        self.feature = feature
        self.dependency = dependency


class AgentError(FathomError):
    """
    Base for agent-related errors.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message, retryable=retryable)


class PlanningError(AgentError):
    """
    Failed to plan next step.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class StuckLoopError(AgentError):
    """
    Agent is stuck in a loop.
    """

    def __init__(self, iterations: int) -> None:
        message = f"Stuck in loop after {iterations} identical screens"
        super().__init__(message, retryable=False)
        self.iterations = iterations


class MaxStepsExceededError(AgentError):
    """
    Maximum step count exceeded.
    """

    def __init__(self, max_steps: int) -> None:
        message = f"Exceeded maximum of {max_steps} steps"
        super().__init__(message, retryable=False)
        self.max_steps = max_steps


class WorkflowError(FathomError):
    """
    Base for workflow-related errors.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message, retryable=retryable)


class WorkflowCancelledError(WorkflowError):
    """
    Workflow was cancelled.
    """

    def __init__(self, workflow_id: str) -> None:
        message = f"Workflow '{workflow_id}' was cancelled"
        super().__init__(message, retryable=False)
        self.workflow_id = workflow_id


class WorkflowTimeoutError(WorkflowError):
    """
    Workflow timed out.
    """

    def __init__(self, workflow_id: str, timeout: float) -> None:
        message = f"Workflow '{workflow_id}' timed out after {timeout}s"
        super().__init__(message, retryable=True)

        self.timeout = timeout
        self.workflow_id = workflow_id


class ScriptExportError(FathomError):
    """
    Script export failed.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message, retryable=retryable)
