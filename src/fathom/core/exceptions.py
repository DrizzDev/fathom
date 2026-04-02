from __future__ import annotations

from fathom.schemas.results import ToolErrorFeedback


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

    def display(self, *, fallback: str) -> str:
        """
        Return a safe display message using the provided fallback by default.
        """

        return fallback


class StrategyError(FathomError):
    """
    Exception raised by strategy execution.
    """


class ExecutionError(FathomError):
    """
    Exception raised during execution.
    """


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
        if isinstance(exception, FathomError):
            return exception.retryable
        return isinstance(exception, (ConnectionError, TimeoutError))


class DeviceConnectionClosedError(DeviceError):
    """
    Device connection is no longer available.
    """

    __CLIENT_MESSAGE = "Lost the device connection. Please retry the run."

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)

    def display(self, *, fallback: str) -> str:
        """
        Return a stable display message for closed device connections.
        """

        _ = fallback
        return self.__CLIENT_MESSAGE


class VisionError(ToolError):
    """
    Error during vision analysis.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message, tool_name="vision", retryable=retryable)


class ToolValidationError(VisionError):
    """
    Tool output could not be validated against its schema.

    This is recoverable at the strategy layer by surfacing the attached
    feedback back to the model and retrying the tool call with corrected
    arguments.
    """

    def __init__(self, feedback: ToolErrorFeedback) -> None:
        # Use the feedback message directly so callers see a concise,
        # model-ready description of what went wrong.
        super().__init__(message=feedback.message, retryable=False)
        self.feedback = feedback


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
