from __future__ import annotations

from typing import Optional

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


class InvariantViolation(FathomError):
    """
    Raised when an internal invariant is broken — a programmer error, never a runtime condition.
    """


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


class EmbeddingError(PortError):
    """
    Terminal failure inside an :class:`EmbeddingPort` implementation.
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
    feedback back to the model and retrying the tool call with corrected arguments.
    """

    def __init__(self, feedback: ToolErrorFeedback) -> None:
        # Use the feedback message directly so callers see a concise, model-ready description of what went wrong.
        super().__init__(message=feedback.message, retryable=True)
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


class OcrError(FathomError):
    """
    OCR provider call failed, timed out, or returned malformed data.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message, retryable=retryable)


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


class TranslationError(AgentError):
    """
    A decomposition proposal could not be translated into a canonical success; fail closed.
    """

    def __init__(self, *, reason: str) -> None:
        """
        Bind the fail-closed reason the caller surfaces.
        """

        super().__init__(f"Proposal translation failed: {reason}", retryable=False)
        self.__reason = reason

    @property
    def reason(self) -> str:
        """
        Machine-readable cause of the translation failure.
        """

        return self.__reason


class DecompositionError(AgentError):
    """
    Intent decomposition could not produce one accepted plan; the run must execute nothing.
    """

    def __init__(self, *, intent: str, reason: str) -> None:
        """
        Bind the offending intent and the fail-closed reason for the caller to surface.
        """

        super().__init__(f"Decomposition failed for intent: {reason}", retryable=False)
        self.__intent = intent
        self.__reason = reason

    @property
    def intent(self) -> str:
        """
        Intent whose decomposition failed closed.
        """

        return self.__intent

    @property
    def reason(self) -> str:
        """
        Machine-readable cause of the decomposition failure.
        """

        return self.__reason


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

    def __init__(self, workflow_id: str, *, reason: Optional[str] = None) -> None:
        message = f"Workflow '{workflow_id}' was cancelled"
        super().__init__(message, retryable=False)

        self.reason = reason
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


class LanguageComplianceError(FathomError):
    """
    A generated flow or rendered script violated a dialect compliance gate.
    """


class LanguageParseError(FathomError):
    """
    Rendered script text could not be parsed into a syntax tree.
    """


class HITLNotAvailableError(FathomError):
    """
    Raised when HITL is requested on a runtime with no human available.
    """

    def __init__(self, *, workflow_id: Optional[str] = None) -> None:
        super().__init__("HITL requested without human availability", retryable=True)

        self.workflow_id = workflow_id


class HITLTimeoutError(FathomError):
    """
    Raised when an interactive ask exhausts its deadline without a human response.
    """

    def __init__(self, *, workflow_id: Optional[str] = None) -> None:
        super().__init__("HITL ask deadline exhausted without a response", retryable=False)

        self.workflow_id = workflow_id


class FinalizationTimeoutError(FathomError):
    """
    Post-terminal finalization phase exceeded its allotted timeout.
    """

    def __init__(self, *, phase: str, timeout: float, workflow_id: Optional[str] = None) -> None:
        suffix = "" if workflow_id is None else f" (workflow_id={workflow_id})"
        super().__init__(
            f"Finalization phase '{phase}' exceeded {timeout}s{suffix}", retryable=False
        )

        self.phase = phase
        self.timeout = timeout
        self.workflow_id = workflow_id


class CheckpointStoreError(FathomError):
    """
    LangGraph checkpoint store could not complete a required operation.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class InteractionError(FathomError):
    """
    Error raised when an interaction invariant is violated.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize an interaction error as non-retryable.
        """

        super().__init__(message, retryable=False)


class IdentityError(InteractionError):
    """
    Error raised when a runtime is invoked without a fully-resolved Principal.

    Carries the offending field name so a host can map this exception to a
    typed validation failure (e.g. JSEND `fail` with the field name) without
    parsing the message string.
    """

    def __init__(self, *, field: str, message: str) -> None:
        """
        Initialize an identity error with the offending field name.
        """

        super().__init__(message)
        self.__field = field

    @property
    def field(self) -> str:
        """
        Name of the missing or invalid identity field.
        """

        return self.__field

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(field={self.__field!r}, message={self.message!r})"


class ThreadNotFoundError(InteractionError):
    """
    Error raised when a tenant-scoped thread lookup misses.
    """

    def __init__(self, *, thread: str, message: str) -> None:
        """
        Initialize with the missing thread identifier.
        """

        super().__init__(message)
        self.__thread = thread

    @property
    def thread(self) -> str:
        """
        Identifier of the thread that does not exist.
        """

        return self.__thread


class ThreadConflictError(InteractionError):
    """
    Error raised when a thread create request collides with an existing thread of the same
    identifier but different content.
    """

    def __init__(self, *, thread: str, message: str) -> None:
        """
        Initialize a thread conflict error with the offending identifier.
        """

        super().__init__(message)
        self.__thread = thread

    @property
    def thread(self) -> str:
        """
        Identifier of the thread whose stored content conflicts.
        """

        return self.__thread


class TaskConflictError(InteractionError):
    """
    Error raised when a task transition conflicts with the existing record.
    """

    def __init__(self, *, task: str, message: str) -> None:
        """
        Initialize a task conflict error with the offending task identifier.
        """

        super().__init__(message)
        self.__task = task

    @property
    def task(self) -> str:
        """
        Identifier of the task whose terminal state conflicts.
        """

        return self.__task


class JobLeaseLostError(InteractionError):
    """
    Error raised when a worker tries to finalize a job whose lease was lost to another worker via stale-claim recovery.
    """

    def __init__(self, *, job: str, message: str) -> None:
        """
        Initialize a lease-lost error with the offending job identifier.
        """

        super().__init__(message)
        self.__job = job

    @property
    def job(self) -> str:
        """
        Identifier of the job whose lease was lost.
        """

        return self.__job


class ConversationSummaryLimitExceeded(InteractionError):
    """
    Raised when a conversation has more rows than the /summary projection caps allow.
    """

    def __init__(self, *, kind: str, thread: str, limit: int) -> None:
        """
        Initialize with the row kind, thread id, and the cap that was exceeded.
        """

        super().__init__(
            f"Conversation {thread!r} has more {kind} rows than the summary projection cap of {limit}."
        )
        self.__kind = kind
        self.__limit = limit
        self.__thread = thread

    @property
    def kind(self) -> str:
        """
        Row kind that overflowed (message kind value, or 'script').
        """

        return self.__kind

    @property
    def thread(self) -> str:
        """
        Identifier of the conversation thread that overflowed.
        """

        return self.__thread

    @property
    def limit(self) -> int:
        """
        Cap value the projection was bounded to.
        """

        return self.__limit


class StorageConfigurationError(InteractionError):
    """
    Error raised when an interaction storage backend is selected without a valid corresponding configuration.
    """

    def __init__(self, *, backend: str, message: str) -> None:
        """
        Initialize a storage configuration error with the backend name.
        """

        super().__init__(message)
        self.__backend = backend

    @property
    def backend(self) -> str:
        """
        Name of the misconfigured storage backend.
        """

        return self.__backend

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backend={self.__backend!r}, message={self.message!r})"
