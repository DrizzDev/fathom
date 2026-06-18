"""
TelemetryPort adapter that drives the exploration TUI instead of the console.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from fathom.constants.exploration import EXPLORATION_PROGRESS_EVENT, BFSPhase
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.exploration import ExplorationProgress, TokenUsage


@runtime_checkable
class ProgressView(Protocol):
    """
    Sink the telemetry adapter pushes exploration updates into (the TUI).
    """

    def update_progress(self, progress: ExplorationProgress) -> None: ...

    def append_activity(self, message: str, *, level: str = "info") -> None: ...


class TuiTelemetryAdapter(TelemetryPort):
    """
    Routes exploration telemetry to a progress view.

    The per-step ``exploration.progress`` event becomes an ExplorationProgress
    snapshot, merged with accumulated token usage; every other call is appended
    to the activity log.
    """

    def __init__(self, *, view: ProgressView) -> None:
        self.__view = view
        self.__tokens = TokenUsage()

    def add_tokens(self, *, prompt: int, completion: int, cached: int) -> None:
        """
        Accumulate LLM token usage shown in the header (fed by the token tap).
        """

        self.__tokens = TokenUsage(
            prompt=self.__tokens.prompt + prompt,
            completion=self.__tokens.completion + completion,
            cached=self.__tokens.cached + cached,
        )

    async def debug(self, message: str, **context: Any) -> None:
        """
        Drop debug telemetry; it is too noisy for the activity log.
        """

        return None

    async def info(self, message: str, **context: Any) -> None:
        """
        Apply a progress snapshot, or append the message as activity.
        """

        if message == EXPLORATION_PROGRESS_EVENT:
            self.__publish(context=context)
            return
        self.__view.append_activity(self.__line(message=message, context=context), level="info")

    async def warning(self, message: str, **context: Any) -> None:
        """
        Append a warning to the activity log.
        """

        self.__view.append_activity(self.__line(message=message, context=context), level="warning")

    async def error(self, message: str, **context: Any) -> None:
        """
        Append an error to the activity log.
        """

        self.__view.append_activity(self.__line(message=message, context=context), level="error")

    async def exception(
        self, message: str, *, exception: Optional[BaseException] = None, **context: Any
    ) -> None:
        """
        Append an exception to the activity log.
        """

        detail = f"{message}: {exception}" if exception is not None else message
        self.__view.append_activity(detail, level="error")

    def __publish(self, *, context: Dict[str, Any]) -> None:
        progress = ExplorationProgress(
            step=int(context.get("step", 0)),
            max_steps=int(context.get("max_steps", 0)),
            phase=self.__phase(value=context.get("phase")),
            unique_screens=int(context.get("unique_screens", 0)),
            coverage=float(context.get("coverage", 0.0)),
            tokens=self.__tokens,
            status=context.get("status"),
        )
        self.__view.update_progress(progress)

        action = context.get("action")
        if action:
            self.__view.append_activity(f"step {progress.step}: {action}", level="info")

    @staticmethod
    def __phase(*, value: Any) -> BFSPhase:
        try:
            return BFSPhase(value)
        except ValueError:
            return BFSPhase.SCAN

    @staticmethod
    def __line(*, message: str, context: Dict[str, Any]) -> str:
        if not context:
            return message
        detail = " ".join(f"{key}={value}" for key, value in context.items())
        return f"{message}  {detail}"
