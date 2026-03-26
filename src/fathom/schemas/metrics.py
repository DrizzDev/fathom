from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class WorkflowOperationMetrics(BaseModel):
    """
    Per-workflow operation counters for Temporal billing observability.
    """

    signals_received: int = Field(
        default=0, description="Total Temporal signals delivered to this workflow"
    )
    signal_checks: int = Field(
        default=0, description="Adapter-side check_signal calls (free, in-process)"
    )
    pause_waits: int = Field(default=0, description="Times the adapter entered wait_for_pause")
    resume_waits: int = Field(default=0, description="Times the adapter entered wait_for_resume")

    context_injections: int = Field(
        default=0, description="User contexts enqueued via inject signal"
    )
    context_consumptions: int = Field(
        default=0, description="User contexts dequeued by the adapter"
    )

    wait_cycles: int = Field(default=0, description="Condition wait iterations (timeout-bounded)")
    heartbeats_sent: int = Field(default=0, description="Activity heartbeats emitted during waits")


class OperationMetric(BaseModel):
    """
    Performance metrics for a specific system operation.
    """

    total_duration: float = Field(default=0.0, description="Cumulative time spent in seconds")
    call_count: int = Field(default=0, description="Total number of times operation was invoked")

    @property
    def average(self) -> float:
        """
        Calculates the average duration per call.
        """

        if self.call_count == 0:
            return 0.0

        return self.total_duration / self.call_count


class ExecutionMetrics(BaseModel):
    """
    Aggregated performance audit for an entire workflow.
    """

    screenshot: OperationMetric = Field(default_factory=OperationMetric)
    hierarchy_dump: OperationMetric = Field(default_factory=OperationMetric)
    hierarchy_processing: OperationMetric = Field(default_factory=OperationMetric)

    analysis: OperationMetric = Field(default_factory=OperationMetric)
    action: OperationMetric = Field(default_factory=OperationMetric)

    # Token usage tracking
    cached_tokens: int = Field(default=0, description="Tokens served from cache")
    prompt_tokens: int = Field(default=0, description="Total prompt tokens consumed")
    completion_tokens: int = Field(default=0, description="Total completion tokens consumed")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record_tokens(self, prompt: int = 0, completion: int = 0, cached: int = 0) -> None:
        """
        Accumulates token usage from an LLM call.
        """

        self.cached_tokens += cached
        self.prompt_tokens += prompt
        self.completion_tokens += completion

    def record(self, operation: str, duration: float) -> None:
        """
        Registers a new timing data point for a specific operation.
        """

        metric = getattr(self, operation, None)
        if metric and isinstance(metric, OperationMetric):
            metric.call_count += 1
            metric.total_duration += duration

    def to_report_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Converts internal metrics into a reporting-optimized dictionary structure.
        """

        return {
            "Screenshot": {"total": self.screenshot.total_duration, "avg": self.screenshot.average},
            "Hierarchy Dump": {
                "avg": self.hierarchy_dump.average,
                "total": self.hierarchy_dump.total_duration,
            },
            "Hierarchy Processing": {
                "avg": self.hierarchy_processing.average,
                "total": self.hierarchy_processing.total_duration,
            },
            "Action": {"avg": self.action.average, "total": self.action.total_duration},
            "Analysis": {"avg": self.analysis.average, "total": self.analysis.total_duration},
            "Tokens": {
                "total": self.total_tokens,
                "cached": self.cached_tokens,
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary for result objects.
        """

        return {
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "prompt_tokens": self.prompt_tokens,
            "action_count": self.action.call_count,
            "analysis_count": self.analysis.call_count,
            "completion_tokens": self.completion_tokens,
            "screenshot_count": self.screenshot.call_count,
            "action_total": int(self.action.total_duration * 1000),
            "analysis_total": int(self.analysis.total_duration * 1000),
            "screenshot_total": int(self.screenshot.total_duration * 1000),
        }
