from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


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

    def record(self, operation: str, duration: float) -> None:
        """
        Registers a new timing data point for a specific operation.
        """

        metric = getattr(self, operation, None)
        if metric and isinstance(metric, OperationMetric):
            metric.total_duration += duration
            metric.call_count += 1

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
        }
