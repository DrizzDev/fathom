from __future__ import annotations

from typing import Dict
from pydantic import BaseModel


class OperationMetric(BaseModel):
    """
    Tracks duration and count for a specific operation.
    """
    total_duration: float = 0.0
    call_count: int = 0

    @property
    def average(self) -> float:
        """
        Calculates the average duration.
        """
        if self.call_count == 0:
            return 0.0
        return self.total_duration / self.call_count


class ExecutionMetrics(BaseModel):
    """
    Schema for tracking execution performance with precise counters.
    """
    screenshot: OperationMetric = OperationMetric()
    hierarchy_dump: OperationMetric = OperationMetric()
    hierarchy_processing: OperationMetric = OperationMetric()
    analysis: OperationMetric = OperationMetric()
    action: OperationMetric = OperationMetric()

    def record(self, operation: str, duration: float) -> None:
        """
        Records a single operation duration.
        """
        metric = getattr(self, operation, None)
        if metric and isinstance(metric, OperationMetric):
            metric.total_duration += duration
            metric.call_count += 1

    def to_report_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Returns a dictionary suitable for reporting.
        """
        return {
            "Screenshot": {"total": self.screenshot.total_duration, "avg": self.screenshot.average},
            "Hierarchy Dump": {"total": self.hierarchy_dump.total_duration, "avg": self.hierarchy_dump.average},
            "Hierarchy Processing": {"total": self.hierarchy_processing.total_duration, "avg": self.hierarchy_processing.average},
            "Analysis": {"total": self.analysis.total_duration, "avg": self.analysis.average},
            "Action": {"total": self.action.total_duration, "avg": self.action.average},
        }
