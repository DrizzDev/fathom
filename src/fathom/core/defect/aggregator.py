"""
Aggregation of detected defects into a bug report.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from fathom.constants.defect import DefectKind, DefectSeverity
from fathom.schemas.defect import BugReport, Defect
from fathom.schemas.report import ReportMetadata


class DefectAggregator:
    """
    Orders defects by severity and summarises them into a bug report.
    """

    def build(self, *, defects: Sequence[Defect], metadata: ReportMetadata) -> BugReport:
        """
        Builds a severity-sorted bug report with per-kind and per-severity counts.
        """

        ordered = sorted(defects, key=self.__sort_key)
        return BugReport(
            metadata=metadata,
            defects=list(ordered),
            by_kind=self.__counts_by_kind(defects=ordered),
            by_severity=self.__counts_by_severity(defects=ordered),
        )

    @staticmethod
    def __sort_key(defect: Defect) -> Tuple[int, str, str]:
        """
        Sorts most-severe first, then by signal and screen for stable output.
        """

        return (defect.severity.rank, defect.signal.value, defect.evidence.screen)

    @staticmethod
    def __counts_by_kind(*, defects: Sequence[Defect]) -> Dict[DefectKind, int]:
        """
        Counts defects per kind.
        """

        counts: Dict[DefectKind, int] = {}
        for defect in defects:
            counts[defect.kind] = counts.get(defect.kind, 0) + 1
        return counts

    @staticmethod
    def __counts_by_severity(*, defects: Sequence[Defect]) -> Dict[DefectSeverity, int]:
        """
        Counts defects per severity.
        """

        counts: Dict[DefectSeverity, int] = {}
        for defect in defects:
            counts[defect.severity] = counts.get(defect.severity, 0) + 1
        return counts
