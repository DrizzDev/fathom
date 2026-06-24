"""
Aggregation of detected defects into a bug report.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from fathom.constants.defect import DefectKind, DefectSeverity, DefectVerification
from fathom.schemas.defect import BugReport, Defect
from fathom.schemas.report import ReportMetadata


class DefectAggregator:
    """
    Orders defects by severity and summarises them into a bug report.
    """

    def build(self, *, defects: Sequence[Defect], metadata: ReportMetadata) -> BugReport:
        """
        Builds a bug report, leading with confirmed defects and holding the rest for review.

        Only confirmed defects feed the headline list and the per-kind/per-severity
        counts; needs-review defects are surfaced separately so uncorroborated
        signals never inflate the headline.
        """

        confirmed = sorted(
            self.__by_verification(defects, DefectVerification.CONFIRMED), key=self.__sort_key
        )
        needs_review = sorted(
            self.__by_verification(defects, DefectVerification.NEEDS_REVIEW), key=self.__sort_key
        )
        return BugReport(
            metadata=metadata,
            defects=list(confirmed),
            needs_review=list(needs_review),
            by_kind=self.__counts_by_kind(defects=confirmed),
            by_severity=self.__counts_by_severity(defects=confirmed),
        )

    @staticmethod
    def __by_verification(
        defects: Sequence[Defect], verification: DefectVerification
    ) -> List[Defect]:
        """
        Selects the defects in the given verification state.
        """

        return [defect for defect in defects if defect.verification is verification]

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
