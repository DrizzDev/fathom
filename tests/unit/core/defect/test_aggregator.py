from __future__ import annotations

import unittest
from typing import Optional

from fathom.constants.defect import DefectKind, DefectSeverity, DefectSignal, DefectSource
from fathom.core.defect.aggregator import DefectAggregator
from fathom.schemas.defect import Defect, DefectEvidence
from fathom.schemas.report import ReportMetadata


class DefectAggregatorTest(unittest.TestCase):
    """
    Verifies severity ordering and per-kind/per-severity counts in the bug report.
    """

    @staticmethod
    def __metadata() -> ReportMetadata:
        """
        Builds report metadata with a fixed timestamp for determinism.
        """

        return ReportMetadata(workflow="wf", package="com.app", generated_at="2026-06-23T00:00:00")

    @staticmethod
    def __defect(
        *, signal: DefectSignal, severity: Optional[DefectSeverity] = None, screen: str = "home"
    ) -> Defect:
        """
        Builds a defect for the given signal and optional severity override.
        """

        return Defect.from_signal(
            signal=signal,
            source=DefectSource.POST_RUN,
            summary="x",
            evidence=DefectEvidence(screen=screen),
            severity=severity,
        )

    def test_orders_blocker_before_minor(self) -> None:
        """
        The most severe defect leads the report.
        """

        report = DefectAggregator().build(
            defects=[
                self.__defect(signal=DefectSignal.DEAD_TAP),
                self.__defect(signal=DefectSignal.CRASH),
            ],
            metadata=self.__metadata(),
        )

        self.assertEqual(
            [defect.severity for defect in report.defects],
            [DefectSeverity.BLOCKER, DefectSeverity.MINOR],
        )

    def test_counts_by_kind_and_severity(self) -> None:
        """
        The report tallies defects per kind and per severity.
        """

        report = DefectAggregator().build(
            defects=[
                self.__defect(signal=DefectSignal.CRASH),
                self.__defect(signal=DefectSignal.LOREM_IPSUM),
            ],
            metadata=self.__metadata(),
        )

        self.assertEqual(report.by_kind[DefectKind.FUNCTIONAL], 1)
        self.assertEqual(report.by_kind[DefectKind.CONTENT], 1)
        self.assertEqual(report.by_severity[DefectSeverity.BLOCKER], 1)
        self.assertEqual(report.by_severity[DefectSeverity.MAJOR], 1)


if __name__ == "__main__":
    unittest.main()
