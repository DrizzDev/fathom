from __future__ import annotations

import unittest

from fathom.constants.defect import (
    DefectKind,
    DefectSeverity,
    DefectSignal,
    DefectSource,
)
from fathom.schemas.defect import BugReport, Defect, DefectEvidence
from fathom.schemas.report import ReportMetadata


class DefectFromSignalTest(unittest.TestCase):
    """
    Verifies the Defect.from_signal factory defaults and overrides.
    """

    def __evidence(self) -> DefectEvidence:
        """
        Builds a minimal evidence stub on a known screen.
        """

        return DefectEvidence(screen="hash-1", activity="com.app.Main")

    def test_defaults_kind_and_severity_from_signal(self) -> None:
        """
        Kind and severity fall back to the signal's defaults when not supplied.
        """

        defect = Defect.from_signal(
            signal=DefectSignal.DEAD_TAP,
            source=DefectSource.INLINE,
            summary="Tap on 'Buy' did nothing",
            evidence=self.__evidence(),
        )
        self.assertEqual(defect.kind, DefectKind.FUNCTIONAL)
        self.assertEqual(defect.severity, DefectSignal.DEAD_TAP.default_severity)

    def test_severity_override_is_respected(self) -> None:
        """
        An explicit severity overrides the signal default.
        """

        defect = Defect.from_signal(
            signal=DefectSignal.DEAD_TAP,
            source=DefectSource.INLINE,
            summary="Primary CTA is dead",
            evidence=self.__evidence(),
            severity=DefectSeverity.MAJOR,
        )
        self.assertEqual(defect.severity, DefectSeverity.MAJOR)


class BugReportTest(unittest.TestCase):
    """
    Verifies the bug report serializes for the renderer and webapp.
    """

    def test_round_trips_through_json(self) -> None:
        """
        The report serializes to JSON and back, including enum-keyed counts.
        """

        defect = Defect.from_signal(
            signal=DefectSignal.CRASH,
            source=DefectSource.INLINE,
            summary="App left the package",
            evidence=DefectEvidence(screen="hash-1"),
        )
        report = BugReport(
            metadata=ReportMetadata(
                workflow="wf-1", package="com.app", generated_at="2026-06-23T00:00:00"
            ),
            defects=[defect],
            by_kind={DefectKind.FUNCTIONAL: 1},
            by_severity={DefectSeverity.BLOCKER: 1},
        )

        restored = BugReport.model_validate_json(report.model_dump_json())

        self.assertEqual(restored.defects[0].signal, DefectSignal.CRASH)
        self.assertEqual(restored.by_kind[DefectKind.FUNCTIONAL], 1)
        self.assertEqual(restored.by_severity[DefectSeverity.BLOCKER], 1)


if __name__ == "__main__":
    unittest.main()
