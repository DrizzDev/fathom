from __future__ import annotations

import unittest

from fathom.constants.defect import (
    DefectKind,
    DefectSeverity,
    DefectSignal,
    DefectSource,
    DefectVerification,
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

    def test_kind_is_derived_from_signal_and_survives_round_trip(self) -> None:
        """
        Kind is computed from the signal, not an independent input, and persists in JSON.
        """

        defect = Defect.from_signal(
            signal=DefectSignal.LOREM_IPSUM,
            source=DefectSource.POST_RUN,
            summary="placeholder copy",
            evidence=self.__evidence(),
        )
        self.assertEqual(defect.kind, DefectKind.CONTENT)

        restored = Defect.model_validate_json(defect.model_dump_json())
        self.assertEqual(restored.kind, DefectKind.CONTENT)

    def test_verification_defaults_to_confirmed(self) -> None:
        """
        A defect is confirmed unless a detector explicitly downgrades it.
        """

        defect = Defect.from_signal(
            signal=DefectSignal.DEAD_TAP,
            source=DefectSource.INLINE,
            summary="Tap on 'Buy' did nothing",
            evidence=self.__evidence(),
        )
        self.assertEqual(defect.verification, DefectVerification.CONFIRMED)

    def test_verification_override_is_respected(self) -> None:
        """
        An explicit needs-review verification is carried onto the defect.
        """

        defect = Defect.from_signal(
            signal=DefectSignal.DEAD_TAP,
            source=DefectSource.INLINE,
            summary="Tap on a settings row did nothing",
            evidence=self.__evidence(),
            verification=DefectVerification.NEEDS_REVIEW,
        )
        self.assertEqual(defect.verification, DefectVerification.NEEDS_REVIEW)

    def test_verification_is_excluded_from_the_dedup_signature(self) -> None:
        """
        The same signal on the same screen dedups regardless of verification state.
        """

        confirmed = Defect.from_signal(
            signal=DefectSignal.DEAD_TAP,
            source=DefectSource.INLINE,
            summary="dead",
            evidence=self.__evidence(),
        )
        review = Defect.from_signal(
            signal=DefectSignal.DEAD_TAP,
            source=DefectSource.INLINE,
            summary="dead",
            evidence=self.__evidence(),
            verification=DefectVerification.NEEDS_REVIEW,
        )
        self.assertEqual(confirmed.signature, review.signature)


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

    def test_needs_review_defects_round_trip(self) -> None:
        """
        Needs-review defects serialize alongside the headline list.
        """

        review = Defect.from_signal(
            signal=DefectSignal.DEAD_TAP,
            source=DefectSource.INLINE,
            summary="uncorroborated dead tap",
            evidence=DefectEvidence(screen="hash-2"),
            verification=DefectVerification.NEEDS_REVIEW,
        )
        report = BugReport(
            metadata=ReportMetadata(
                workflow="wf-1", package="com.app", generated_at="2026-06-23T00:00:00"
            ),
            needs_review=[review],
        )

        restored = BugReport.model_validate_json(report.model_dump_json())

        self.assertEqual(restored.needs_review[0].verification, DefectVerification.NEEDS_REVIEW)


if __name__ == "__main__":
    unittest.main()
