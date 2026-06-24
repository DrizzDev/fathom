from __future__ import annotations

import unittest

from fathom.constants.defect import DefectSignal, DefectSource, DefectVerification
from fathom.core.defect.aggregator import DefectAggregator
from fathom.core.services.exporter.defect import BugReportRenderer
from fathom.schemas.defect import BugReport, Defect, DefectEvidence
from fathom.schemas.report import ReportMetadata


class BugReportRendererTest(unittest.TestCase):
    """
    Verifies the Markdown rendering of a bug report.
    """

    @staticmethod
    def __metadata() -> ReportMetadata:
        """
        Builds report metadata with a fixed timestamp for determinism.
        """

        return ReportMetadata(workflow="wf", package="com.app", generated_at="2026-06-23T00:00:00")

    def test_renders_header_and_defect_rows(self) -> None:
        """
        The rendered report names the defect, its signal, and its summary.
        """

        defect = Defect.from_signal(
            signal=DefectSignal.CRASH,
            source=DefectSource.INLINE,
            summary="App left the package",
            evidence=DefectEvidence(screen="home"),
        )
        report = DefectAggregator().build(defects=[defect], metadata=self.__metadata())

        markdown = BugReportRenderer().render(report=report)

        self.assertIn("# Bug Report", markdown)
        self.assertIn("## Defects", markdown)
        self.assertIn("App left the package", markdown)
        self.assertIn(DefectSignal.CRASH.value, markdown)

    def test_renders_empty_report(self) -> None:
        """
        An empty report still renders a header and a no-defects note.
        """

        report = BugReport(metadata=self.__metadata())

        markdown = BugReportRenderer().render(report=report)

        self.assertIn("# Bug Report", markdown)
        self.assertIn("No defects detected.", markdown)

    def test_needs_review_section_is_separate_from_the_headline(self) -> None:
        """
        Needs-review defects render in their own section, out of the headline table.
        """

        confirmed = Defect.from_signal(
            signal=DefectSignal.CRASH,
            source=DefectSource.INLINE,
            summary="App crashed",
            evidence=DefectEvidence(screen="home"),
        )
        review = Defect.from_signal(
            signal=DefectSignal.DEAD_TAP,
            source=DefectSource.INLINE,
            summary="Maybe a dead tap",
            evidence=DefectEvidence(screen="cart"),
            verification=DefectVerification.NEEDS_REVIEW,
        )
        report = DefectAggregator().build(defects=[confirmed, review], metadata=self.__metadata())

        markdown = BugReportRenderer().render(report=report)

        self.assertIn("## Defects", markdown)
        self.assertIn("## Needs review", markdown)
        self.assertIn("- Defects: 1", markdown)
        self.assertIn("- Needs review: 1", markdown)
        headline = markdown.split("## Needs review")[0]
        self.assertIn("App crashed", headline)
        self.assertNotIn("Maybe a dead tap", headline)

    def test_no_needs_review_section_when_all_confirmed(self) -> None:
        """
        A report with only confirmed defects renders no needs-review section.
        """

        defect = Defect.from_signal(
            signal=DefectSignal.CRASH,
            source=DefectSource.INLINE,
            summary="App crashed",
            evidence=DefectEvidence(screen="home"),
        )
        report = DefectAggregator().build(defects=[defect], metadata=self.__metadata())

        markdown = BugReportRenderer().render(report=report)

        self.assertNotIn("## Needs review", markdown)
        self.assertNotIn("- Needs review:", markdown)


if __name__ == "__main__":
    unittest.main()
