"""
Renders an aggregated bug report as Markdown.
"""

from __future__ import annotations

from typing import List

from fathom.schemas.defect import BugReport, Defect


class BugReportRenderer:
    """
    Renders an aggregated bug report as a Markdown document.
    """

    def render(self, *, report: BugReport) -> str:
        """
        Returns the bug report as a Markdown document.
        """

        sections = [
            self.__header(report=report),
            self.__summary(report=report),
            self.__defects(defects=report.defects),
        ]
        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def __header(*, report: BugReport) -> str:
        """
        Renders the title and run metadata.
        """

        metadata = report.metadata
        return "\n".join(
            [
                "# Bug Report",
                "",
                f"- Workflow: {metadata.workflow}",
                f"- Package: {metadata.package}",
                f"- Generated: {metadata.generated_at}",
                f"- Defects: {len(report.defects)}",
            ]
        )

    @staticmethod
    def __summary(*, report: BugReport) -> str:
        """
        Renders the per-severity and per-kind counts.
        """

        if not report.defects:
            return "No defects detected."

        severity = " · ".join(
            f"{severity.value}: {count}" for severity, count in report.by_severity.items()
        )
        kind = " · ".join(f"{kind.value}: {count}" for kind, count in report.by_kind.items())
        return "\n".join(["## Summary", "", f"- By severity: {severity}", f"- By kind: {kind}"])

    @staticmethod
    def __defects(*, defects: List[Defect]) -> str:
        """
        Renders the defect table, most severe first.
        """

        if not defects:
            return ""

        rows = [
            f"| {defect.severity.value} | {defect.kind.value} | {defect.signal.value} | "
            f"{defect.evidence.screen} | {defect.occurrence} | {defect.summary} |"
            for defect in defects
        ]
        return "\n".join(
            [
                "## Defects",
                "",
                "| Severity | Kind | Signal | Screen | Count | Summary |",
                "| --- | --- | --- | --- | --- | --- |",
                *rows,
            ]
        )
