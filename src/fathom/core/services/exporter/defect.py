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
            self.__table(title="Defects", defects=report.defects),
            self.__table(
                title="Needs review",
                defects=report.needs_review,
                note="Uncorroborated signals held back from the headline; confirm before reporting.",
            ),
        ]
        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def __header(*, report: BugReport) -> str:
        """
        Renders the title and run metadata.
        """

        metadata = report.metadata
        lines = [
            "# Bug Report",
            "",
            f"- Workflow: {metadata.workflow}",
            f"- Package: {metadata.package}",
            f"- Generated: {metadata.generated_at}",
            f"- Defects: {len(report.defects)}",
        ]
        if report.needs_review:
            lines.append(f"- Needs review: {len(report.needs_review)}")
        return "\n".join(lines)

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

    @classmethod
    def __table(cls, *, title: str, defects: List[Defect], note: str = "") -> str:
        """
        Renders a titled defect table, most severe first; empty when there are none.
        """

        if not defects:
            return ""

        heading = [f"## {title}", ""]
        if note:
            heading += [note, ""]
        return "\n".join(
            [
                *heading,
                "| Severity | Kind | Signal | Screen | Count | Summary |",
                "| --- | --- | --- | --- | --- | --- |",
                *cls.__rows(defects=defects),
            ]
        )

    @staticmethod
    def __rows(*, defects: List[Defect]) -> List[str]:
        """
        Renders one Markdown table row per defect.
        """

        return [
            f"| {defect.severity.value} | {defect.kind.value} | {defect.signal.value} | "
            f"{defect.evidence.screen} | {defect.occurrence} | {defect.summary} |"
            for defect in defects
        ]
