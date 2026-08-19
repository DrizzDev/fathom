from __future__ import annotations

from typing import List, TextIO

from governance.schemas.debt import DebtRecord
from governance.schemas.report import Audit


class Reporter:
    """
    Renders an audit as a human-readable summary onto an injected stream.
    """

    def __init__(self, *, stream: TextIO) -> None:
        """
        Bind the output stream the summary is written to.
        """

        self.__stream = stream

    def render(self, *, audit: Audit) -> None:
        """
        Write the mode, taxonomy state, per-disposition counts, and every actionable item.
        """

        report = audit.report
        self.__write(text=f"mode: {audit.mode.value}")
        self.__write(text=f"taxonomy: {'PROVISIONAL' if audit.provisional else 'APPROVED'}")
        self.__write(
            text=(
                f"known: {len(report.known)}  new: {len(report.new)}  "
                f"stale: {len(report.stale)}  expiring: {len(report.expiring)}  "
                f"expired: {len(report.expired)}  duplicate: {len(report.duplicate)}  "
                f"invalid: {len(report.invalid)}"
            )
        )
        for violation in report.new:
            selector = violation.selector
            self.__write(
                text=f"  NEW       [{selector.rule.value}] {selector.path}:{violation.line} {violation.message}"
            )
        self.__records(label="STALE", records=report.stale)
        self.__records(label="EXPIRING", records=report.expiring)
        self.__records(label="EXPIRED", records=report.expired)
        self.__records(label="DUPLICATE", records=report.duplicate)
        self.__records(label="INVALID", records=report.invalid)

    def __records(self, *, label: str, records: List[DebtRecord]) -> None:
        """
        Write one line per debt record under a disposition label.
        """

        for record in records:
            self.__write(
                text=f"  {label:9s} {record.reference} {record.selector.path} :: {record.selector.detail}"
            )

    def __write(self, *, text: str) -> None:
        """
        Emit a single line to the bound stream.
        """

        self.__stream.write(f"{text}\n")
