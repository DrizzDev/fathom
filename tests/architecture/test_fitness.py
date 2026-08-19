from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

from governance.checker import Checker
from governance.schemas.report import Audit


class FitnessTest(unittest.TestCase):
    """
    Consumes the single checked-in governance mode. In report mode this asserts the audit
    completes against a valid manifest without failing on findings; once governance flips the
    mode to ratchet, the same assertion enforces zero blocking dispositions.
    """

    def test_repository_audit_passes_active_mode(self) -> None:
        """
        The repository passes its own governance mode.
        """

        root = Path(__file__).resolve().parents[2]
        audit = Checker.audit(root=root, today=date.today())

        self.assertTrue(audit.passed(), self.__summary(audit=audit))

    @staticmethod
    def __summary(*, audit: Audit) -> str:
        """
        Render blocking dispositions as a diagnostic message for a failed assertion.
        """

        report = audit.report
        lines = [f"mode: {audit.mode.value}"]
        lines += [f"  NEW      {v.selector.path}:{v.line} {v.message}" for v in report.new]
        lines += [
            f"  STALE    {r.reference} {r.selector.path} :: {r.selector.detail}"
            for r in report.stale
        ]
        lines += [
            f"  EXPIRED  {r.reference} {r.selector.path} :: {r.selector.detail}"
            for r in report.expired
        ]
        lines += [
            f"  DUPLICATE {r.reference} {r.selector.path} :: {r.selector.detail}"
            for r in report.duplicate
        ]
        lines += [
            f"  INVALID  {r.reference} {r.selector.path} :: {r.selector.detail}"
            for r in report.invalid
        ]
        return "\n".join(lines)
