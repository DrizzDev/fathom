from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from governance.checker import Checker
from governance.constants import ExitCode
from governance.report import Reporter


class Cli:
    """
    Command-line entry point for the architecture-fitness checker.

    The rollout mode lives in the checked-in manifest, not a flag: report mode always exits 0,
    ratchet mode exits non-zero on any blocking disposition.
    """

    @classmethod
    def run(cls) -> int:
        """
        Run the audit against the repository, render it, and return the process exit code.
        """

        root = Path(__file__).resolve().parents[1]
        audit = Checker.audit(root=root, today=date.today())

        Reporter(stream=sys.stdout).render(audit=audit)

        return ExitCode.OK.value if audit.passed() else ExitCode.VIOLATIONS.value


raise SystemExit(Cli.run())
