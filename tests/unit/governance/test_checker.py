from __future__ import annotations

import subprocess
import tempfile
import unittest
from datetime import date
from pathlib import Path
from typing import List, Set

from governance.checker import Checker
from governance.constants import DebtState, GovernanceMode, RuleId
from governance.errors import GovernanceError
from governance.schemas.debt import DebtRecord
from governance.schemas.manifest import Manifest
from governance.schemas.selector import Selector
from governance.schemas.taxonomy import Taxonomy


class CheckerScanTest(unittest.TestCase):
    """
    Pins rule application over the git-reported first-party modules of a repository.
    """

    __DATACLASS = "from pydantic.dataclasses import dataclass\n\n\n@dataclass\nclass Sample:\n    x: int\n"

    @staticmethod
    def __repo(*, root: Path) -> None:
        """
        Initialize a git repository at the given root.
        """

        subprocess.run(["git", "init", "-q", str(root)], check=True)

    @staticmethod
    def __write(*, root: Path, relative: str, code: str) -> None:
        """
        Write a module at a repository-relative path.
        """

        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)

    def __checker(self) -> Checker:
        """
        Build the canonical checker with a minimal domain taxonomy.
        """

        return Checker.standard(taxonomy=Taxonomy(domain=("fathom.schemas",)))

    def __paths(self, *, root: Path) -> Set[str]:
        """
        Scan the repository and return the paths of all findings.
        """

        return {finding.selector.path for finding in self.__checker().scan(repo=root)}

    def test_scan_flags_domain_purity_in_taxonomy_scope(self) -> None:
        """
        A forbidden import inside the taxonomy domain scope is a purity finding.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.__repo(root=root)
            self.__write(root=root, relative="src/fathom/schemas/example.py", code="import logging\n")

            findings = self.__checker().scan(repo=root)

            self.assertEqual([finding.selector.rule for finding in findings], [RuleId.DOMAIN_PURITY])

    def test_scan_flags_pydantic_dataclass_across_first_party_roots(self) -> None:
        """
        The dataclass ban covers Pydantic dataclasses in governance and test modules alike.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.__repo(root=root)
            self.__write(root=root, relative="governance/tool.py", code=self.__DATACLASS)
            self.__write(root=root, relative="tests/unit/helper.py", code=self.__DATACLASS)

            paths = self.__paths(root=root)

            self.assertIn("governance/tool.py", paths)
            self.assertIn("tests/unit/helper.py", paths)


class CheckerAuditTest(unittest.TestCase):
    """
    Pins the audit path: fail-closed configuration, mode gating, and provisional-taxonomy blocking.
    """

    @staticmethod
    def __repository(*, root: Path, mode: GovernanceMode, provisional: bool, records: List[DebtRecord]) -> None:
        """
        Lay out a git repository with one domain violation and governance configuration.
        """

        subprocess.run(["git", "init", "-q", str(root)], check=True)

        package = root / "src" / "fathom" / "schemas"
        package.mkdir(parents=True)
        (package / "example.py").write_text("import logging\n")

        governance = root / "governance"
        governance.mkdir()
        (governance / "taxonomy.json").write_text(
            Taxonomy(provisional=provisional, domain=("fathom.schemas",)).model_dump_json()
        )
        (governance / "debt.json").write_text(Manifest(mode=mode, records=records).model_dump_json())

    @staticmethod
    def __record() -> DebtRecord:
        """
        Build an approved record accepting the example domain violation.
        """

        return DebtRecord(
            reference="ARCH-1",
            selector=Selector(
                rule=RuleId.DOMAIN_PURITY, path="src/fathom/schemas/example.py", detail="logging"
            ),
            owner="execution",
            ticket="FATHOM-1",
            reason="baseline",
            expires=date(2026, 12, 31),
            state=DebtState.APPROVED,
        )

    def test_missing_manifest_fails_closed(self) -> None:
        """
        A repository with no debt manifest raises rather than silently defaulting to report mode.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "governance").mkdir()
            (root / "governance" / "taxonomy.json").write_text(
                Taxonomy(domain=("fathom.schemas",)).model_dump_json()
            )

            with self.assertRaises(GovernanceError):
                Checker.audit(root=root, today=date(2026, 7, 1))

    def test_report_mode_passes_with_unaccepted_finding(self) -> None:
        """
        In report mode an unaccepted finding is surfaced as new but the audit passes.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.__repository(root=root, mode=GovernanceMode.REPORT, provisional=True, records=[])

            audit = Checker.audit(root=root, today=date(2026, 7, 1))

            self.assertEqual(len(audit.report.new), 1)
            self.assertTrue(audit.passed())

    def test_ratchet_mode_fails_with_unaccepted_finding(self) -> None:
        """
        In ratchet mode an unaccepted finding blocks the audit.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.__repository(root=root, mode=GovernanceMode.RATCHET, provisional=False, records=[])

            audit = Checker.audit(root=root, today=date(2026, 7, 1))

            self.assertFalse(audit.passed())

    def test_ratchet_blocked_by_provisional_taxonomy(self) -> None:
        """
        Ratchet mode cannot activate against a provisional taxonomy even with no findings.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.__repository(
                root=root, mode=GovernanceMode.RATCHET, provisional=True, records=[self.__record()]
            )

            audit = Checker.audit(root=root, today=date(2026, 7, 1))

            self.assertEqual(audit.report.new, [])
            self.assertTrue(audit.provisional)
            self.assertFalse(audit.passed())
