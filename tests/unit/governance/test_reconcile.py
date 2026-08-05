from __future__ import annotations

import unittest
from datetime import date, timedelta
from typing import FrozenSet, List

from governance.constants import DebtState, GovernanceMode, RuleId
from governance.reconcile import Reconciler
from governance.schemas.debt import DebtRecord
from governance.schemas.finding import Violation
from governance.schemas.manifest import Manifest
from governance.schemas.report import Report
from governance.schemas.selector import Selector


class ReconcilerTest(unittest.TestCase):
    """
    Pins one-to-one reconciliation, non-waivable enforcement, and the expiry warning window.
    """

    __TODAY = date(2026, 7, 1)
    __WARNING = 30
    __WAIVABLE: FrozenSet[RuleId] = frozenset({RuleId.DOMAIN_PURITY})
    __PATH = "src/fathom/schemas/state.py"

    @classmethod
    def __selector(
        cls, *, rule: RuleId = RuleId.DOMAIN_PURITY, detail: str = "logging"
    ) -> Selector:
        """
        Build a selector fixture.
        """

        return Selector(rule=rule, path=cls.__PATH, detail=detail)

    @classmethod
    def __violation(
        cls, *, rule: RuleId = RuleId.DOMAIN_PURITY, line: int = 3, detail: str = "logging"
    ) -> Violation:
        """
        Build a finding fixture.
        """

        return Violation(
            selector=cls.__selector(rule=rule, detail=detail), line=line, message="message"
        )

    @classmethod
    def __record(
        cls,
        *,
        reference: str = "ARCH-1",
        rule: RuleId = RuleId.DOMAIN_PURITY,
        detail: str = "logging",
        **overrides: object,
    ) -> DebtRecord:
        """
        Build a record fixture matching the finding fixture by default.
        """

        fields: dict[str, object] = {
            "reference": reference,
            "selector": cls.__selector(rule=rule, detail=detail),
            "owner": "execution",
            "ticket": "FATHOM-1",
            "reason": "baseline",
        }
        fields.update(overrides)
        return DebtRecord.model_validate(fields)

    def __reconcile(
        self,
        *,
        findings: List[Violation],
        records: List[DebtRecord],
        mode: GovernanceMode = GovernanceMode.REPORT,
    ) -> Report:
        """
        Reconcile findings against a manifest as of the fixed test date.
        """

        return Reconciler().reconcile(
            findings=findings,
            manifest=Manifest(mode=mode, records=records),
            today=self.__TODAY,
            waivable=self.__WAIVABLE,
            warning=self.__WARNING,
        )

    def test_matched_waivable_record_is_known(self) -> None:
        """
        A waivable finding covered one-to-one by a record is known, not new.
        """

        report = self.__reconcile(findings=[self.__violation()], records=[self.__record()])

        self.assertEqual(len(report.known), 1)
        self.assertEqual(report.new, [])

    def test_removed_debt_becomes_stale(self) -> None:
        """
        A record whose finding no longer exists is stale and must be removed.
        """

        report = self.__reconcile(findings=[], records=[self.__record()])

        self.assertEqual([record.reference for record in report.stale], ["ARCH-1"])

    def test_one_record_covering_two_findings_is_duplicate(self) -> None:
        """
        A single record cannot accept two findings with the same selector.
        """

        findings = [self.__violation(line=3), self.__violation(line=9)]
        report = self.__reconcile(findings=findings, records=[self.__record()])

        self.assertEqual([record.reference for record in report.duplicate], ["ARCH-1"])
        self.assertEqual(report.new, [])

    def test_expired_record_is_reported_and_finding_resurfaces(self) -> None:
        """
        A lapsed record is expired, and the finding it used to cover resurfaces as new.
        """

        lapsed = self.__record(expires=date(2020, 1, 1), state=DebtState.APPROVED)
        report = self.__reconcile(findings=[self.__violation()], records=[lapsed])

        self.assertEqual([record.reference for record in report.expired], ["ARCH-1"])
        self.assertEqual(len(report.new), 1)

    def test_record_inside_warning_window_is_expiring_not_blocking(self) -> None:
        """
        An active record within the expiry warning window is expiring; it still covers its finding.
        """

        soon = self.__record(expires=self.__TODAY + timedelta(days=10), state=DebtState.APPROVED)
        report = self.__reconcile(findings=[self.__violation()], records=[soon])

        self.assertEqual([record.reference for record in report.expiring], ["ARCH-1"])
        self.assertEqual(len(report.known), 1)
        self.assertFalse(report.blocking())

    def test_baseline_record_is_invalid_in_ratchet_mode(self) -> None:
        """
        Unapproved baseline debt cannot exist in ratchet mode; it is invalid and blocking.
        """

        report = self.__reconcile(
            findings=[self.__violation()], records=[self.__record()], mode=GovernanceMode.RATCHET
        )

        self.assertEqual([record.reference for record in report.invalid], ["ARCH-1"])
        self.assertTrue(report.blocking())

    def test_nonwaivable_finding_is_always_new(self) -> None:
        """
        A dataclass finding is never accepted, even with a matching approved record.
        """

        finding = self.__violation(rule=RuleId.DATACLASS_FORBIDDEN, detail="Point")
        record = self.__record(
            rule=RuleId.DATACLASS_FORBIDDEN,
            detail="Point",
            state=DebtState.APPROVED,
            expires=date(2026, 12, 31),
        )
        report = self.__reconcile(findings=[finding], records=[record])

        self.assertEqual(len(report.new), 1)
        self.assertEqual(report.known, [])

    def test_nonwaivable_record_is_invalid(self) -> None:
        """
        A debt record targeting the non-waivable dataclass rule is invalid and blocking.
        """

        record = self.__record(rule=RuleId.DATACLASS_FORBIDDEN, detail="Point")
        report = self.__reconcile(findings=[], records=[record])

        self.assertEqual([record.reference for record in report.invalid], ["ARCH-1"])
        self.assertTrue(report.blocking())
