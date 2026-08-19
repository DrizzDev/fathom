from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, FrozenSet, List, Sequence, Set

from governance.constants import DebtState, GovernanceMode, RuleId
from governance.schemas.debt import DebtRecord
from governance.schemas.finding import Violation
from governance.schemas.manifest import Manifest
from governance.schemas.report import Report
from governance.schemas.selector import Selector


class Reconciler:
    """
    Reconciles current findings against the debt manifest one-to-one under the active mode.

    Waivable rules may be accepted one-to-one by an active record; a non-waivable rule's
    findings always remain new and a record targeting one is invalid. Lapsed and ungoverned
    records never consume a finding, so their findings resurface as new.
    """

    def reconcile(
        self,
        *,
        findings: Sequence[Violation],
        manifest: Manifest,
        today: date,
        waivable: FrozenSet[RuleId],
        warning: int,
    ) -> Report:
        """
        Partition findings and records into new, known, stale, expiring, expired, duplicate, and invalid.
        """

        invalid = [
            record
            for record in manifest.records
            if self.__invalid(record=record, mode=manifest.mode, waivable=waivable)
        ]
        invalid_refs = {record.reference for record in invalid}
        remaining = [record for record in manifest.records if record.reference not in invalid_refs]
        expired = [record for record in remaining if self.__expired(record=record, today=today)]
        expired_refs = {record.reference for record in expired}
        active = [record for record in remaining if record.reference not in expired_refs]
        expiring = [
            record
            for record in active
            if self.__expiring(record=record, today=today, warning=warning)
        ]

        new = [finding for finding in findings if finding.selector.rule not in waivable]
        waivable_findings = [finding for finding in findings if finding.selector.rule in waivable]

        grouped = self.__group(findings=waivable_findings)
        known: List[Violation] = []
        stale: List[DebtRecord] = []
        duplicate: List[DebtRecord] = []
        consumed: Set[Violation] = set()

        for record in active:
            matched = grouped.get(record.selector, [])
            if not matched:
                stale.append(record)
            elif len(matched) == 1:
                known.append(matched[0])
                consumed.add(matched[0])
            else:
                duplicate.append(record)
                consumed.update(matched)

        new.extend(finding for finding in waivable_findings if finding not in consumed)

        return Report(
            new=new,
            known=known,
            stale=stale,
            expiring=expiring,
            expired=expired,
            duplicate=duplicate,
            invalid=invalid,
        )

    @staticmethod
    def __group(*, findings: Sequence[Violation]) -> Dict[Selector, List[Violation]]:
        """
        Group findings by their selector.
        """

        grouped: Dict[Selector, List[Violation]] = defaultdict(list)
        for finding in findings:
            grouped[finding.selector].append(finding)
        return grouped

    @staticmethod
    def __invalid(*, record: DebtRecord, mode: GovernanceMode, waivable: FrozenSet[RuleId]) -> bool:
        """
        Whether a record fails governance: a non-waivable rule, baseline debt in ratchet, or an ungoverned approval.
        """

        if record.selector.rule not in waivable:
            return True

        if mode is GovernanceMode.RATCHET and record.state is DebtState.BASELINE:
            return True

        return record.state is DebtState.APPROVED and not record.governed()

    @staticmethod
    def __expired(*, record: DebtRecord, today: date) -> bool:
        """
        Whether a record's bounded exception has lapsed.
        """

        return record.expires is not None and record.expires < today

    @staticmethod
    def __expiring(*, record: DebtRecord, today: date, warning: int) -> bool:
        """
        Whether an active record is within its expiry warning window.
        """

        return record.expires is not None and record.expires <= today + timedelta(days=warning)
