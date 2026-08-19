from __future__ import annotations

import unittest
from datetime import date

from pydantic import ValidationError

from governance.constants import DebtState, Placeholder, RuleId
from governance.schemas.debt import DebtRecord
from governance.schemas.manifest import Manifest
from governance.schemas.selector import Selector


class DebtRecordTest(unittest.TestCase):
    """
    Pins DebtRecord governance completeness.
    """

    @staticmethod
    def __approved(**overrides: object) -> DebtRecord:
        """
        Build a fully governed approved record, applying overrides.
        """

        fields: dict[str, object] = {
            "reference": "ARCH-PURE-001",
            "selector": Selector(
                rule=RuleId.DOMAIN_PURITY, path="src/fathom/schemas/state.py", detail="logging"
            ),
            "owner": "execution",
            "ticket": "FATHOM-1",
            "reason": "pending migration",
            "expires": date(2026, 12, 31),
            "state": DebtState.APPROVED,
        }
        fields.update(overrides)
        return DebtRecord.model_validate(fields)

    def test_approved_record_with_full_ownership_is_governed(self) -> None:
        """
        An approved record with a real owner, ticket, and expiry is governed.
        """

        self.assertTrue(self.__approved().governed())

    def test_approved_record_with_placeholder_owner_is_not_governed(self) -> None:
        """
        An approved record still carrying a placeholder owner is not governed.
        """

        self.assertFalse(self.__approved(owner=Placeholder.OWNER.value).governed())

    def test_approved_record_without_expiry_is_not_governed(self) -> None:
        """
        An approved record with no expiry is not governed.
        """

        self.assertFalse(self.__approved(expires=None).governed())

    def test_baseline_record_is_never_governed(self) -> None:
        """
        A baseline record is never governed regardless of its fields.
        """

        self.assertFalse(self.__approved(state=DebtState.BASELINE).governed())

    def test_empty_required_field_is_rejected(self) -> None:
        """
        A record with an empty required field fails validation.
        """

        with self.assertRaises(ValidationError):
            self.__approved(owner="")


class ManifestTest(unittest.TestCase):
    """
    Pins Manifest uniqueness validation and fail-closed loading.
    """

    @staticmethod
    def __record(*, reference: str, detail: str) -> DebtRecord:
        """
        Build a baseline record fixture with the given reference and selector detail.
        """

        return DebtRecord(
            reference=reference,
            selector=Selector(
                rule=RuleId.DOMAIN_PURITY, path="src/fathom/schemas/state.py", detail=detail
            ),
            owner="o",
            ticket="t",
            reason="r",
        )

    def test_duplicate_reference_is_rejected(self) -> None:
        """
        Two records sharing a reference make the manifest invalid.
        """

        with self.assertRaises(ValidationError):
            Manifest(
                records=[
                    self.__record(reference="ARCH-1", detail="logging"),
                    self.__record(reference="ARCH-1", detail="numpy"),
                ]
            )

    def test_duplicate_selector_is_rejected(self) -> None:
        """
        Two records targeting the same selector make the manifest invalid.
        """

        with self.assertRaises(ValidationError):
            Manifest(
                records=[
                    self.__record(reference="ARCH-1", detail="logging"),
                    self.__record(reference="ARCH-2", detail="logging"),
                ]
            )

    def test_distinct_records_are_accepted(self) -> None:
        """
        Records with distinct references and selectors are valid.
        """

        manifest = Manifest(
            records=[
                self.__record(reference="ARCH-1", detail="logging"),
                self.__record(reference="ARCH-2", detail="numpy"),
            ]
        )

        self.assertEqual(len(manifest.records), 2)
