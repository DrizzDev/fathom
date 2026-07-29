from __future__ import annotations

from enum import IntEnum, StrEnum


class ExitCode(IntEnum):
    """
    Process exit codes for the governance CLI.
    """

    OK = 0
    VIOLATIONS = 1


class ExpiryPolicy(IntEnum):
    """
    Expiry warning policy, expressed in calendar days before a record's expiry.
    """

    WARNING = 30


class GovernanceMode(StrEnum):
    """
    Rollout mode for the fitness ratchet; the single control shared by the CLI and the tests.
    """

    REPORT = "REPORT"
    RATCHET = "RATCHET"


class DebtState(StrEnum):
    """
    Governance state of a debt record.
    """

    BASELINE = "BASELINE"
    APPROVED = "APPROVED"


class Placeholder(StrEnum):
    """
    Reserved values marking a debt record as recorded but not yet governance-owned.
    """

    OWNER = "UNASSIGNED"
    TICKET = "TBD"


class RuleId(StrEnum):
    """
    Stable identifier for an architecture-fitness rule.

    Values are a persisted, rendered vocabulary (they key the debt manifest and appear
    in reports), so they keep their dotted literal form.
    """

    DOMAIN_PURITY = "domain.purity"
    DATACLASS_FORBIDDEN = "dataclass.forbidden"
